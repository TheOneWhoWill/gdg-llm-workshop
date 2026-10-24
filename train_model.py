import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

import constants

bin_path = "data.bin"
seq_len = constants.seq_len
batch_size = 16
vocab_size = constants.vocab_size
epochs = constants.epochs

if not os.path.exists(bin_path):
    print("First process your dataset using process_dataset.py ")

# Load the dataset and prepare it for training
raw_tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
num_samples = (len(raw_tokens) - 1) // seq_len

# Split the dataset to have inputs and associated outputs. N tokens -> N + 1 token
x_data = raw_tokens[: num_samples * seq_len].reshape(-1, seq_len).astype(np.int64)
y_data = raw_tokens[1 : num_samples * seq_len + 1].reshape(-1, seq_len).astype(np.int64)

dataset = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# We're using a llama architecture because it's dense and that works better
# for small things like this. Feel free to increase any parameters here or
# in the other files.
config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=constants.hidden_size,
    intermediate_size=constants.intermediate_size,
    num_hidden_layers=constants.num_hidden_layers,
    num_attention_heads=constants.num_attention_heads,
    max_position_embeddings=constants.max_position_embeddings,
)
model = LlamaForCausalLM(config)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
num_training_steps = epochs * len(loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, num_training_steps // 10),
    num_training_steps=num_training_steps,
)

# 3. Functional training loop
print(f"Training ~{model.num_parameters() / 1e6:.2f}M parameter model on {device} for {epochs} epoch(s)...")
model.train()
step = 0

# Training loop. We have to do this manually because we're not using a Trainer class
for epoch in range(epochs):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Hugging Face models calculate loss automatically when labels are provided
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        if step % 25 == 0 or step == 1:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Step {step:4d}/{num_training_steps} | Loss: {loss.item():.4f}")

print("Training complete.")

# Save using standard Hugging Face export
model.save_pretrained("./model")
print("Model saved to ./model")