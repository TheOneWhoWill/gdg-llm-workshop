import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

MODEL_DIR = "./model"
TOKENIZER_DIR = "./tokenizer"

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("Model folder not found. Train the model first.")

if not os.path.exists(TOKENIZER_DIR):
    raise FileNotFoundError("Tokenizer folder not found. Build it first.")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = LlamaForCausalLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

print("Type 'q' to quit.\n")

while True:
    prompt = input("Enter some text: ").strip()
    if prompt.lower() in {"q", "quit", "exit"}:
        print("Goodbye!")
        break

    try:
        num_tokens = int(input("How many tokens to generate? ").strip())
    except ValueError:
        print("Please enter a valid integer.\n")
        continue

    if num_tokens <= 0:
        print("Please enter a positive number.\n")
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    print("\nPrompt:", prompt)
    print(f"Generated {num_tokens} tokens:")
    print(generated_text)
    print("\n" + "-" * 60 + "\n")