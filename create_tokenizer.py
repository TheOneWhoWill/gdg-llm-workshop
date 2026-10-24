from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from pathlib import Path

import constants

VOCAB_SIZE=constants.vocab_size
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TXT_FILES = sorted(DATASET_DIR.glob("*.txt"))

tokenizer = Tokenizer(models.BPE(unk_token=None))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<bos>", "<eos>"],
    min_frequency=2 # Minimum frequency for a token to be included in the vocabulary
)

print("Training tokenizer on dataset...")
tokenizer.train(files=[str(path) for path in TXT_FILES], trainer=trainer)

test_text = "Space test:   3 spaces! Hello World! Welcome to Vice City!"
    
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids)

print("\n--- Tokenizer Test ---")
print("Original Text: ", test_text)
print("Token IDs:     ", encoded.ids)
print("Tokens:        ", encoded.tokens)
print("Decoded Back:  ", decoded)

Path("tokenizer").mkdir(parents=True, exist_ok=True)
tokenizer.save("./tokenizer/tokenizer.json")