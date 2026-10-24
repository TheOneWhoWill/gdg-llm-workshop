import struct
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
OUTPUT_PATH = ROOT / "data.bin"

if not TOKENIZER_PATH.exists():
    print(f"Tokenizer not found at {TOKENIZER_PATH}.")
    print("Run create_tokenizer.py first.")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
txt_files = sorted(DATASET_DIR.glob("*.txt"))

if not txt_files:
    print(f"No .txt files found in {DATASET_DIR}.")
    exit()

total_tokens = 0

with OUTPUT_PATH.open("wb") as out:
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text)

        for token_id in encoded.ids:
            if token_id > 65535:
                print(f"Token ID {token_id} exceeds 16-bit range for {txt_file.name}.")
                exit()
            out.write(struct.pack("<H", token_id))
            total_tokens += 1

print(f"Processed {len(txt_files)} files into {OUTPUT_PATH}")
print(f"Wrote {total_tokens} tokens ({total_tokens * 2} bytes)")
