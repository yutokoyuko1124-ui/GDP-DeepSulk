#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize JSONL dataset into a flat .bin token file.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <|eos|>")

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size < 65536 else np.uint32

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_tokens = 0

    with open(args.input, "r", encoding="utf-8") as src, open(out, "wb") as dst:
        for line in tqdm(src, desc="tokenizing"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text).ids + [eos_id]
            arr = np.asarray(ids, dtype=dtype)
            arr.tofile(dst)
            n_docs += 1
            n_tokens += len(ids)

    meta = {
        "tokenizer": args.tokenizer,
        "input": args.input,
        "output": args.output,
        "dtype": str(np.dtype(dtype)),
        "vocab_size": vocab_size,
        "documents": n_docs,
        "tokens": n_tokens,
    }
    with open(str(out) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
