#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|user|>",
    "<|assistant|>",
]


def iter_text_batches(paths: List[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Byte-level BPE tokenizer from JSONL text.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL file(s).")
    parser.add_argument("--output", required=True, help="Output tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iter_text_batches(args.input, args.batch_size),
        trainer=trainer,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))

    print(f"saved tokenizer: {out}")
    print(f"vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
