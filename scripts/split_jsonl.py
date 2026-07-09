#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL into train/valid files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.valid).parent.mkdir(parents=True, exist_ok=True)

    total = train_count = valid_count = bad_count = 0

    with open(args.input, "r", encoding="utf-8") as src, \
         open(args.train, "w", encoding="utf-8") as train, \
         open(args.valid, "w", encoding="utf-8") as valid:
        for line in src:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    bad_count += 1
                    continue
            except json.JSONDecodeError:
                bad_count += 1
                continue

            if random.random() < args.valid_ratio:
                valid.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                train.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                train_count += 1

    print(f"total={total}")
    print(f"train={train_count}")
    print(f"valid={valid_count}")
    print(f"bad={bad_count}")


if __name__ == "__main__":
    main()
