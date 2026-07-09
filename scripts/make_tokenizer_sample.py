#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a low-RAM tokenizer sample spread across the whole JSONL dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    reservoir: list[str] = []
    total = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(reservoir) < args.max_lines:
                reservoir.append(line)
            else:
                j = random.randint(1, total)
                if j <= args.max_lines:
                    reservoir[j - 1] = line

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for line in reservoir:
            out.write(line + "\n")

    print(f"input_lines={total}")
    print(f"sample_lines={len(reservoir)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
