#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained GDP-DeepSulk checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bos_id = tokenizer.token_to_id("<|bos|>")
    prompt_ids = tokenizer.encode(args.prompt).ids
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    if not prompt_ids:
        prompt_ids = [0]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    ids = y[0].tolist()
    text = tokenizer.decode(ids)
    print(text)


if __name__ == "__main__":
    main()
