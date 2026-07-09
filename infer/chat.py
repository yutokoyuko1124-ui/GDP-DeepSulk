#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig


def load_model(checkpoint: str, tokenizer_path: str, device: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint, map_location=device)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CLI chat for GDP-DeepSulk.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    print("GDP-DeepSulk chat. exit / quit で終了。")

    while True:
        user = input("You> ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        prompt = f"<|user|>{user}\n<|assistant|>"
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
        out = tokenizer.decode(y[0].tolist())
        if "<|assistant|>" in out:
            out = out.split("<|assistant|>", 1)[-1]
        print("AI>", out.strip())


if __name__ == "__main__":
    main()
