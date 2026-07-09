#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig


def decode_ids(tokenizer: Tokenizer, ids: list[int], skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained GDP-DeepSulk checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1, help="Generate multiple samples in one batched forward loop.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Values above 1.0 reduce repeated tokens.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Ban repeated ngrams of this size. 0 disables it.")
    parser.add_argument("--stop-at-eos", action="store_true", help="Stop generation when <|eos|> is sampled.")
    parser.add_argument("--show-special-tokens", action="store_true", help="Do not hide special tokens during decode.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    eos_id = tokenizer.token_to_id("<|eos|>") if args.stop_at_eos else None
    prompt_ids = tokenizer.encode(args.prompt).ids
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    if not prompt_ids:
        prompt_ids = [0]

    x = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_id,
        )

    for i, row in enumerate(y.tolist(), start=1):
        if args.num_samples > 1:
            print(f"--- sample {i} ---")
        text = decode_ids(tokenizer, row, skip_special_tokens=not args.show_special_tokens)
        print(text)
        if i != args.num_samples:
            print()


if __name__ == "__main__":
    main()
