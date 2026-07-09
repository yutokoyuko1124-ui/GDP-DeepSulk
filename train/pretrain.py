#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

# Allow running as: python train/pretrain.py
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.gpt import GPT, GPTConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dtype_from_meta(bin_path: str) -> np.dtype:
    meta_path = bin_path + ".meta.json"
    if not os.path.exists(meta_path):
        return np.uint16
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return np.dtype(meta.get("dtype", "uint16"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GDP-DeepSulk tiny GPT from scratch.")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/tiny.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg["vocab_size"] = tokenizer.get_vocab_size()

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    block_size = int(cfg["block_size"])
    batch_size = int(cfg["batch_size"])
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    learning_rate = float(cfg["learning_rate"])
    max_iters = int(cfg["max_iters"])
    eval_interval = int(cfg["eval_interval"])
    eval_iters = int(cfg["eval_iters"])

    train_dtype = get_dtype_from_meta(args.train_bin)
    valid_dtype = get_dtype_from_meta(args.valid_bin)
    train_data = np.memmap(args.train_bin, dtype=train_dtype, mode="r")
    valid_data = np.memmap(args.valid_bin, dtype=valid_dtype, mode="r")

    if len(train_data) < block_size + 2:
        raise RuntimeError("train dataset is too small for block_size")
    if len(valid_data) < block_size + 2:
        raise RuntimeError("valid dataset is too small for block_size")

    def get_batch(split: str):
        data = train_data if split == "train" else valid_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i+block_size], dtype=np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.asarray(data[i+1:i+1+block_size], dtype=np.int64)) for i in ix])
        return x.to(device), y.to(device)

    model_config = GPTConfig(
        vocab_size=int(cfg["vocab_size"]),
        block_size=block_size,
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"params={n_params/1e6:.2f}M")
    print(f"train_tokens={len(train_data):,}")
    print(f"valid_tokens={len(valid_data):,}")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "valid"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    best_val = float("inf")
    t0 = time.time()
    model.train()

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"iter {it}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}, time {elapsed:.1f}s")
            if losses["valid"] < best_val:
                best_val = losses["valid"]
                ckpt = {
                    "model": model.state_dict(),
                    "model_config": model_config.__dict__,
                    "train_config": cfg,
                    "iter": it,
                    "best_val_loss": best_val,
                }
                torch.save(ckpt, out_dir / "ckpt.pt")
                print(f"saved {out_dir / 'ckpt.pt'}")

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            X, Y = get_batch("train")
            _, loss = model(X, Y)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if not math.isfinite(loss_accum):
            raise RuntimeError(f"non-finite loss at iter {it}: {loss_accum}")

    print("done")


if __name__ == "__main__":
    main()
