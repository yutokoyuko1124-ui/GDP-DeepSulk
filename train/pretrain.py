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
    parser.add_argument("--resume", action="store_true", help="Resume from out_dir/last.pt if it exists, otherwise out_dir/ckpt.pt.")
    parser.add_argument("--resume-best", action="store_true", help="Resume from out_dir/ckpt.pt even when last.pt exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg["vocab_size"] = tokenizer.get_vocab_size()

    num_threads = int(cfg.get("num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    interop_threads = int(cfg.get("interop_threads", 0))
    if interop_threads > 0:
        torch.set_num_interop_threads(interop_threads)

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
    log_interval = int(cfg.get("log_interval", eval_interval))
    eval_train_loss = bool(cfg.get("eval_train_loss", True))
    save_last_interval = int(cfg.get("save_last_interval", eval_interval))

    train_dtype = get_dtype_from_meta(args.train_bin)
    valid_dtype = get_dtype_from_meta(args.valid_bin)
    train_data = np.memmap(args.train_bin, dtype=train_dtype, mode="r")
    valid_data = np.memmap(args.valid_bin, dtype=valid_dtype, mode="r")

    if len(train_data) < block_size + 2:
        raise RuntimeError("train dataset is too small for block_size")
    if len(valid_data) < block_size + 2:
        raise RuntimeError("valid dataset is too small for block_size")

    offsets = np.arange(block_size, dtype=np.int64)

    def get_batch(split: str):
        data = train_data if split == "train" else valid_data
        ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,), dtype=np.int64)
        x_np = np.asarray(data[ix[:, None] + offsets], dtype=np.int64)
        y_np = np.asarray(data[ix[:, None] + offsets + 1], dtype=np.int64)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
    best_ckpt_path = out_dir / "ckpt.pt"
    last_ckpt_path = out_dir / "last.pt"

    def save_checkpoint(path: Path, it: int, best_val_loss: float) -> None:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.__dict__,
            "train_config": cfg,
            "iter": it,
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, path)

    best_val = float("inf")
    start_iter = 0
    if args.resume or args.resume_best:
        if args.resume_best:
            resume_path = best_ckpt_path
        else:
            resume_path = last_ckpt_path if last_ckpt_path.exists() else best_ckpt_path
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = int(ckpt.get("iter", -1)) + 1
            best_val = float(ckpt.get("best_val_loss", best_val))
            print(f"resumed {resume_path} from iter {start_iter - 1}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"params={n_params/1e6:.2f}M")
    print(f"train_tokens={len(train_data):,}")
    print(f"valid_tokens={len(valid_data):,}")
    print(f"batch_size={batch_size}")
    print(f"grad_accum_steps={grad_accum_steps}")
    print(f"log_interval={log_interval}")
    print(f"eval_interval={eval_interval}")
    print(f"eval_iters={eval_iters}")
    print(f"eval_train_loss={eval_train_loss}")
    print(f"save_last_interval={save_last_interval}")
    print(f"torch_num_threads={torch.get_num_threads()}")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        splits = ["valid"] if not eval_train_loss else ["train", "valid"]
        for split in splits:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    model.train()

    for it in range(start_iter, max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            if eval_train_loss:
                print(f"iter {it}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}, time {elapsed:.1f}s")
            else:
                print(f"iter {it}: valid loss {losses['valid']:.4f}, time {elapsed:.1f}s")
            if losses["valid"] < best_val:
                best_val = losses["valid"]
                save_checkpoint(best_ckpt_path, it, best_val)
                print(f"saved best {best_ckpt_path}")

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

        if it % log_interval == 0:
            elapsed = time.time() - t0
            print(f"iter {it}: step train loss {loss_accum:.4f}, time {elapsed:.1f}s")

        if save_last_interval > 0 and it % save_last_interval == 0:
            save_checkpoint(last_ckpt_path, it, best_val)
            print(f"saved last {last_ckpt_path}")

    save_checkpoint(last_ckpt_path, max_iters, best_val)
    print(f"saved last {last_ckpt_path}")
    print("done")


if __name__ == "__main__":
    main()
