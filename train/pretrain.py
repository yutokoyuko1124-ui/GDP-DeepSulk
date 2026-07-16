#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
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


def xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def resolve_device(requested: str) -> str:
    requested = requested.lower()

    if requested == "auto":
        if xpu_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested == "xpu":
        if xpu_available():
            return "xpu"
        print("XPU requested but not available. Falling back to CPU.")
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    raise ValueError(f"unsupported device: {requested}")


def resolve_precision(requested: str, device_type: str) -> tuple[str, torch.dtype | None, bool]:
    requested = requested.lower()

    if requested == "auto":
        if device_type == "xpu":
            return "bf16", torch.bfloat16, True
        if device_type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bf16", torch.bfloat16, True
            return "fp16", torch.float16, True
        return "fp32", None, False

    if requested == "bf16":
        if device_type == "xpu":
            return "bf16", torch.bfloat16, True
        if device_type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bf16", torch.bfloat16, True
            print("BF16 requested but this CUDA device does not report BF16 support. Falling back to FP32.")
            return "fp32", None, False
        print("BF16 requested on CPU. Falling back to FP32 for maximum compatibility.")
        return "fp32", None, False

    if requested == "fp16":
        if device_type == "cuda":
            return "fp16", torch.float16, True
        print("FP16 requested without CUDA. Falling back to FP32.")
        return "fp32", None, False

    if requested == "fp32":
        return "fp32", None, False

    raise ValueError(f"unsupported precision: {requested}")


def get_device_name(device_type: str) -> str:
    try:
        if device_type == "xpu":
            return torch.xpu.get_device_name(0)
        if device_type == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GDP-DeepSulk tiny GPT from scratch.")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default=None,
                        help="Override config device. auto prefers XPU, then CUDA, then CPU.")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default=None,
                        help="Override config precision. XPU auto uses BF16; CPU auto uses FP32.")
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

    requested_device = args.device or str(cfg.get("device", "auto"))
    requested_precision = args.precision or str(cfg.get("precision", "auto"))
    device_type = resolve_device(requested_device)
    precision, amp_dtype, amp_enabled = resolve_precision(requested_precision, device_type)
    device = torch.device(device_type)

    # FP16 needs gradient scaling. BF16 normally does not.
    use_grad_scaler = device_type == "cuda" and precision == "fp16"
    scaler = None
    if use_grad_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    def autocast_context():
        if not amp_enabled or amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=amp_dtype)

    def synchronize() -> None:
        if device_type == "xpu":
            torch.xpu.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()

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

    cfg["resolved_device"] = device_type
    cfg["resolved_precision"] = precision

    def save_checkpoint(path: Path, it: int, best_val_loss: float) -> None:
        synchronize()
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.__dict__,
            "train_config": cfg,
            "iter": it,
            "best_val_loss": best_val_loss,
        }
        if scaler is not None:
            ckpt["grad_scaler"] = scaler.state_dict()
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
            if scaler is not None and "grad_scaler" in ckpt:
                scaler.load_state_dict(ckpt["grad_scaler"])
            start_iter = int(ckpt.get("iter", -1)) + 1
            best_val = float(ckpt.get("best_val_loss", best_val))
            print(f"resumed {resume_path} from iter {start_iter - 1}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device_type}")
    print(f"device_name={get_device_name(device_type)}")
    print(f"precision={precision}")
    print(f"amp_enabled={amp_enabled}")
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
                with autocast_context():
                    _, loss = model(X, Y)
                losses[k] = loss.float().item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    model.train()

    for it in range(start_iter, max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            synchronize()
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
            with autocast_context():
                _, loss = model(X, Y)
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.detach().float().item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if not math.isfinite(loss_accum):
            raise RuntimeError(f"non-finite loss at iter {it}: {loss_accum}")

        if it % log_interval == 0:
            synchronize()
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
