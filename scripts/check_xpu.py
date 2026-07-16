#!/usr/bin/env python3
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

has_xpu = hasattr(torch, "xpu")
print(f"torch.xpu exists: {has_xpu}")

xpu_ok = bool(has_xpu and torch.xpu.is_available())
print(f"XPU available: {xpu_ok}")

if xpu_ok:
    try:
        print(f"XPU device: {torch.xpu.get_device_name(0)}")
    except Exception as exc:
        print(f"XPU device name error: {exc}")

    a = torch.randn((256, 256), device="xpu", dtype=torch.bfloat16)
    b = torch.randn((256, 256), device="xpu", dtype=torch.bfloat16)
    c = a @ b
    torch.xpu.synchronize()
    print(f"BF16 matrix test: OK, dtype={c.dtype}, device={c.device}")
else:
    print("XPU is unavailable. GDP-DeepSulk will fall back to CUDA or CPU when device=auto.")
