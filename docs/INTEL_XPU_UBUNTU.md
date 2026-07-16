# Intel Arc XPU BF16 on Ubuntu Desktop

Target example:

- Ubuntu Desktop
- Intel Core Ultra 7 255H
- Intel Arc Graphics (iGPU)
- PyTorch XPU
- BF16 mixed precision

## Clone and install

```bash
git clone https://github.com/yutokoyuko1124-ui/GDP-DeepSulk.git
cd GDP-DeepSulk
chmod +x setup_xpu_ubuntu.sh run_xpu_bf16.sh
./setup_xpu_ubuntu.sh
```

The setup script creates `.venv`, installs the base requirements and the PyTorch XPU wheel, then runs a BF16 matrix test on XPU.

Expected check output includes:

```text
XPU available: True
BF16 matrix test: OK
```

## Local files

The private training data and generated tokenizer are not stored in GitHub. Copy these files from the old training machine:

```text
data/train.bin
data/train.bin.meta.json
data/valid.bin
data/valid.bin.meta.json
tokenizer/gdp_deepsulk_tokenizer.json
```

To continue an existing run, also copy:

```text
checkpoints/tiny_lowram/last.pt
```

You can either place the checkpoint under `checkpoints/tiny_xpu/last.pt`, or start a new XPU run without it.

## Start training

```bash
cd GDP-DeepSulk
source .venv/bin/activate
./run_xpu_bf16.sh
```

With `device=auto` and `precision=auto`, runtime selection is:

```text
Intel XPU available -> XPU + BF16
CUDA available      -> CUDA + BF16 or FP16
Neither available   -> CPU + FP32
```

Expected startup output on Intel Arc:

```text
device=xpu
precision=bf16
amp_enabled=True
params=12.28M
```

## Manual override

Force XPU BF16:

```bash
python -u train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/tiny_xpu \
  --config configs/tiny_xpu_bf16.json \
  --device xpu \
  --precision bf16 \
  --resume
```

Force CPU FP32 for compatibility testing:

```bash
python -u train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/tiny_cpu_test \
  --config configs/tiny_xpu_bf16.json \
  --device cpu \
  --precision fp32
```
