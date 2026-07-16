#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -u train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/tiny_xpu \
  --config configs/tiny_xpu_bf16.json \
  --device auto \
  --precision auto \
  --resume
