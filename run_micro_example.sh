#!/usr/bin/env bash
set -e

python scripts/split_jsonl.py \
  --input data/aozora_train.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --valid-ratio 0.02

python scripts/train_tokenizer.py \
  --input data/train.jsonl \
  --output tokenizer/gdp_deepsulk_tokenizer.json \
  --vocab-size 8000

python scripts/tokenize_dataset.py \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --input data/train.jsonl \
  --output data/train.bin

python scripts/tokenize_dataset.py \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --input data/valid.jsonl \
  --output data/valid.bin

python train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/micro \
  --config configs/micro.json
