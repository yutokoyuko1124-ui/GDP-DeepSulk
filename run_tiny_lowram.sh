#!/usr/bin/env bash
set -e

mkdir -p data tokenizer checkpoints/tiny_lowram

python scripts/split_jsonl.py \
  --input data/aozora_train.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --valid-ratio 0.02

# Create a tokenizer sample spread across the whole dataset to avoid OOM.
# BPE training itself is RAM-heavy, so we train the tokenizer on a representative sample.
python scripts/make_tokenizer_sample.py \
  --input data/train.jsonl \
  --output data/tokenizer_sample.jsonl \
  --max-lines 12000

python scripts/train_tokenizer.py \
  --input data/tokenizer_sample.jsonl \
  --output tokenizer/gdp_deepsulk_tokenizer.json \
  --vocab-size 4000 \
  --min-frequency 5

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
  --out-dir checkpoints/tiny_lowram \
  --config configs/tiny_lowram.json
