#!/usr/bin/env bash
set -e

mkdir -p data tokenizer checkpoints/tiny_lowram

python scripts/split_jsonl.py \
  --input data/aozora_train.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --valid-ratio 0.02

# Train tokenizer on a sample to avoid OOM on 8GB RAM machines.
# The full dataset is still used for language-model training after tokenization.
head -n 10000 data/train.jsonl > data/tokenizer_sample.jsonl

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
