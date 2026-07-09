# GDP-DeepSulk: 「非力なPC」のためのAI

**GDP-DeepSulk** は、「AIを動かすには非力すぎる」と言われがちな環境でも、実用的な言語モデルを動作させることを目的としたローカルLLM実験プロジェクトです。

PCが重い処理に悲鳴を上げるような環境でも、負荷をできるだけ抑え、限界ギリギリのハードウェアでAIを動かす楽しさを追求します。

---

## 目的

最新のAIモデルが求める高いスペックと、現実のハードウェア環境とのギャップを埋めること。

GPUを搭載していないPCや、限られたメモリ環境でも、実用性をなるべく損なわずにAIを体験できる構成を目指します。

### 対象環境

- **CPU:** Ryzen 3 相当、またはそれ以下の古いCPU
- **メモリ:** システム全体で 8GB RAM 程度
- **計算資源:** CPUのみ（GPUオフロードなし）
- **モデル:** 完全自作 tokenizer + 完全自作 decoder-only Transformer
- **データ:** ローカルに保存した JSONL 学習データ

---

## ネーミングの由来

- **GDP / Gross Domestic Product / 国内総生産**  
  PCの「計算能力」を経済規模に見立てたものです。スペックが低い状態を、GDPが低い状態として表現しています。

- **DeepSulk / 深くすねる**  
  GDPが低すぎて、架空の総理大臣がすねてしまう、というジョークです。

また、巨大なLLMを古いノートPCで無理やり動かそうとして、PCもユーザーもAIも一緒に「深くすねる」ような感覚も表しています。

---

## Dataset

GDP-DeepSulk uses a locally generated training dataset.

The full training dataset is **not included** in this repository.  
It is generated and stored locally on the developer's machine.

> このAIの学習用データは俺のPCにしかないぜ！

Expected JSONL format:

```jsonl
{"text": "吾輩は猫である。名前はまだ無い。"}
{"text": "ある日の暮方の事である。"}
```

---

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Put your local dataset

```bash
mkdir -p data
dd if=/path/to/aozora_train.jsonl of=data/aozora_train.jsonl bs=1M status=progress
```

通常コピーでOKです。`data/*.jsonl` は `.gitignore` に入っているので、GitHubには入りません。

### 3. Split train / valid

```bash
python scripts/split_jsonl.py \
  --input data/aozora_train.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --valid-ratio 0.02
```

### 4. Train tokenizer from scratch

```bash
python scripts/train_tokenizer.py \
  --input data/train.jsonl \
  --output tokenizer/gdp_deepsulk_tokenizer.json \
  --vocab-size 8000
```

### 5. Tokenize dataset

```bash
python scripts/tokenize_dataset.py \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --input data/train.jsonl \
  --output data/train.bin

python scripts/tokenize_dataset.py \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --input data/valid.jsonl \
  --output data/valid.bin
```

### 6. Pretrain Tiny model

Ryzen 3 / 8GB RAM 向けの軽量設定です。

```bash
python train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/tiny \
  --config configs/tiny.json
```

### 6.5. Fast low-RAM mode

`configs/tiny_lowram_fast.json` は、Ryzen 3 / 8GB RAM で長時間回すための速め設定です。

- `batch_size=4`
- `eval_interval=1000`
- `eval_iters=2`
- `eval_train_loss=false`
- `save_last_interval=500`
- `num_threads=8`
- `max_iters=1000000`

```bash
python train/pretrain.py \
  --train-bin data/train.bin \
  --valid-bin data/valid.bin \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --out-dir checkpoints/tiny_lowram \
  --config configs/tiny_lowram_fast.json \
  --resume
```

`--resume` は `last.pt` があればそこから再開します。ベストcheckpointから再開したい場合は `--resume-best` を使います。

### 7. Generate text

```bash
python infer/generate.py \
  --checkpoint checkpoints/tiny/ckpt.pt \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --prompt "吾輩は" \
  --max-new-tokens 100
```

### 7.5. Generate more samples with anti-repeat

同じプロンプトから複数候補を一気に出します。

```bash
python infer/generate.py \
  --checkpoint checkpoints/tiny_lowram/ckpt.pt \
  --tokenizer tokenizer/gdp_deepsulk_tokenizer.json \
  --prompt "吾輩は猫である。" \
  --max-new-tokens 60 \
  --temperature 0.7 \
  --top-k 50 \
  --num-samples 4 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 3
```

---

## Project policy

This project starts from scratch:

- no pretrained model
- no pretrained tokenizer
- local JSONL dataset only
- tiny decoder-only Transformer
- CPU-friendly training settings

The first goal is not to beat ChatGPT.  
The first goal is:

> 自分で作った tokenizer と自分で作った model が、日本語っぽい文章を出すこと。

---

## Notice

GDP-DeepSulk is an independent local LLM experiment.

This project is not affiliated with OpenAI, DeepSeek, Microsoft, or any other AI company.  
This repository does not include proprietary model weights or the full private training dataset.
