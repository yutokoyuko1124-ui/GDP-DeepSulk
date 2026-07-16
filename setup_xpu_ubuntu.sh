#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch-xpu.txt

chmod +x run_xpu_bf16.sh setup_xpu_ubuntu.sh

python scripts/check_xpu.py

echo
echo "Setup complete."
echo "Put train.bin, valid.bin and their .meta.json files in data/."
echo "Put gdp_deepsulk_tokenizer.json in tokenizer/."
echo "Then run: source .venv/bin/activate && ./run_xpu_bf16.sh"
