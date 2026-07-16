#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 が見つかりません。" >&2
  exit 1
fi

if ! python3 -m venv .venv; then
  echo "venv作成に失敗したため python3-venv を導入します。"
  sudo apt update
  sudo apt install -y python3-venv python3-pip
  rm -rf .venv
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-gui.txt

chmod +x run_dataset_gui.sh install_dataset_gui_desktop.sh

echo
echo "セットアップ完了。"
echo "起動: ./run_dataset_gui.sh"
