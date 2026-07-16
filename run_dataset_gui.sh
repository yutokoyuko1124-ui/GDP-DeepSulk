#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo ".venv がありません。先に ./setup_dataset_gui_ubuntu.sh を実行してください。" >&2
  exit 1
fi

exec .venv/bin/python -m tools.dataset_builder_aria2 "$@"
