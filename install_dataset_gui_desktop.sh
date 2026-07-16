#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
DESKTOP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/gdp-deepsulk-dataset-builder.desktop"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo ".venv がありません。先に ./setup_dataset_gui_ubuntu.sh を実行してください。" >&2
  exit 1
fi

mkdir -p "${DESKTOP_DIR}"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=GDP-DeepSulk Dataset Builder
Comment=RealPersonaChatとWikipediaを取得・前処理・洗濯するGUI
Exec=${ROOT}/run_dataset_gui.sh
Path=${ROOT}
Icon=applications-science
Terminal=false
Categories=Development;Science;Education;
EOF
chmod +x "${DESKTOP_FILE}"

echo "アプリケーションメニューへ登録しました: ${DESKTOP_FILE}"
