# Dataset Builder Qt GUI

GDP-DeepSulk用の学習データをUbuntu Desktop上で自動取得・前処理するQt GUIです。

## 対応する取得元

- RealPersonaChat
  - 公式リポジトリのZIPを取得
  - 対話JSONを読み込み、短い会話ブロックへ分割
  - 品質評価による除外、マスク記号を含む発話の除外、重複除外
- 日本語Wikipedia
  - Wikimedia公式の最新 `pages-articles` ダンプを取得
  - bzip2を展開しながらXMLをストリーミング処理
  - テンプレート、参照、表、カテゴリ、HTMLタグなどを簡易除去
  - 記事を指定文字数で分割し、短すぎる文章と重複を除外
- 洗濯
  - RealPersonaChatとWikipediaのJSONLを、指定した乱数シードで混合

## GUI機能

- 取得元・洗濯モードの選択
- 出力先の選択
- 取得件数、品質しきい値、分割文字数などの調整
- ダウンロードと変換の進捗バー
- リアルタイムログ
- キャンセル
- 出力フォルダーを開くボタン
- ライセンス表示とmanifestの自動生成

## aria2cダウンロード

GUIから開始するダウンロードは `aria2c` を優先して使用します。

- 1ファイルにつき最大8接続
- 中断後の続きから再開
- GUIへ受信容量、進捗率、速度を表示
- キャンセル時も途中ファイルと `.aria2` 制御ファイルを保持
- `aria2c` がない場合だけPython標準ダウンローダーへ自動退避

`setup_dataset_gui_ubuntu.sh` はUbuntuへ `aria2` を自動導入します。すでに起動している旧版GUIのダウンロード方式は実行途中では切り替わりません。完了後にGUIを再起動すると、次回からaria2cが使われます。

## Ubuntu Desktopで起動

```bash
git clone https://github.com/yutokoyuko1124-ui/GDP-DeepSulk.git
cd GDP-DeepSulk
chmod +x setup_dataset_gui_ubuntu.sh run_dataset_gui.sh install_dataset_gui_desktop.sh
./setup_dataset_gui_ubuntu.sh
./run_dataset_gui.sh
```

アプリケーションメニューへ登録する場合:

```bash
./install_dataset_gui_desktop.sh
```

## 出力

既定では `data/dataset_builder/` 以下へ出力します。

```text
data/dataset_builder/
├── raw/
│   ├── real-persona-chat-main.zip
│   ├── real-persona-chat/
│   └── jawiki-latest-pages-articles.xml.bz2
├── processed/
│   ├── realpersonachat.jsonl
│   ├── wikipedia.jsonl
│   └── mixed_dataset.jsonl
├── dataset_manifest.json
└── DATASET_LICENSE.md
```

JSONLはGDP-DeepSulkの既存スクリプトで読める形式です。

```jsonl
{"text":"話者1: こんにちは。\n話者2: こんにちは！","source":"RealPersonaChat"}
{"text":"人工知能\n人工知能は…","source":"Wikipedia"}
```

## 学習用ファイルへ変換

混合データをtrain/validへ分割します。

```bash
source .venv/bin/activate
python scripts/split_jsonl.py \
  --input data/dataset_builder/processed/mixed_dataset.jsonl \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --valid-ratio 0.02
```

その後、既存のtokenizer作成・tokenize・学習手順を利用できます。

## 容量について

日本語Wikipediaの全件ダンプは大容量です。最初はGUIの「Wikipedia最大レコード」を10万件程度にして動作確認することを推奨します。`0`を指定すると全件処理になります。

ダウンロード済みファイルは再利用されるため、2回目以降は再ダウンロードしません。途中で止まったaria2cダウンロードは次回起動時に続きから再開します。

## ライセンス

生成先には、取得URL、生成日時、前処理内容、件数を記録した `dataset_manifest.json` と、次の帰属表示を含む `DATASET_LICENSE.md` を自動生成します。

- RealPersonaChat: CC BY-SA 4.0
- Wikipedia: CC BY-SA 4.0、提供元 Wikimedia Foundationおよび各記事の執筆者
- 加工後データセット: CC BY-SA 4.0

公開・再配布時は、実際に利用したデータと各提供元の最新のライセンス・利用条件も確認してください。
