#!/usr/bin/env python3
from __future__ import annotations

import bz2
import hashlib
import html
import json
import random
import re
import shutil
import sys
import threading
import traceback
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from PySide6.QtCore import QObject, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "GDP-DeepSulk Dataset Builder"
REAL_PERSONA_ARCHIVE = (
    "https://github.com/nu-dialogue/real-persona-chat/"
    "archive/refs/heads/main.zip"
)
WIKIPEDIA_DUMP = (
    "https://dumps.wikimedia.org/jawiki/latest/"
    "jawiki-latest-pages-articles.xml.bz2"
)
USER_AGENT = "GDP-DeepSulk-Dataset-Builder/1.0"

SPACE_RE = re.compile(r"[ \t\u3000]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
REF_SINGLE_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
FILE_LINK_RE = re.compile(
    r"\[\[(?:File|Image|ファイル|画像):.*?\]\]",
    re.DOTALL | re.IGNORECASE,
)
CATEGORY_RE = re.compile(r"\[\[(?:Category|カテゴリ):.*?\]\]", re.IGNORECASE)
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
WIKI_LINK_SIMPLE_RE = re.compile(r"\[\[([^\]]+)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?://\S+)(?:\s+([^\]]+))?\]")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
BOLD_ITALIC_RE = re.compile(r"'{2,5}")
MASKED_NAME_RE = re.compile(r"<[A-Z]{2,}>")


class CancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class BuildConfig:
    mode: str
    output_dir: str
    real_url: str
    wiki_url: str
    max_real_records: int
    max_wiki_records: int
    real_turns_per_record: int
    min_real_quality: int
    skip_masked_real: bool
    wiki_chunk_chars: int
    wiki_min_chars: int
    mix_seed: int
    keep_source_files: bool


@dataclass(slots=True)
class BuildStats:
    real_records: int = 0
    wiki_records: int = 0
    mixed_records: int = 0
    real_dialogues_seen: int = 0
    wiki_pages_seen: int = 0
    skipped_records: int = 0


class DatasetBuilder:
    def __init__(
        self,
        config: BuildConfig,
        log: Callable[[str], None],
        progress: Callable[[int, str], None],
        cancel_event: threading.Event,
    ) -> None:
        self.cfg = config
        self.log = log
        self.progress = progress
        self.cancel_event = cancel_event
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.stats = BuildStats()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("処理をキャンセルしました。")

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        real_path: Path | None = None
        wiki_path: Path | None = None

        if self.cfg.mode in {"real", "mix"}:
            self.progress(2, "RealPersonaChatを準備中")
            real_path = self.build_real_persona()

        if self.cfg.mode in {"wiki", "mix"}:
            self.progress(42 if self.cfg.mode == "mix" else 5, "Wikipediaを準備中")
            wiki_path = self.build_wikipedia()

        final_path: Path
        if self.cfg.mode == "mix":
            assert real_path is not None and wiki_path is not None
            self.progress(88, "洗濯中：2つのデータを混合")
            final_path = self.merge_jsonl(real_path, wiki_path)
            if not self.cfg.keep_source_files:
                real_path.unlink(missing_ok=True)
                wiki_path.unlink(missing_ok=True)
        elif self.cfg.mode == "real":
            assert real_path is not None
            final_path = real_path
        else:
            assert wiki_path is not None
            final_path = wiki_path

        self.write_metadata(final_path)
        self.progress(100, "完了")
        return {
            "output": str(final_path),
            "stats": asdict(self.stats),
            "manifest": str(self.output_dir / "dataset_manifest.json"),
            "license": str(self.output_dir / "DATASET_LICENSE.md"),
        }

    def download(
        self,
        url: str,
        destination: Path,
        stage_start: int,
        stage_end: int,
    ) -> Path:
        if destination.exists() and destination.stat().st_size > 0:
            self.log(f"再利用: {destination}")
            return destination

        part = destination.with_suffix(destination.suffix + ".part")
        part.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        self.log(f"ダウンロード開始: {url}")

        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                part.open("wb") as dst,
            ):
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                while True:
                    self.check_cancel()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        ratio = min(downloaded / total, 1.0)
                        value = stage_start + int((stage_end - stage_start) * ratio)
                        self.progress(
                            value,
                            f"ダウンロード {downloaded / 2**20:.1f}/"
                            f"{total / 2**20:.1f} MiB",
                        )
                    elif downloaded % (32 * 1024 * 1024) < len(chunk):
                        self.log(f"受信済み: {downloaded / 2**20:.1f} MiB")
            part.replace(destination)
        except Exception:
            part.unlink(missing_ok=True)
            raise

        self.log(
            f"ダウンロード完了: {destination} "
            f"({destination.stat().st_size / 2**20:.1f} MiB)"
        )
        return destination

    def build_real_persona(self) -> Path:
        archive = self.download(
            self.cfg.real_url,
            self.raw_dir / "real-persona-chat-main.zip",
            3,
            18 if self.cfg.mode == "mix" else 30,
        )
        extract_root = self.raw_dir / "real-persona-chat"
        dataset_dir = self.find_real_dataset(extract_root)
        if dataset_dir is None:
            self.log("RealPersonaChat ZIPを展開中")
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_root)
            dataset_dir = self.find_real_dataset(extract_root)
        if dataset_dir is None:
            raise RuntimeError(
                "RealPersonaChatの dialogues/ と interlocutors.json を"
                "発見できませんでした。"
            )

        output = self.processed_dir / "realpersonachat.jsonl"
        dialogue_paths = sorted((dataset_dir / "dialogues").glob("*.json"))
        total = len(dialogue_paths)
        if total == 0:
            raise RuntimeError("RealPersonaChatの対話JSONがありません。")

        self.log(f"RealPersonaChat変換開始: {total:,}対話")
        seen: set[str] = set()
        emitted = 0
        with output.open("w", encoding="utf-8") as dst:
            for index, path in enumerate(dialogue_paths, 1):
                self.check_cancel()
                self.stats.real_dialogues_seen += 1
                try:
                    dialogue = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self.log(f"読込失敗をスキップ: {path.name}: {exc}")
                    self.stats.skipped_records += 1
                    continue

                if not self.real_quality_ok(dialogue):
                    self.stats.skipped_records += 1
                    continue

                utterances = dialogue.get("utterances")
                if not isinstance(utterances, list):
                    self.stats.skipped_records += 1
                    continue

                speaker_map: dict[str, str] = {}
                cleaned: list[str] = []
                for row in utterances:
                    if not isinstance(row, dict):
                        continue
                    text = normalize_plain_text(str(row.get("text", "")))
                    if not text:
                        continue
                    if self.cfg.skip_masked_real and (
                        MASKED_NAME_RE.search(text) or "＊＊" in text
                    ):
                        continue
                    speaker = str(row.get("interlocutor_id", "?"))
                    if speaker not in speaker_map:
                        speaker_map[speaker] = f"話者{len(speaker_map) + 1}"
                    cleaned.append(f"{speaker_map[speaker]}: {text}")

                turn_count = max(2, self.cfg.real_turns_per_record)
                for start in range(0, len(cleaned), turn_count):
                    block = cleaned[start : start + turn_count]
                    if len(block) < 2:
                        continue
                    text = "\n".join(block)
                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    write_jsonl(dst, text, "RealPersonaChat")
                    emitted += 1
                    if (
                        self.cfg.max_real_records
                        and emitted >= self.cfg.max_real_records
                    ):
                        break
                if self.cfg.max_real_records and emitted >= self.cfg.max_real_records:
                    break

                if index == 1 or index % 100 == 0 or index == total:
                    ratio = index / total
                    start_p, end_p = (
                        (18, 38) if self.cfg.mode == "mix" else (30, 82)
                    )
                    self.progress(
                        start_p + int((end_p - start_p) * ratio),
                        f"RealPersonaChat {index:,}/{total:,}対話",
                    )

        self.stats.real_records = emitted
        self.log(f"RealPersonaChat出力: {emitted:,}レコード -> {output}")
        return output

    @staticmethod
    def find_real_dataset(root: Path) -> Path | None:
        if not root.exists():
            return None
        for candidate in [root, *root.glob("*"), *root.glob("*/*")]:
            nested = candidate / "real_persona_chat"
            for item in (candidate, nested):
                if (
                    (item / "dialogues").is_dir()
                    and (item / "interlocutors.json").is_file()
                ):
                    return item
        return None

    def real_quality_ok(self, dialogue: dict) -> bool:
        threshold = self.cfg.min_real_quality
        if threshold <= 0:
            return True
        evaluations = dialogue.get("evaluations")
        if not isinstance(evaluations, list) or not evaluations:
            return False
        metrics = ("familiarity", "comprehension", "satisfaction")
        for metric in metrics:
            values = [
                row.get(metric)
                for row in evaluations
                if isinstance(row, dict) and isinstance(row.get(metric), int)
            ]
            if not values or sum(values) / len(values) < threshold:
                return False
        return True

    def build_wikipedia(self) -> Path:
        dump = self.download(
            self.cfg.wiki_url,
            self.raw_dir / "jawiki-latest-pages-articles.xml.bz2",
            43 if self.cfg.mode == "mix" else 6,
            62 if self.cfg.mode == "mix" else 36,
        )
        output = self.processed_dir / "wikipedia.jsonl"
        self.log(
            "Wikipedia XMLをストリーミング解析中"
            "（大きなダンプもRAMへ全展開しません）"
        )
        emitted = 0
        page_seen = 0
        seen: set[str] = set()

        with (
            bz2.open(dump, "rb") as source,
            output.open("w", encoding="utf-8") as dst,
        ):
            context = ET.iterparse(source, events=("end",))
            for _event, elem in context:
                self.check_cancel()
                if local_name(elem.tag) != "page":
                    continue
                page_seen += 1
                self.stats.wiki_pages_seen = page_seen
                title = child_text(elem, "title")
                namespace = child_text(elem, "ns")
                redirect = any(local_name(child.tag) == "redirect" for child in elem)
                raw = descendant_text(elem, "text")
                elem.clear()

                if namespace != "0" or redirect or not raw:
                    continue
                cleaned = clean_wikitext(raw)
                if len(cleaned) < self.cfg.wiki_min_chars:
                    self.stats.skipped_records += 1
                    continue

                for chunk in split_text(
                    cleaned,
                    self.cfg.wiki_chunk_chars,
                    self.cfg.wiki_min_chars,
                ):
                    text = f"{title}\n{chunk}" if title else chunk
                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    write_jsonl(dst, text, "Wikipedia")
                    emitted += 1
                    if self.cfg.max_wiki_records and emitted >= self.cfg.max_wiki_records:
                        break
                if self.cfg.max_wiki_records and emitted >= self.cfg.max_wiki_records:
                    break

                if page_seen == 1 or page_seen % 1000 == 0:
                    base, ceiling = (
                        (63, 86) if self.cfg.mode == "mix" else (37, 88)
                    )
                    moving = base + min(
                        ceiling - base - 1,
                        int(page_seen**0.5 / 3),
                    )
                    self.progress(
                        moving,
                        f"Wikipedia {page_seen:,}ページ解析 / "
                        f"{emitted:,}レコード出力",
                    )
                    self.log(
                        f"Wikipedia進捗: {page_seen:,}ページ確認、"
                        f"{emitted:,}レコード"
                    )

        self.stats.wiki_records = emitted
        self.log(f"Wikipedia出力: {emitted:,}レコード -> {output}")
        return output

    def merge_jsonl(self, first: Path, second: Path) -> Path:
        output = self.processed_dir / "mixed_dataset.jsonl"
        remaining_a = self.stats.real_records
        remaining_b = self.stats.wiki_records
        total = remaining_a + remaining_b
        rng = random.Random(self.cfg.mix_seed)
        written = 0

        with (
            first.open("r", encoding="utf-8") as a,
            second.open("r", encoding="utf-8") as b,
            output.open("w", encoding="utf-8") as dst,
        ):
            while remaining_a or remaining_b:
                self.check_cancel()
                if remaining_a == 0:
                    choose_a = False
                elif remaining_b == 0:
                    choose_a = True
                else:
                    choose_a = (
                        rng.randrange(remaining_a + remaining_b) < remaining_a
                    )
                src = a if choose_a else b
                line = src.readline()
                if not line:
                    if choose_a:
                        remaining_a = 0
                    else:
                        remaining_b = 0
                    continue
                dst.write(line)
                if choose_a:
                    remaining_a -= 1
                else:
                    remaining_b -= 1
                written += 1
                if written == 1 or written % 5000 == 0 or written == total:
                    value = 88 + int(8 * written / max(total, 1))
                    self.progress(
                        value,
                        f"洗濯中 {written:,}/{total:,}レコード",
                    )

        self.stats.mixed_records = written
        self.log(f"混合完了: {written:,}レコード -> {output}")
        return output

    def write_metadata(self, final_path: Path) -> None:
        generated_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "name": "GDP-DeepSulk generated dataset",
            "generated_at_utc": generated_at,
            "mode": self.cfg.mode,
            "output": str(final_path),
            "configuration": asdict(self.cfg),
            "statistics": asdict(self.stats),
            "sources": {
                "RealPersonaChat": {
                    "url": "https://github.com/nu-dialogue/real-persona-chat",
                    "download_url": self.cfg.real_url,
                    "license": "CC BY-SA 4.0",
                },
                "Wikipedia": {
                    "provider": "Wikimedia Foundation",
                    "download_url": self.cfg.wiki_url,
                    "license": "CC BY-SA 4.0",
                },
            },
            "preprocessing": [
                "不要・空データの除外",
                "テキスト正規化",
                "形式変換（JSONL）",
                "完全一致重複の除外",
                "選択時はRealPersonaChatとWikipediaをランダム順で混合",
            ],
            "license": "CC BY-SA 4.0",
        }
        (self.output_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        license_text = f"""# Dataset License and Attribution

このデータセットは以下の資料を利用しています。

- RealPersonaChat
  - 提供元: nu-dialogue
  - URL: https://github.com/nu-dialogue/real-persona-chat
  - ライセンス: CC BY-SA 4.0

- Wikipedia
  - 提供元: Wikimedia Foundation および各記事の執筆者
  - 取得元: {self.cfg.wiki_url}
  - ライセンス: CC BY-SA 4.0

本データセットは、上記資料から不要なデータの除外、形式変換、
テキスト正規化、重複除外、混合などの前処理を行っています。

生成日時（UTC）: {generated_at}

ライセンス: CC BY-SA 4.0

再配布・公開時は、元資料へのクレジット表示と同一ライセンスの継承を行ってください。
"""
        (self.output_dir / "DATASET_LICENSE.md").write_text(
            license_text,
            encoding="utf-8",
        )
        self.log("manifestとライセンス表示ファイルを作成しました。")


def write_jsonl(dst, text: str, source: str) -> None:
    dst.write(
        json.dumps(
            {"text": text, "source": source},
            ensure_ascii=False,
        )
        + "\n"
    )


def normalize_plain_text(text: str) -> str:
    text = html.unescape(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = SPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def remove_balanced_templates(text: str, passes: int = 8) -> str:
    pattern = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    for _ in range(passes):
        new = pattern.sub(" ", text)
        if new == text:
            break
        text = new
    return text


def clean_wikitext(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = REF_SINGLE_RE.sub(" ", text)
    text = TABLE_RE.sub(" ", text)
    text = FILE_LINK_RE.sub(" ", text)
    text = CATEGORY_RE.sub(" ", text)
    text = remove_balanced_templates(text)
    text = WIKI_LINK_RE.sub(lambda match: match.group(2), text)
    text = WIKI_LINK_SIMPLE_RE.sub(lambda match: match.group(1), text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or " ", text)
    text = HEADING_RE.sub(lambda match: f"\n{match.group(1)}\n", text)
    text = BOLD_ITALIC_RE.sub("", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(("|", "!", "[[Category:", "[[カテゴリ:")):
            continue
        stripped = re.sub(r"^[*#:;]+\s*", "", stripped)
        stripped = SPACE_RE.sub(" ", stripped).strip()
        if stripped:
            lines.append(stripped)
    return MULTI_NL_RE.sub("\n\n", "\n".join(lines)).strip()


def split_text(text: str, max_chars: int, min_chars: int) -> Iterable[str]:
    max_chars = max(200, max_chars)
    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[。！？!?])", paragraph)
        else:
            sentences = [paragraph]
        for piece in sentences:
            piece = piece.strip()
            if not piece:
                continue
            if len(buffer) + len(piece) + 1 <= max_chars:
                buffer = f"{buffer}\n{piece}".strip()
            else:
                if len(buffer) >= min_chars:
                    yield buffer
                buffer = piece[:max_chars]
    if len(buffer) >= min_chars:
        yield buffer


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(elem: ET.Element, name: str) -> str:
    for child in elem:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def descendant_text(elem: ET.Element, name: str) -> str:
    for child in elem.iter():
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


class BuildWorker(QObject):
    log = Signal(str)
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, config: BuildConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            result = DatasetBuilder(
                self.config,
                self.log.emit,
                self.progress.emit,
                self.cancel_event,
            ).run()
            self.finished.emit(result)
        except CancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def cancel(self) -> None:
        self.cancel_event.set()
        self.log.emit(
            "キャンセル要求を受け付けました。"
            "現在の処理単位が終わるまで待機します。"
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.thread: QThread | None = None
        self.worker: BuildWorker | None = None
        self.close_after_cancel = False
        self.setWindowTitle(APP_NAME)
        self.resize(1040, 760)
        self._build_ui()
        self._update_mode()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        title = QLabel("GDP-DeepSulk 学習データ自動取得・洗濯GUI")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        source_group = QGroupBox("取得元と出力")
        source_form = QFormLayout(source_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("RealPersonaChatのみ", "real")
        self.mode_combo.addItem("Wikipediaのみ", "wiki")
        self.mode_combo.addItem(
            "洗濯（RealPersonaChat + Wikipediaを混ぜる）",
            "mix",
        )
        self.mode_combo.setCurrentIndex(2)
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        source_form.addRow("モード", self.mode_combo)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit(
            str(Path.cwd() / "data" / "dataset_builder")
        )
        browse = QPushButton("参照")
        browse.clicked.connect(self._browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse)
        source_form.addRow("出力先", output_row)
        layout.addWidget(source_group)

        options_row = QHBoxLayout()
        real_group = QGroupBox("RealPersonaChat設定")
        self.real_group = real_group
        real_form = QFormLayout(real_group)
        self.real_limit = self._spin(0, 10_000_000, 0, "0 = 全件")
        self.real_turns = self._spin(
            2,
            64,
            8,
            "1レコードにまとめる発話数",
        )
        self.real_quality = self._spin(
            0,
            5,
            4,
            "0 = 品質評価で除外しない",
        )
        self.skip_masked = QCheckBox("マスク記号を含む発話を除外")
        self.skip_masked.setChecked(True)
        self.real_url = QLineEdit(REAL_PERSONA_ARCHIVE)
        real_form.addRow("最大レコード", self.real_limit)
        real_form.addRow("発話数/レコード", self.real_turns)
        real_form.addRow("最低評価", self.real_quality)
        real_form.addRow("フィルター", self.skip_masked)
        real_form.addRow("取得URL", self.real_url)
        options_row.addWidget(real_group)

        wiki_group = QGroupBox("Wikipedia設定")
        self.wiki_group = wiki_group
        wiki_form = QFormLayout(wiki_group)
        self.wiki_limit = self._spin(
            0,
            20_000_000,
            100_000,
            "0 = 全件（大容量）",
        )
        self.wiki_chunk = self._spin(
            200,
            20_000,
            2000,
            "記事をこの文字数前後に分割",
        )
        self.wiki_min = self._spin(
            1,
            5000,
            100,
            "短すぎる文章を除外",
        )
        self.wiki_url = QLineEdit(WIKIPEDIA_DUMP)
        wiki_form.addRow("最大レコード", self.wiki_limit)
        wiki_form.addRow("最大文字数", self.wiki_chunk)
        wiki_form.addRow("最低文字数", self.wiki_min)
        wiki_form.addRow("取得URL", self.wiki_url)
        options_row.addWidget(wiki_group)
        layout.addLayout(options_row)

        mix_group = QGroupBox("洗濯設定")
        mix_form = QFormLayout(mix_group)
        self.mix_seed = self._spin(
            0,
            2_147_483_647,
            334,
            "同じ値なら同じ混合順",
        )
        self.keep_sources = QCheckBox("混合後も個別JSONLを残す")
        self.keep_sources.setChecked(True)
        mix_form.addRow("乱数シード", self.mix_seed)
        mix_form.addRow("中間出力", self.keep_sources)
        layout.addWidget(mix_group)
        self.mix_group = mix_group

        license_label = QLabel(
            "出力には DATASET_LICENSE.md と dataset_manifest.json を"
            "自動生成します。RealPersonaChat / Wikipedia: CC BY-SA 4.0"
        )
        license_label.setWordWrap(True)
        layout.addWidget(license_label)

        progress_group = QGroupBox("進捗")
        progress_layout = QVBoxLayout(progress_group)
        self.status_label = QLabel("待機中")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)

        log_group = QGroupBox("ログ")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(10_000)
        fixed = QFont("monospace")
        fixed.setStyleHint(QFont.StyleHint.Monospace)
        self.log_view.setFont(fixed)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 1)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("自動取得・前処理を開始")
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.setEnabled(False)
        self.open_button = QPushButton("出力先を開く")
        clear_button = QPushButton("ログ消去")
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.open_button.clicked.connect(self._open_output)
        clear_button.clicked.connect(self.log_view.clear)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)

    @staticmethod
    def _spin(
        minimum: int,
        maximum: int,
        value: int,
        tooltip: str,
    ) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setToolTip(tooltip)
        return widget

    @Slot()
    def _update_mode(self) -> None:
        mode = self.mode_combo.currentData()
        self.real_group.setEnabled(mode in {"real", "mix"})
        self.wiki_group.setEnabled(mode in {"wiki", "mix"})
        self.mix_group.setEnabled(mode == "mix")

    @Slot()
    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "出力先を選択",
            self.output_edit.text(),
        )
        if chosen:
            self.output_edit.setText(chosen)

    @Slot()
    def _open_output(self) -> None:
        path = Path(self.output_edit.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    @Slot()
    def _start(self) -> None:
        if self.thread is not None:
            return
        mode = str(self.mode_combo.currentData())
        if mode in {"wiki", "mix"} and self.wiki_limit.value() == 0:
            answer = QMessageBox.warning(
                self,
                "Wikipedia全件取得",
                "Wikipedia全件はダウンロード・処理・保存に長時間と"
                "大容量を使います。続行しますか？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        output = self.output_edit.text().strip()
        if not output:
            QMessageBox.critical(
                self,
                "入力エラー",
                "出力先を指定してください。",
            )
            return

        config = BuildConfig(
            mode=mode,
            output_dir=output,
            real_url=self.real_url.text().strip(),
            wiki_url=self.wiki_url.text().strip(),
            max_real_records=self.real_limit.value(),
            max_wiki_records=self.wiki_limit.value(),
            real_turns_per_record=self.real_turns.value(),
            min_real_quality=self.real_quality.value(),
            skip_masked_real=self.skip_masked.isChecked(),
            wiki_chunk_chars=self.wiki_chunk.value(),
            wiki_min_chars=self.wiki_min.value(),
            mix_seed=self.mix_seed.value(),
            keep_source_files=self.keep_sources.isChecked(),
        )

        self.log_view.clear()
        self._append_log(f"開始: {APP_NAME}")
        self._append_log(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        self.progress_bar.setValue(0)
        self.status_label.setText("開始中")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.thread = QThread(self)
        self.worker = BuildWorker(config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._set_progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot()
    def _cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.status_label.setText("キャンセル要求中")
        if self.worker is not None:
            self.worker.cancel()

    @Slot(str)
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")
        scroll = self.log_view.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    @Slot(int, str)
    def _set_progress(self, value: int, text: str) -> None:
        self.progress_bar.setValue(max(0, min(100, value)))
        self.status_label.setText(text)

    @Slot(dict)
    def _finished(self, result: dict) -> None:
        self._append_log("完了")
        stats = result.get("stats", {})
        message = (
            f"出力: {result.get('output')}\n\n"
            f"RealPersonaChat: {stats.get('real_records', 0):,}件\n"
            f"Wikipedia: {stats.get('wiki_records', 0):,}件\n"
            f"混合: {stats.get('mixed_records', 0):,}件"
        )
        QMessageBox.information(
            self,
            "データセット作成完了",
            message,
        )

    @Slot(str)
    def _failed(self, detail: str) -> None:
        self._append_log(detail)
        self.status_label.setText("エラー")
        QMessageBox.critical(self, "処理失敗", detail[-4000:])

    @Slot(str)
    def _cancelled(self, message: str) -> None:
        self._append_log(message)
        self.status_label.setText("キャンセル済み")

    @Slot()
    def _thread_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.close_after_cancel:
            self.close_after_cancel = False
            self.close()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.thread is not None:
            answer = QMessageBox.question(
                self,
                "処理中",
                "処理をキャンセルして終了しますか？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.close_after_cancel = True
            if self.worker is not None:
                self.worker.cancel()
            self.status_label.setText("キャンセル後に終了します")
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
