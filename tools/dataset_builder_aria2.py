#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from tools import dataset_builder_gui as gui


_ORIGINAL_DOWNLOAD = gui.DatasetBuilder.download
_MIB = 1024 * 1024


def _remote_size(url: str) -> int:
    """Best-effort Content-Length lookup for GUI progress display."""
    try:
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": gui.USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def download_with_aria2(
    self: gui.DatasetBuilder,
    url: str,
    destination: Path,
    stage_start: int,
    stage_end: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    control_file = Path(str(destination) + ".aria2")

    # A finished aria2 download has no .aria2 control file.
    if (
        destination.exists()
        and destination.stat().st_size > 0
        and not control_file.exists()
    ):
        self.log(f"再利用: {destination}")
        return destination

    aria2c = shutil.which("aria2c")
    if aria2c is None:
        self.log("aria2cが見つからないため、標準ダウンローダーへ切り替えます。")
        return _ORIGINAL_DOWNLOAD(self, url, destination, stage_start, stage_end)

    # An older GUI version used *.part. Preserve it and let aria2c continue it.
    legacy_part = destination.with_suffix(destination.suffix + ".part")
    if legacy_part.exists() and not destination.exists():
        legacy_part.replace(destination)
        self.log(f"旧形式の途中ファイルを引き継ぎます: {destination}")

    total = _remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing:
        self.log(f"aria2cで再開: {existing / _MIB:.1f} MiB受信済み")
    else:
        self.log(f"aria2cダウンロード開始: {url}")

    command = [
        aria2c,
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=0",
        "--retry-wait=3",
        "--connect-timeout=30",
        "--timeout=60",
        "--summary-interval=0",
        "--console-log-level=warn",
        "--download-result=hide",
        f"--user-agent={gui.USER_AGENT}",
        f"--dir={destination.parent}",
        f"--out={destination.name}",
        url,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    last_time = started
    last_size = existing
    last_log = started - 10

    try:
        while process.poll() is None:
            self.check_cancel()
            time.sleep(0.25)

            now = time.monotonic()
            if now - last_time < 0.75:
                continue

            downloaded = destination.stat().st_size if destination.exists() else 0
            elapsed = max(now - last_time, 0.001)
            speed = max(downloaded - last_size, 0) / elapsed
            last_time = now
            last_size = downloaded

            if total > 0:
                ratio = min(downloaded / total, 1.0)
                value = stage_start + int((stage_end - stage_start) * ratio)
                status = (
                    f"aria2c {downloaded / _MIB:.1f}/{total / _MIB:.1f} MiB "
                    f"({ratio * 100:.1f}%) {speed / _MIB:.1f} MiB/s"
                )
            else:
                value = stage_start
                status = (
                    f"aria2c {downloaded / _MIB:.1f} MiB "
                    f"{speed / _MIB:.1f} MiB/s"
                )

            self.progress(value, status)
            if now - last_log >= 5:
                self.log(status)
                last_log = now

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"aria2cが終了コード{return_code}で停止しました。"
                "途中ファイルは次回の再開用に保持します。"
            )
    except BaseException:
        _stop_process(process)
        raise

    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("aria2cは正常終了しましたが、出力ファイルがありません。")

    elapsed_total = max(time.monotonic() - started, 0.001)
    self.progress(stage_end, f"aria2cダウンロード完了: {destination.name}")
    self.log(
        f"aria2cダウンロード完了: {destination} "
        f"({destination.stat().st_size / _MIB:.1f} MiB, "
        f"{destination.stat().st_size / elapsed_total / _MIB:.1f} MiB/s平均)"
    )
    return destination


gui.DatasetBuilder.download = download_with_aria2


if __name__ == "__main__":
    raise SystemExit(gui.main())
