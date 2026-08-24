"""Google Colab runner: install, full scan, persist cache, and download reports."""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ===== 可修改配置 =====
BOARD_TYPES = ["industry", "concept"]
WORKERS = 2
LOOKBACK_CALENDAR_DAYS = 75
REFRESH = False
SCAN_ALL_SOURCE_BOARDS = True
USE_GOOGLE_DRIVE = True
# ====================


def main() -> None:
    try:
        from google.colab import drive, files
    except ImportError as exc:
        raise RuntimeError("此脚本用于 Google Colab；本地请直接运行 mainline-scanner") from exc

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)], check=True)

    if USE_GOOGLE_DRIVE:
        drive.mount("/content/drive", force_remount=False)
        base = Path("/content/drive/MyDrive/a-share-mainline-scanner")
    else:
        base = Path("/content/a-share-mainline-scanner-data")
    cache_dir = base / "cache"
    output_dir = base / "reports" / "latest"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "mainline_scanner.cli",
        "--board-types", *BOARD_TYPES,
        "--workers", str(WORKERS),
        "--lookback-calendar-days", str(LOOKBACK_CALENDAR_DAYS),
        "--cache-dir", str(cache_dir),
        "--output-dir", str(output_dir),
        "--cache-hours", "24",
    ]
    if REFRESH:
        command.append("--refresh")
    if SCAN_ALL_SOURCE_BOARDS:
        command.extend(["--exclude-regex", ""])
    subprocess.run(command, cwd=repo_root, check=True)

    omitted = output_dir / "遗漏板块明细.csv"
    if omitted.exists():
        import pandas as pd
        missing = pd.read_csv(omitted)
        print(f"\n遗漏板块共 {len(missing)} 个：")
        if len(missing):
            print(missing[["kind", "code", "name", "audit_status", "fetch_error"]].head(50).to_string(index=False))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_base = Path("/content") / f"a_share_mainline_reports_{stamp}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_dir))
    print(f"\n报告目录: {output_dir}")
    print(f"压缩包: {zip_path}")
    files.download(str(zip_path))


if __name__ == "__main__":
    main()

