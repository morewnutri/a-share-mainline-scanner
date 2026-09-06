"""Google Colab valuation runner with persistent custom stocks in Google Drive.

Run in a Colab cell with:

    %run colab/run_valuation.py

Using %run (rather than !python ...) keeps execution in the notebook kernel, so
Python input() renders an interactive input field for adding stock codes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ===== 可修改配置 =====
REFRESH = True
USE_GOOGLE_DRIVE = True
DRIVE_ROOT = "/content/drive/MyDrive/a-share-mainline-scanner"
# ====================


def main() -> None:
    try:
        from google.colab import drive
        from IPython.display import Markdown, display
    except ImportError as exc:
        raise RuntimeError("此脚本用于 Google Colab；本地请运行 valuation-scanner 或 python -m mainline_scanner.valuation_cli") from exc

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)], check=True)

    if USE_GOOGLE_DRIVE:
        drive.mount("/content/drive", force_remount=False)
        persistent_root = Path(DRIVE_ROOT)
        cache_dir = persistent_root / "valuation_cache"
        state_dir = persistent_root / "valuation_state"
        output_dir = persistent_root / "valuation_reports" / "latest"
    else:
        cache_dir = Path("/content/a-share-valuation-cache")
        state_dir = Path("/content/a-share-valuation-state")
        output_dir = Path("/content/a-share-valuation-results")

    for path in (cache_dir, state_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Import only after pip install -e so this always uses the checked-out code.
    from mainline_scanner.valuation_cli import main as valuation_main

    argv = [
        "--config", str(repo_root / "valuation_config.json"),
        "--cache-dir", str(cache_dir),
        "--state-dir", str(state_dir),
        "--output-dir", str(output_dir),
    ]
    if REFRESH:
        argv.append("--refresh")

    # Direct in-kernel call is intentional: input() below becomes a Colab input
    # field. A shell subprocess invoked with !python may not have interactive stdin.
    valuation_main(argv)

    display(Markdown("## 估值结果文件"))
    print(output_dir / "A股主线估值.xlsx")
    print(output_dir / "板块估值.csv")
    print(output_dir / "个股估值.csv")
    print(output_dir / "数据新鲜度.csv")
    print(f"\n自定义股票永久保存在: {state_dir / 'custom_stocks.json'}")


if __name__ == "__main__":
    main()
