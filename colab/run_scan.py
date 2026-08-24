"""Google Colab runner: scan, then display tables and charts inline."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

# ===== 可修改配置 =====
BOARD_TYPES = ["industry", "concept"]
WORKERS = 2
LOOKBACK_CALENDAR_DAYS = 75
REFRESH = False
SCAN_ALL_SOURCE_BOARDS = True
USE_GOOGLE_DRIVE_CACHE = True
# ====================


def main() -> None:
    try:
        from google.colab import drive
        from IPython.display import Image, Markdown, display
    except ImportError as exc:
        raise RuntimeError("此脚本用于 Google Colab；本地请直接运行 mainline-scanner") from exc

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)], check=True)

    # Matplotlib 可能命中一个“声明支持中文但字形不完整”的字体；按文件绝对路径注册。
    font_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    if not any(path.is_file() for path in font_candidates):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"], check=True)
        subprocess.run(["fc-cache", "-f"], check=True)
    chinese_font = next((path for path in font_candidates if path.is_file()), None)
    if chinese_font is None:
        raise RuntimeError("已安装 fonts-noto-cjk，但没有找到可用中文字体文件")
    os.environ["A_SHARE_CHINESE_FONT_PATH"] = str(chinese_font)
    print(f"中文绘图字体: {chinese_font}")

    if USE_GOOGLE_DRIVE_CACHE:
        drive.mount("/content/drive", force_remount=False)
        cache_dir = Path("/content/drive/MyDrive/a-share-mainline-scanner/cache")
    else:
        cache_dir = Path("/content/a-share-mainline-cache")
    output_dir = Path("/content/a-share-mainline-results")
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

    import pandas as pd

    scored = pd.read_csv(output_dir / "板块完整评分.csv")
    display_scored = scored.copy()
    display_scored["_display_group"] = display_scored["name"].astype(str).str.replace(
        r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", regex=True,
    )
    columns = [
        "kind", "name", "status", "mainline_score", "candidate_score",
        "ret_1d", "ret_5d", "ret_10d", "slope_3d", "slope_5d",
        "acceleration", "flow_1d_pct", "flow_5d_pct", "breadth",
        "flow_1d_source", "flow_5d_source",
    ]
    columns = [col for col in columns if col in scored]
    display(Markdown("## 当前主线 Top 30"))
    main_rank = display_scored[
        display_scored["status"].astype(str).str.startswith("主线")
    ].sort_values("mainline_score", ascending=False).drop_duplicates(["kind", "_display_group"])
    display(main_rank[columns].head(30))
    display(Markdown("## 潜在启动 Top 30"))
    candidate_rank = display_scored[
        display_scored["status"].isin(["潜在启动", "值得关注"])
    ].sort_values("candidate_score", ascending=False).drop_duplicates(["kind", "_display_group"])
    display(candidate_rank[columns].head(30))

    audit_file = output_dir / "数据完整性审计.xlsx"
    summary = pd.read_excel(audit_file, sheet_name="汇总")
    audit = pd.read_excel(audit_file, sheet_name="全部板块审计")
    display(Markdown("## 数据完整性汇总"))
    display(summary)
    display(Markdown(
        "`flow_*_match_pct` 是真实主力资金直接命中率；"
        "`flow_*_available_pct` 包含明确标注的 CMF 量价代理可用率。"
    ))
    display(Markdown("## 失败原因分布"))
    display(audit.groupby(["audit_status", "history_source"], dropna=False).size().rename("数量").reset_index())
    display(Markdown("## 已成功行情的数据源分布"))
    display(
        audit[audit["history_rows"] > 0]
        .groupby(["kind", "history_source"], dropna=False)
        .size().rename("板块数").reset_index()
    )
    missing = audit[audit["is_omitted"] == True]  # noqa: E712
    if len(missing):
        display(Markdown(f"## 遗漏板块（{len(missing)} 个，显示前100个）"))
        missing_columns = [
            "kind", "code", "name", "audit_status", "history_source",
            "fetch_error", "history_rows", "history_end",
        ]
        display(missing[missing_columns].head(100))

    display(Markdown("## 主线雷达"))
    display(Image(filename=str(output_dir / "主线雷达.png")))
    display(Markdown("## 领先板块近30日走势"))
    display(Image(filename=str(output_dir / "领先板块走势.png")))
    print(f"\n结果已在上方直接显示；临时报告目录：{output_dir}（不自动下载）")


if __name__ == "__main__":
    main()
