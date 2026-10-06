"""Google Colab runner: scan, then display tables and charts inline.

All files stay under /content; Google Drive is neither mounted nor required.
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

# ===== 可修改配置 =====
BOARD_TYPES = ["industry", "concept"]
WORKERS = 2
LOOKBACK_CALENDAR_DAYS = 120
REFRESH = False
SCAN_ALL_SOURCE_BOARDS = True
BAOSTOCK_MODE = "off"  # industry 较慢；all 还会合成概念，首次运行可能很慢
RESEARCH_SIGNALS_FILE = Path("/content/board_signals.csv")  # 可选：带 available_at 的研究证据
MARKET_HISTORY_FILE = Path("/content/market_daily.csv")  # 可选：独立全A成交额/基准
# ====================


def main() -> None:
    try:
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
    revision_result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                                     capture_output=True, text=True, check=False)
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "未识别（非 Git 检出）"
    print(f"扫描器提交: {revision}")

    cache_dir = Path("/content/a-share-mainline-cache")
    snapshot_dir = Path("/content/a-share-mainline-snapshots")
    output_dir = Path("/content/a-share-mainline-results")
    for path in (cache_dir, snapshot_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "mainline_scanner.cli",
        "--board-types", *BOARD_TYPES,
        "--workers", str(WORKERS),
        "--lookback-calendar-days", str(LOOKBACK_CALENDAR_DAYS),
        "--cache-dir", str(cache_dir),
        "--output-dir", str(output_dir),
        "--cache-hours", "24",
        "--snapshot-cache-minutes", "5",
        "--snapshot-dir", str(snapshot_dir),
        "--baostock-mode", BAOSTOCK_MODE,
    ]
    if REFRESH:
        command.append("--refresh")
    if SCAN_ALL_SOURCE_BOARDS:
        command.extend(["--exclude-regex", ""])
    if RESEARCH_SIGNALS_FILE.is_file():
        command.extend(["--research-signals", str(RESEARCH_SIGNALS_FILE)])
    else:
        print("研究证据文件缺失：潜在主线分将为空；可将 board_signals.csv 上传到 /content。")
    if MARKET_HISTORY_FILE.is_file():
        command.extend(["--market-history", str(MARKET_HISTORY_FILE)])
    else:
        print("全A市场日线文件缺失：真实成交占比和指数相对强度将为空，使用已标注的量价代理。")
    scan_result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    if scan_result.returncode:
        if scan_result.stdout:
            print(scan_result.stdout[-8000:])
        if scan_result.stderr:
            print(scan_result.stderr[-12000:], file=sys.stderr)
        raise RuntimeError(f"扫描器退出码 {scan_result.returncode}；上方是原始日志和异常。")

    import pandas as pd

    scored = pd.read_csv(output_dir / "板块完整评分.csv")
    display_scored = scored.copy()
    display_scored["_display_group"] = display_scored["name"].astype(str).str.replace(
        r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", regex=True,
    )

    def show_research_rank(title: str, filename: str, rank_col: str, wanted: list[str]) -> None:
        display(Markdown(f"## {title}"))
        path = output_dir / filename
        if not path.is_file():
            display(Markdown(f"未生成 `{filename}`；请核对扫描器提交和运行日志。"))
            return
        table = pd.read_csv(path)
        if rank_col not in table or table[rank_col].notna().sum() == 0:
            reason = "尚无按时点记录的盈利/产业/政策证据" if rank_col == "potential_rank_score" else (
                "需要连续交易日快照" if rank_col == "switch_score" else "当前缺少可计算的观测数据"
            )
            display(Markdown(f"本次无有效排名：{reason}。完整板块仍保留在 `{filename}`。"))
            return
        table["_display_group"] = table["name"].astype(str).str.replace(
            r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", regex=True,
        )
        table = table.drop_duplicates(["kind", "_display_group"])
        display(table[[col for col in wanted if col in table]].head(30))

    show_research_rank("研究潜在主线 Top 30", "研究潜在主线.csv", "potential_rank_score", [
        "kind", "code", "name", "research_phase", "potential_rank_score", "potential_score",
        "potential_coverage", "potential_state", "market_confirmation_score", "geo_net_exposure",
    ])
    show_research_rank("市场确认主线 Top 30", "市场确认主线.csv", "market_confirmation_rank_score", [
        "kind", "code", "name", "research_phase", "market_confirmation_rank_score",
        "market_confirmation_score", "confirmation_coverage", "top_rank_days_10",
        "turnover_share", "rs_market_5d", "breadth",
    ])
    show_research_rank("主线切换 Top 30", "主线切换.csv", "switch_score", [
        "kind", "code", "name", "switch_score", "switch_from", "confirmation_change",
        "turnover_change", "market_confirmation_score", "potential_score",
    ])
    show_research_rank("退潮风险 Top 30", "退潮风险.csv", "exhaustion_rank_score", [
        "kind", "code", "name", "exhaustion_rank_score", "exhaustion_score",
        "exhaustion_coverage", "risk_breadth_divergence", "risk_leader_divergence",
        "risk_turnover_efficiency_loss", "risk_failure_rate", "risk_catalyst_exhaustion",
    ])
    columns = [
        "kind", "name", "lifecycle", "mainline_score", "ignition_score", "confirmation_score",
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
    display(Markdown("## 火种 / 点火 Top 30"))
    candidate_rank = display_scored[
        display_scored["lifecycle"].isin(["Seed", "Ignition"])
    ].sort_values("ignition_score", ascending=False).drop_duplicates(["kind", "_display_group"])
    display(candidate_rank[columns].head(30))

    sideways_columns = [
        "kind", "name", "sideways_seed_status", "sideways_seed_score",
        "box_range_20d_pct", "range_position_60d_pct", "distance_high_60d_pct",
        "slope_20d", "volatility_ratio_5_20",
    ]
    sideways_columns = [col for col in sideways_columns if col in display_scored]
    display(Markdown("## 横盘火种（低位箱体）Top 30"))
    sideways_rank = display_scored[
        display_scored["sideways_seed_status"].isin(["横盘火种", "横盘观察"])
    ].sort_values("sideways_seed_score", ascending=False).drop_duplicates(["kind", "_display_group"])
    if len(sideways_rank):
        display(sideways_rank[sideways_columns].head(30))
    else:
        display(Markdown("本次没有板块同时通过低位、窄箱体、低斜率和波动收缩门槛。"))

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
    sideways_chart = output_dir / "横盘火种雷达.png"
    if sideways_chart.is_file():
        display(Markdown("## 横盘火种雷达"))
        display(Image(filename=str(sideways_chart)))
    display(Markdown("## 领先板块近30日走势"))
    display(Image(filename=str(output_dir / "领先板块走势.png")))
    print(f"\n结果已在上方直接显示；临时报告目录：{output_dir}（Colab 会话结束后清除，不自动下载）")
    print(f"横盘火种明细：{output_dir / '横盘火种.csv'}")
    print(f"估值连接文件：{output_dir / '板块完整评分.csv'}")


if __name__ == "__main__":
    main()
