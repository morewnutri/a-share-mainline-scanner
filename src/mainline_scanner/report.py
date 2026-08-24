from __future__ import annotations

import html
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOG = logging.getLogger(__name__)
KIND_CN = {"industry": "行业", "concept": "概念"}


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", rc={"font.sans-serif": plt.rcParams["font.sans-serif"]})


def _label_top(ax, data: pd.DataFrame, x: str, y: str, n: int = 12) -> None:
    chosen = data.nlargest(n, "mainline_score")
    for _, row in chosen.iterrows():
        ax.annotate(str(row["name"]), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)


def make_dashboard(scored: pd.DataFrame, out_path: Path) -> None:
    configure_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    plot = scored.dropna(subset=["slope_5d", "acceleration"]).copy()
    sizes = np.clip(plot["amount_ratio_5_20"].fillna(1), .4, 3) * 55
    scatter = axes[0, 0].scatter(
        plot["slope_5d"], plot["acceleration"], c=plot["mainline_score"], s=sizes,
        cmap="RdYlGn", alpha=.72, edgecolor="white", linewidth=.4,
    )
    axes[0, 0].axvline(0, color="#666", lw=.8); axes[0, 0].axhline(0, color="#666", lw=.8)
    axes[0, 0].set(title="趋势速度 vs 加速度（气泡=量能比）", xlabel="5日趋势斜率（%/交易日）", ylabel="加速度")
    _label_top(axes[0, 0], plot, "slope_5d", "acceleration")
    fig.colorbar(scatter, ax=axes[0, 0], label="主线分")

    top = scored.nlargest(18, "mainline_score").sort_values("mainline_score")
    colors = ["#c0392b" if x >= 80 else "#f39c12" if x >= 68 else "#3498db" for x in top["mainline_score"]]
    axes[0, 1].barh(top["name"], top["mainline_score"], color=colors)
    axes[0, 1].axvline(68, color="#f39c12", ls="--", lw=1); axes[0, 1].axvline(80, color="#c0392b", ls="--", lw=1)
    axes[0, 1].set(title="主线综合评分", xlabel="0–100")

    cand = scored[scored["status"].isin(["潜在启动", "值得关注"])].nlargest(18, "candidate_score").sort_values("candidate_score")
    if cand.empty:
        cand = scored.nlargest(18, "candidate_score").sort_values("candidate_score")
    axes[1, 0].barh(cand["name"], cand["candidate_score"], color="#8e44ad")
    axes[1, 0].axvline(65, color="#777", ls="--", lw=1); axes[1, 0].axvline(75, color="#222", ls="--", lw=1)
    axes[1, 0].set(title="潜在主线（启动）评分", xlabel="0–100")

    heat = scored.nlargest(20, "mainline_score").set_index("name")[["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"]]
    heat.columns = ["1日", "3日", "5日", "10日", "20日"]
    sns.heatmap(heat, cmap="RdYlGn", center=0, annot=True, fmt=".1f", ax=axes[1, 1], cbar_kws={"label": "%"})
    axes[1, 1].set(title="领先板块多周期涨跌幅", xlabel="", ylabel="")
    fig.suptitle("A股板块主线雷达", fontsize=20, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_trend_chart(scored: pd.DataFrame, histories: dict[tuple[str, str], pd.DataFrame], out_path: Path) -> None:
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    for _, row in scored.nlargest(10, "mainline_score").iterrows():
        h = histories[(str(row["kind"]), str(row["code"]))].tail(30)
        normalized = h["close"] / h["close"].iloc[0] * 100
        ax.plot(h["date"], normalized, lw=1.8, label=f"{row['name']} ({row['mainline_score']:.0f})")
    ax.axhline(100, color="#555", lw=.8)
    ax.set(title="主线候选近 30 个交易日相对走势（起点=100）", ylabel="归一化指数", xlabel="")
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fmt_table(df: pd.DataFrame, n: int = 20) -> str:
    cols = ["kind", "name", "status", "mainline_score", "candidate_score", "ret_5d", "ret_10d", "slope_5d", "acceleration", "flow_5d_pct", "breadth", "amount_ratio_5_20"]
    cols = [c for c in cols if c in df]
    x = df[cols].head(n).copy()
    if "kind" in x: x["kind"] = x["kind"].map(KIND_CN).fillna(x["kind"])
    return x.to_markdown(index=False, floatfmt=".2f")


def write_outputs(
    scored: pd.DataFrame,
    histories: dict[tuple[str, str], pd.DataFrame],
    failures: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = output_dir / "板块完整评分.csv"
    xlsx = output_dir / "板块主线扫描.xlsx"
    dashboard = output_dir / "主线雷达.png"
    trends = output_dir / "领先板块走势.png"
    md = output_dir / "主线判断报告.md"
    html_path = output_dir / "主线判断报告.html"
    scored.to_csv(full_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="完整评分", index=False)
        scored[scored["status"].str.startswith("主线")].to_excel(writer, sheet_name="当前主线", index=False)
        scored[scored["status"].isin(["潜在启动", "值得关注"])].sort_values("candidate_score", ascending=False).to_excel(writer, sheet_name="潜在主线", index=False)
        failures.to_excel(writer, sheet_name="抓取失败", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
            ws.column_dimensions["B"].width = 20
    make_dashboard(scored, dashboard)
    make_trend_chart(scored, histories, trends)

    main = scored[scored["status"].str.startswith("主线")].sort_values("mainline_score", ascending=False)
    candidate = scored[scored["status"].isin(["潜在启动", "值得关注"])].sort_values("candidate_score", ascending=False)
    as_of = pd.to_datetime(scored["as_of"]).max().date()
    report = f"""# A股板块主线判断报告

数据截止：{as_of}  
覆盖：{len(scored)} 个有效板块（行业 + 概念）；抓取失败/数据不足：{len(failures)} 个。

## 当前主线

{_fmt_table(main if not main.empty else scored.sort_values('mainline_score', ascending=False), 20)}

## 可能成为下一主线的板块

{_fmt_table(candidate if not candidate.empty else scored.sort_values('candidate_score', ascending=False), 20)}

## 判定逻辑

- **主线分**：5/10 日涨幅、趋势斜率及拟合质量、相对强弱、5/10 日主力净流入占比、上涨家数占比、量能和上涨持续性。
- **启动分**：3 日斜率、斜率加速度、5 日相对强弱、当日/5 日资金变化、量价扩张、突破位置和市场广度；对 10 日暴涨或显著偏离 20 日均线的板块扣除拥挤分。
- “潜在启动”要求短斜率与加速度均为正、10 日尚未过度上涨；“主线核心”要求趋势与 10 日收益为正且市场广度不弱。

> 这是量价与资金行为筛选器，不是收益保证或买卖建议。板块概念存在重叠，应用时还需结合政策/事件驱动、指数环境、个股位置与风险预算复核。
"""
    md.write_text(report, encoding="utf-8")
    table_html = scored.head(100).to_html(index=False, classes="data", border=0, float_format=lambda v: f"{v:.2f}")
    html_path.write_text(f"""<!doctype html><meta charset='utf-8'><title>A股主线雷达</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f7f8fa}}img{{max-width:100%;background:white}}table{{border-collapse:collapse;background:white;font-size:12px}}th,td{{padding:6px 8px;border:1px solid #ddd;white-space:nowrap}}th{{position:sticky;top:0;background:#263238;color:white}}h1{{color:#263238}}</style>
<h1>A股板块主线雷达</h1><p>数据截止 {as_of}；共 {len(scored)} 个有效板块。</p>
<img src='{html.escape(dashboard.name)}'><img src='{html.escape(trends.name)}'><h2>完整评分（前100）</h2>{table_html}
""", encoding="utf-8")
    return {"csv": full_csv, "xlsx": xlsx, "dashboard": dashboard, "trends": trends, "markdown": md, "html": html_path}

