from __future__ import annotations

import html
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

LOG = logging.getLogger(__name__)
KIND_CN = {"industry": "行业", "concept": "概念"}


def configure_chinese_font() -> None:
    explicit_path = os.environ.get("A_SHARE_CHINESE_FONT_PATH", "")
    families: list[str] = []
    if explicit_path and Path(explicit_path).is_file():
        try:
            font_manager.fontManager.addfont(explicit_path)
            families.append(font_manager.FontProperties(fname=explicit_path).get_name())
        except Exception as exc:
            LOG.warning("中文字体文件注册失败 %s: %s", explicit_path, exc)
    families.extend([
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ])
    plt.rcParams["font.family"] = families
    plt.rcParams["font.sans-serif"] = families
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", rc={"font.family": families, "font.sans-serif": families})


def _deduplicate_for_display(scored: pd.DataFrame, score_col: str = "mainline_score") -> pd.DataFrame:
    """完整结果不删行；排行榜和图片合并“银行/银行Ⅱ”等同主题层级镜像。"""
    out = scored.sort_values(score_col, ascending=False).copy()
    out["_display_group"] = out["name"].astype(str).map(
        lambda value: re.sub(r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", value.strip())
    )
    return out.drop_duplicates(["kind", "_display_group"]).drop(columns="_display_group")


def _label_top(ax, data: pd.DataFrame, x: str, y: str, n: int = 12) -> None:
    chosen = data.nlargest(n, "mainline_score")
    for _, row in chosen.iterrows():
        ax.annotate(str(row["name"]), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)


def make_dashboard(scored: pd.DataFrame, out_path: Path) -> None:
    configure_chinese_font()
    scored = _deduplicate_for_display(scored)
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
    colors = ["#c0392b" if state == "Mainline" else "#f39c12" if state in {"Seed", "Ignition", "Diffusion"} else "#3498db"
              for state in top["lifecycle"]]
    axes[0, 1].barh(top["name"], top["mainline_score"], color=colors)
    axes[0, 1].axvline(68, color="#f39c12", ls="--", lw=1); axes[0, 1].axvline(80, color="#c0392b", ls="--", lw=1)
    axes[0, 1].set(title="主线综合评分（红色=正式确认）", xlabel="0–100")

    cand = scored[scored.get("lifecycle", pd.Series(index=scored.index, dtype=str)).isin(["Seed", "Ignition"])].nlargest(18, "ignition_score").sort_values("ignition_score")
    if cand.empty:
        cand = scored.nlargest(18, "ignition_score").sort_values("ignition_score")
    axes[1, 0].barh(cand["name"], cand["ignition_score"], color="#8e44ad")
    axes[1, 0].axvline(60, color="#777", ls="--", lw=1); axes[1, 0].axvline(72, color="#222", ls="--", lw=1)
    axes[1, 0].set(title="量化火种评分（须结合正式状态）", xlabel="0–100")

    heat = scored.nlargest(20, "mainline_score").set_index("name")[["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"]]
    heat.columns = ["1日", "3日", "5日", "10日", "20日"]
    sns.heatmap(heat, cmap="RdYlGn", center=0, annot=True, fmt=".1f", ax=axes[1, 1], cbar_kws={"label": "%"})
    axes[1, 1].set(title="领先板块多周期涨跌幅", xlabel="", ylabel="")
    fig.suptitle("A股板块主线雷达", fontsize=20, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_trend_chart(scored: pd.DataFrame, histories: dict[tuple[str, str], pd.DataFrame], out_path: Path) -> None:
    configure_chinese_font()
    scored = _deduplicate_for_display(scored)
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


def make_sideways_seed_chart(scored: pd.DataFrame, out_path: Path) -> None:
    configure_chinese_font()
    ranked = _deduplicate_for_display(scored, "sideways_seed_score").nlargest(24, "sideways_seed_score")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    bars = ranked.head(18).sort_values("sideways_seed_score")
    colors = ["#d35400" if s == "横盘火种" else "#f5b041" if s == "横盘观察" else "#95a5a6" for s in bars["sideways_seed_status"]]
    axes[0].barh(bars["name"], bars["sideways_seed_score"], color=colors)
    axes[0].axvline(60, color="#777", ls="--"); axes[0].axvline(70, color="#222", ls="--")
    axes[0].set(title="低位箱体横盘火种评分", xlabel="0–100")
    axes[1].scatter(ranked["box_range_20d_pct"], ranked["range_position_60d_pct"], c=ranked["sideways_seed_score"], cmap="YlOrRd", s=70)
    for _, row in ranked.head(12).iterrows():
        axes[1].annotate(str(row["name"]), (row["box_range_20d_pct"], row["range_position_60d_pct"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[1].axvline(15, color="#777", ls="--"); axes[1].axhline(55, color="#777", ls="--")
    axes[1].set(title="箱体宽度 vs 60日区间位置", xlabel="20日箱体振幅（%）", ylabel="60日区间位置（%）")
    fig.suptitle("A股板块横盘火种雷达", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fmt_table(df: pd.DataFrame, n: int = 20) -> str:
    cols = ["kind", "name", "history_source", "comparison_peer_count", "lifecycle", "absolute_strength_state", "mainline_score", "mainline_coverage", "ignition_score", "ignition_coverage", "ret_5d", "ret_10d", "breadth", "mainline_blockers"]
    cols = [c for c in cols if c in df]
    x = df[cols].head(n).copy()
    if "kind" in x: x["kind"] = x["kind"].map(KIND_CN).fillna(x["kind"])
    return x.to_markdown(index=False, floatfmt=".2f")


def _fmt_sideways_table(df: pd.DataFrame, n: int = 20) -> str:
    cols = [
        "kind", "name", "sideways_seed_status", "sideways_seed_score", "box_range_20d_pct",
        "range_position_60d_pct", "distance_high_60d_pct", "slope_20d", "volatility_ratio_5_20",
    ]
    x = df[[c for c in cols if c in df]].head(n).copy()
    if "kind" in x:
        x["kind"] = x["kind"].map(KIND_CN).fillna(x["kind"])
    return x.to_markdown(index=False, floatfmt=".2f")


def write_outputs(
    scored: pd.DataFrame,
    histories: dict[tuple[str, str], pd.DataFrame],
    failures: pd.DataFrame,
    output_dir: Path,
    audit: pd.DataFrame | None = None,
    audit_summary: pd.DataFrame | None = None,
    market_audit: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = output_dir / "板块完整评分.csv"
    xlsx = output_dir / "板块主线扫描.xlsx"
    dashboard = output_dir / "主线雷达.png"
    trends = output_dir / "领先板块走势.png"
    md = output_dir / "主线判断报告.md"
    html_path = output_dir / "主线判断报告.html"
    audit_xlsx = output_dir / "数据完整性审计.xlsx"
    omitted_csv = output_dir / "遗漏板块明细.csv"
    sideways_csv = output_dir / "横盘火种.csv"
    sideways_chart = output_dir / "横盘火种雷达.png"
    market_audit_path = output_dir / "全A数据质量.csv"
    if market_audit is not None and not market_audit.empty:
        market_audit.to_csv(market_audit_path, index=False, encoding="utf-8-sig")
    radar_specs = {
        "量化火种": ("ignition_score", ["ignition_score", "ignition_coverage", "absolute_strength_state", "historical_rank_change_3d", "top_rank_days_10", "mainline_blockers", "人工核查提醒"]),
        "市场确认主线": ("market_confirmation_rank_score", ["market_confirmation_score", "market_confirmation_coverage", "top_rank_days_10", "rs_market_5d", "breadth", "mainline_gate_passed"]),
        "结构强势": ("structure_score", ["structure_score", "structure_status", "mapping_coverage", "structure_coverage", "structure_breadth", "strong_stock_ratio", "limit_up_density", "consecutive_limit_up_count", "leader_follower_gap"]),
        "主线切换": ("switch_score", ["switch_score", "switch_from", "confirmation_change", "turnover_change", "market_confirmation_score"]),
        "退潮风险": ("exhaustion_rank_score", ["exhaustion_score", "exhaustion_coverage", "risk_trend", "risk_relative", "risk_volume"]),
        "潜在漏检诊断": ("ignition_score", ["ignition_score", "miss_diagnosis", "lifecycle", "mainline_blockers", "structure_status", "人工核查提醒"]),
    }
    radar_tables: dict[str, pd.DataFrame] = {}
    for title, (sort_col, columns) in radar_specs.items():
        if sort_col not in scored:
            continue
        order = scored.sort_values(sort_col, ascending=False, na_position="last")
        if title == "市场确认主线":
            order = order[order["lifecycle"].eq("Mainline")]
        elif title == "量化火种":
            order = order[order["lifecycle"].isin(["Seed", "Ignition"])]
        elif title == "结构强势":
            order = order[order["structure_score"].notna()]
        elif title == "潜在漏检诊断":
            order = order[order["miss_diagnosis"].astype(str).ne("")]
        elif title == "主线切换":
            order = order[order["switch_score"].notna()]
        radar_tables[title] = order[[c for c in ["kind", "code", "name", *columns] if c in order]]
        radar_tables[title].to_csv(output_dir / f"{title}.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(full_csv, index=False, encoding="utf-8-sig")
    sideways = scored[scored.get("sideways_seed_status", pd.Series("", index=scored.index)).isin(["横盘火种", "横盘观察"])].sort_values("sideways_seed_score", ascending=False)
    sideways.to_csv(sideways_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="完整评分", index=False)
        scored[scored["lifecycle"].eq("Mainline")].to_excel(writer, sheet_name="当前主线", index=False)
        if "lifecycle" in scored:
            scored[scored["lifecycle"].isin(["Seed", "Ignition"])].sort_values("ignition_score", ascending=False).to_excel(
                writer, sheet_name="火种雷达", index=False,
            )
        sideways.to_excel(writer, sheet_name="横盘火种", index=False)
        for title, table in radar_tables.items():
            table.to_excel(writer, sheet_name=title, index=False)
        scored[scored["status"].isin(["潜在启动", "值得关注"])].sort_values("candidate_score", ascending=False).to_excel(writer, sheet_name="潜在主线", index=False)
        failures.to_excel(writer, sheet_name="抓取失败", index=False)
        if audit_summary is not None:
            audit_summary.to_excel(writer, sheet_name="完整性汇总", index=False)
        if market_audit is not None and not market_audit.empty:
            market_audit.tail(60).to_excel(writer, sheet_name="全A数据质量", index=False)
        if audit is not None:
            audit[audit["is_omitted"]].to_excel(writer, sheet_name="遗漏板块", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
            ws.column_dimensions["B"].width = 20
    make_dashboard(scored, dashboard)
    make_trend_chart(scored, histories, trends)
    make_sideways_seed_chart(scored, sideways_chart)
    if audit is not None and audit_summary is not None:
        with pd.ExcelWriter(audit_xlsx, engine="openpyxl") as writer:
            audit_summary.to_excel(writer, sheet_name="汇总", index=False)
            audit.to_excel(writer, sheet_name="全部板块审计", index=False)
            audit[audit["is_omitted"]].to_excel(writer, sheet_name="遗漏板块", index=False)
            audit[(~audit["is_omitted"]) & (audit["audit_status"] != "完整")].to_excel(
                writer, sheet_name="质量警告", index=False,
            )
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
                ws.column_dimensions["C"].width = 20
                ws.column_dimensions["L"].width = 45
        audit[audit["is_omitted"]].to_csv(omitted_csv, index=False, encoding="utf-8-sig")

    display_scored = _deduplicate_for_display(scored)
    main = display_scored[display_scored["lifecycle"].eq("Mainline")].sort_values("mainline_score", ascending=False)
    candidate = _deduplicate_for_display(scored, "ignition_score")
    candidate = candidate[candidate.get("lifecycle", pd.Series(index=candidate.index, dtype=str)).isin(["Seed", "Ignition"])].sort_values("ignition_score", ascending=False)
    sideways_display = _deduplicate_for_display(scored, "sideways_seed_score")
    sideways_display = sideways_display[sideways_display["sideways_seed_status"].isin(["横盘火种", "横盘观察"])].sort_values("sideways_seed_score", ascending=False)
    radar_text = "\n\n".join(
        f"## {title}\n\n{table.head(20).to_markdown(index=False, floatfmt='.2f')}"
        for title, table in radar_tables.items()
    )
    manual_text = "\n".join(
        f"- {row['name']}：{row['人工核查提醒']}"
        for _, row in scored.nlargest(10, "ignition_score").iterrows()
    ) if "人工核查提醒" in scored else "本次没有人工核查条目。"
    as_of = pd.to_datetime(scored["as_of"]).max().date()
    run_mode = str(scored["run_mode"].iloc[0]) if "run_mode" in scored else "close"
    coverage_text = ""
    if audit_summary is not None and not audit_summary.empty:
        total = audit_summary[audit_summary["kind"] == "all"].iloc[0]
        coverage_text = (
            f"源板块全集：{int(total['source_universe'])}；主动过滤：{int(total['intentional_filtered'])}；"
            f"扫描目标：{int(total['scan_target'])}；最终评分：{int(total['final_scored'])}；"
            f"目标覆盖率：{total['target_coverage_pct']:.2f}%；"
            f"行情有效率：{total['history_valid_pct']:.2f}%；"
            f"主线指标平均覆盖率：{total['indicator_coverage_mean']:.1%}；"
            f"代理数据占比：{total['proxy_data_pct']:.2f}%。  \n"
        )
    report = f"""# A股量化主线扫描报告

实际行情截止：{as_of}
模型版本：quant-v2
运行模式：{'当日收盘' if run_mode == 'close' else '休市/盘中复查，未生成新交易日收盘快照'}
覆盖：{len(scored)} 个有效板块（行业 + 概念）；抓取失败/数据不足：{len(failures)} 个。
{coverage_text}

## 当前主线

{_fmt_table(main, 20) if not main.empty else '本次暂无满足确认条件的主线。以下量化火种仅为关注候选。'}

## 火种/点火板块

{_fmt_table(candidate if not candidate.empty else scored.sort_values('ignition_score', ascending=False), 20)}

## 横盘火种（低位箱体）

{_fmt_sideways_table(sideways_display, 20) if not sideways_display.empty else '当前没有满足绝对门槛的低位箱体板块。'}

{radar_text}

## 人工核查提醒（均未纳入计算）

{manual_text}

## 判定逻辑

- **主线分**：只使用可计算的量价、相对强度、广度与持续性；缺失因子保持缺失，按有效权重计算原始分并按覆盖率向 50 分收缩。覆盖率不足不能确认主线。
- **火种分**：优先使用排名跃迁、广度增量、板块成交额份额增量、同口径资金强度变化；快照历史不足时才更多依赖当日加速度与量能异常。过去 5/10 日已经大涨会扣分。
- **横盘火种分**：独立筛选 20 日窄箱体、20 日低斜率、5/20 日波动收缩、接近箱底且位于 60 日区间低位的行业或概念；至少需要 40 个交易日，绝对门槛未通过不会仅凭横截面排名入选。
- **确认分**（兼容字段 `candidate_score`）：保留原有短斜率、加速度、相对强弱和量价扩张逻辑，用于确认扩散，而不再冒充真正的早期发现分。
- 生命周期为 `Dormant → Seed → Ignition → Diffusion → Mainline → Crowded/Decay`；首次运行缺少排名和广度轨迹，连续保存快照后火种分才具备完整信息。
- 东方财富主力资金与 CMF 代理分别做横截面标准化，CMF 信号按较低置信度收缩；资金加速度也只在同口径内计算。
- **市场确认主线**只用趋势、相对强度、量能、广度和有效日数足够的持续性。
- **退潮与切换**只用趋势转弱、相对强度下降、量能衰减及实际交易日之间的确认分和成交变化。
- 政策、供需、订单、业绩预期和产业事件均为“未检查”的人工提醒，不进入分数、排名或生命周期。
- `mainline_coverage`、`ignition_coverage` 和 `market_confirmation_coverage` 显示各模型有效权重比例；成分结构未验证时会明确标记。

> 这是量价与资金行为筛选器，不是收益保证或买卖建议。板块概念存在重叠，应用时还需结合政策/事件驱动、指数环境、个股位置与风险预算复核。
"""
    md.write_text(report, encoding="utf-8")
    table_html = scored.head(100).to_html(index=False, classes="data", border=0, float_format=lambda v: f"{v:.2f}")
    radar_html = "".join(
        f"<h2>{html.escape(title)}</h2>" + table.head(20).to_html(index=False, classes="data", border=0,
                                                            float_format=lambda v: f"{v:.2f}")
        for title, table in radar_tables.items()
    )
    html_path.write_text(f"""<!doctype html><meta charset='utf-8'><title>A股主线雷达</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f7f8fa}}img{{max-width:100%;background:white}}table{{border-collapse:collapse;background:white;font-size:12px}}th,td{{padding:6px 8px;border:1px solid #ddd;white-space:nowrap}}th{{position:sticky;top:0;background:#263238;color:white}}h1{{color:#263238}}</style>
<h1>A股板块主线雷达</h1><p>数据截止 {as_of}；共 {len(scored)} 个有效板块。</p>
<img src='{html.escape(dashboard.name)}'><img src='{html.escape(sideways_chart.name)}'><img src='{html.escape(trends.name)}'>{radar_html}<h2>完整评分（前100）</h2>{table_html}
""", encoding="utf-8")
    paths = {"csv": full_csv, "sideways_csv": sideways_csv, "xlsx": xlsx, "dashboard": dashboard, "sideways_chart": sideways_chart, "trends": trends, "markdown": md, "html": html_path}
    paths.update({title: output_dir / f"{title}.csv" for title in radar_tables})
    if audit is not None:
        paths.update({"audit_xlsx": audit_xlsx, "omitted_csv": omitted_csv})
    if market_audit is not None and not market_audit.empty:
        paths["market_audit"] = market_audit_path
    return paths
