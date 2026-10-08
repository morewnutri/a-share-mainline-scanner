from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .analysis import build_metric_table, score_boards
from .audit import build_completeness_audit
from .backtest import write_backtest, replay_daily_histories
from .data import EastmoneyAkshareProvider
from .report import write_outputs
from .market_metrics import enrich_market_history, load_market
from .quant_scores import add_quant_radar, add_switch_signals
from .manual_checklist import attach_manual_checklist
from .stock_structure import enrich_candidate_structure
from .snapshot_store import SnapshotStore
from .trading_calendar import expected_close_session

DEFAULT_EXCLUDE = ""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A股行业/概念主线生命周期与火种扫描器")
    p.add_argument("--board-types", nargs="+", choices=["industry", "concept"], default=["industry", "concept"])
    p.add_argument("--lookback-calendar-days", type=int, default=120, help="抓取自然日数；默认覆盖约80个交易日以计算60日成交占比基线")
    p.add_argument("--workers", type=int, default=3, help="并发抓取数；默认保守限速，接口稳定时可调到5-8")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/latest"))
    p.add_argument("--refresh", action="store_true", help="忽略当天缓存，重新抓取")
    p.add_argument("--cache-hours", type=float, default=24, help="历史日线缓存时长")
    p.add_argument("--snapshot-cache-minutes", type=float, default=5, help="实时板块列表/资金流缓存分钟数")
    p.add_argument("--snapshot-dir", type=Path, default=Path("data/snapshots"), help="评分历史快照目录")
    p.add_argument("--no-save-snapshot", action="store_true", help="不保存本次评分历史（不建议）")
    p.add_argument(
        "--baostock-mode", choices=["off", "industry", "all"], default="off",
        help="末级回退：off关闭；industry使用BaoStock行业分类；all还会按成分股合成概念（较慢）",
    )
    p.add_argument("--baostock-max-constituents", type=int, default=24, help="每个合成板块最多抽取的成分股数")
    p.add_argument("--backtest", action="store_true", help="基于已有快照输出火种发现能力回放评估")
    p.add_argument("--historical-replay", action="store_true", help="按历史日线逐交易日重建量化信号并回测（耗时较长）")
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE, help="可选的概念过滤正则；默认不主动过滤")
    p.add_argument("--research-signals", type=Path, help="已弃用：主观研究文件不参与自动评分")
    p.add_argument("--market-history", type=Path, help="独立全A成交额与可选基准指数日线 CSV")
    p.add_argument("--market-reconciliation-tolerance", type=float, default=.05, help="提供个股成交额总和时允许的最大对账误差")
    p.add_argument("--limit", type=int, default=0, help="仅调试：每类最多抓取N个板块，0为全部")
    p.add_argument("--structure-candidates", type=int, default=20, help="自动获取个股结构的每榜候选数；0为关闭")
    p.add_argument("--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, Path]:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = EastmoneyAkshareProvider(
        args.cache_dir,
        refresh=args.refresh,
        ttl_hours=args.cache_hours,
        snapshot_ttl_minutes=args.snapshot_cache_minutes,
        baostock_mode=args.baostock_mode,
        baostock_max_constituents=args.baostock_max_constituents,
    )
    source_universes = []
    universes = []
    for kind in args.board_types:
        u = provider.get_universe(kind)
        source_universes.append(u.copy())
        if args.exclude_regex and kind == "concept":
            u = u[~u["name"].astype(str).str.contains(args.exclude_regex, regex=True, na=False)]
        if args.limit:
            u = u.head(args.limit)
        universes.append(u)
    boards = pd.concat(universes, ignore_index=True)
    source_boards = pd.concat(source_universes, ignore_index=True)
    captured_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    end = captured_at.date()
    start = end - timedelta(days=args.lookback_calendar_days)
    market_as_of = expected_close_session(pd.Timestamp(end), captured_at)
    logging.info("扫描 %d 个板块，日期 %s 至 %s", len(boards), start, end)
    fetched = provider.get_histories(boards, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), args.workers)
    flows = provider.get_fund_flows(args.board_types) if market_as_of.date() == captured_at.date() else pd.DataFrame()
    metrics = build_metric_table(boards, fetched.histories, flows)
    market = load_market(args.market_history, args.market_reconciliation_tolerance)
    metrics = enrich_market_history(metrics, fetched.histories, market)
    if args.research_signals:
        logging.warning("--research-signals 已弃用，文件内容不参与任何分数或状态判断")
    if market_as_of.date() != captured_at.date():
        for column in ("breadth", "snapshot_return", "snapshot_amount", "snapshot_turnover"):
            if column in metrics:
                metrics[column] = float("nan")
        metrics["snapshot_alignment"] = "非当日收盘，实时字段禁用"
    elif "snapshot_return" in metrics:
        discrepancy = (pd.to_numeric(metrics["snapshot_return"], errors="coerce")
                       - pd.to_numeric(metrics["ret_1d"], errors="coerce")).abs()
        mismatched = discrepancy > .75
        for column in ("breadth", "snapshot_amount", "snapshot_turnover"):
            if column in metrics:
                metrics.loc[mismatched, column] = float("nan")
        metrics["snapshot_alignment"] = "一致"
        metrics.loc[mismatched, "snapshot_alignment"] = "实时涨跌幅与日线不一致，实时字段禁用"
    preliminary = add_quant_radar(score_boards(metrics))
    store = SnapshotStore(args.snapshot_dir)
    enriched = store.enrich(preliminary, captured_at)
    scored = add_switch_signals(add_quant_radar(score_boards(enriched)))
    scored = attach_manual_checklist(enrich_candidate_structure(scored, provider, args.structure_candidates, captured_at))
    scored["run_mode"] = "close" if market_as_of.date() == captured_at.date() else "historical_replay"
    if scored.empty:
        raise RuntimeError("没有获得足够的有效板块数据，请检查网络、日期或 AKShare 接口状态")
    failures = pd.DataFrame(fetched.failures)
    audit, audit_summary = build_completeness_audit(
        source_boards, boards, fetched.histories, fetched.failures, flows, scored,
    )
    paths = write_outputs(
        scored, fetched.histories, failures, args.output_dir, audit, audit_summary, market,
    )
    if not args.no_save_snapshot:
        paths["snapshot"] = store.save(scored, captured_at)
    if args.backtest:
        paths.update(write_backtest(args.snapshot_dir, args.output_dir / "backtest"))
    if args.historical_replay:
        paths.update(replay_daily_histories(boards, fetched.histories, market,
                                           args.output_dir / "historical_replay"))
    print("\n=== 数据完整性 ===")
    print(audit_summary.to_string(index=False))
    show_cols = ["kind", "name", "lifecycle", "mainline_score", "ignition_score", "confirmation_score", "ret_5d", "ret_10d", "slope_5d", "acceleration"]
    print("\n=== 当前主线 Top 20 ===")
    current_mainline = scored[scored["lifecycle"].eq("Mainline")]
    print(current_mainline[show_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print("\n=== 潜在主线 Top 20 ===")
    potential = scored[scored["lifecycle"].isin(["Seed", "Ignition"])]
    print(potential.sort_values("ignition_score", ascending=False)[show_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    sideways_cols = [
        "kind", "name", "sideways_seed_status", "sideways_seed_score", "box_range_20d_pct",
        "range_position_60d_pct", "distance_high_60d_pct", "slope_20d", "volatility_ratio_5_20",
    ]
    sideways = scored[scored["sideways_seed_status"].isin(["横盘火种", "横盘观察"])]
    print("\n=== 横盘火种 Top 20 ===")
    print(sideways.sort_values("sideways_seed_score", ascending=False)[sideways_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print("\n=== 量化火种与漏检候选 Top 20 ===")
    research = scored.sort_values("ignition_score", ascending=False, na_position="last")
    print(research[["kind", "name", "ignition_score", "ignition_coverage", "mainline_blockers", "人工核查提醒"]].head(20).to_string(index=False))
    print("\n=== 退潮/切换监测 Top 20 ===")
    risk = scored.sort_values("exhaustion_rank_score", ascending=False, na_position="last")
    print(risk[["kind", "name", "exhaustion_rank_score", "exhaustion_score", "exhaustion_coverage", "switch_score", "switch_from"]].head(20).to_string(index=False))
    print(f"\n报告已写入: {args.output_dir.resolve()}")
    return paths


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
