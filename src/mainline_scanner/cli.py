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
from .backtest import write_backtest
from .data import EastmoneyAkshareProvider
from .report import write_outputs
from .research import (add_research_scores, add_switch_signals, enrich_market_history,
                       load_market, load_observations)
from .snapshot_store import SnapshotStore

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
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE, help="可选的概念过滤正则；默认不主动过滤")
    p.add_argument("--research-signals", type=Path, help="按 available_at 生效的板块基本面/事件/结构信号 CSV")
    p.add_argument("--market-history", type=Path, help="独立全A成交额与可选基准指数日线 CSV")
    p.add_argument("--market-reconciliation-tolerance", type=float, default=.05, help="提供个股成交额总和时允许的最大对账误差")
    p.add_argument("--limit", type=int, default=0, help="仅调试：每类最多抓取N个板块，0为全部")
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
    end = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    start = end - timedelta(days=args.lookback_calendar_days)
    logging.info("扫描 %d 个板块，日期 %s 至 %s", len(boards), start, end)
    fetched = provider.get_histories(boards, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), args.workers)
    flows = provider.get_fund_flows(args.board_types)
    metrics = build_metric_table(boards, fetched.histories, flows)
    market = load_market(args.market_history, args.market_reconciliation_tolerance)
    metrics = enrich_market_history(metrics, fetched.histories, market)
    observations = load_observations(args.research_signals)
    captured_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    preliminary = add_research_scores(score_boards(metrics), observations, pd.Timestamp(captured_at))
    store = SnapshotStore(args.snapshot_dir)
    enriched = store.enrich(preliminary, captured_at)
    scored = add_switch_signals(add_research_scores(score_boards(enriched), observations, pd.Timestamp(captured_at)))
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
    print("\n=== 数据完整性 ===")
    print(audit_summary.to_string(index=False))
    show_cols = ["kind", "name", "lifecycle", "mainline_score", "ignition_score", "confirmation_score", "ret_5d", "ret_10d", "slope_5d", "acceleration"]
    print("\n=== 当前主线 Top 20 ===")
    current_mainline = scored[scored["status"].astype(str).str.startswith("主线")]
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
    print("\n=== 研究潜在主线 Top 20 ===")
    research = scored.sort_values(["potential_rank_score", "market_confirmation_rank_score"], ascending=False, na_position="last")
    print(research[["kind", "name", "potential_rank_score", "potential_score", "potential_coverage", "market_confirmation_score", "potential_state"]].head(20).to_string(index=False))
    print("\n=== 退潮/切换监测 Top 20 ===")
    risk = scored.sort_values("exhaustion_rank_score", ascending=False, na_position="last")
    print(risk[["kind", "name", "exhaustion_rank_score", "exhaustion_score", "exhaustion_coverage", "switch_score", "switch_from"]].head(20).to_string(index=False))
    print(f"\n报告已写入: {args.output_dir.resolve()}")
    return paths


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
