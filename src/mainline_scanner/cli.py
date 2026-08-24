from __future__ import annotations

import argparse
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .analysis import build_metric_table, score_boards
from .audit import build_completeness_audit
from .data import EastmoneyAkshareProvider
from .report import write_outputs

DEFAULT_EXCLUDE = r"昨日|融资融券|沪股通|深股通|MSCI|富时罗素|标准普尔|证金持股|QFII|机构重仓|预盈预增|转债标的|破净股"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A股行业/概念板块主线扫描器")
    p.add_argument("--board-types", nargs="+", choices=["industry", "concept"], default=["industry", "concept"])
    p.add_argument("--lookback-calendar-days", type=int, default=75, help="抓取自然日数；默认覆盖约50个交易日")
    p.add_argument("--workers", type=int, default=3, help="并发抓取数；默认保守限速，接口稳定时可调到5-8")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/latest"))
    p.add_argument("--refresh", action="store_true", help="忽略当天缓存，重新抓取")
    p.add_argument("--cache-hours", type=float, default=8)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE, help="过滤非主题型概念；传空字符串可关闭")
    p.add_argument("--limit", type=int, default=0, help="仅调试：每类最多抓取N个板块，0为全部")
    p.add_argument("--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, Path]:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = EastmoneyAkshareProvider(args.cache_dir, refresh=args.refresh, ttl_hours=args.cache_hours)
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
    end = date.today()
    start = end - timedelta(days=args.lookback_calendar_days)
    logging.info("扫描 %d 个板块，日期 %s 至 %s", len(boards), start, end)
    fetched = provider.get_histories(boards, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), args.workers)
    flows = provider.get_fund_flows(args.board_types)
    metrics = build_metric_table(boards, fetched.histories, flows)
    scored = score_boards(metrics)
    if scored.empty:
        raise RuntimeError("没有获得足够的有效板块数据，请检查网络、日期或 AKShare 接口状态")
    failures = pd.DataFrame(fetched.failures)
    audit, audit_summary = build_completeness_audit(
        source_boards, boards, fetched.histories, fetched.failures, flows, scored,
    )
    paths = write_outputs(
        scored, fetched.histories, failures, args.output_dir, audit, audit_summary,
    )
    print("\n=== 数据完整性 ===")
    print(audit_summary.to_string(index=False))
    show_cols = ["kind", "name", "status", "mainline_score", "candidate_score", "ret_5d", "ret_10d", "slope_5d", "acceleration"]
    print("\n=== 当前主线 Top 20 ===")
    current_mainline = scored[scored["status"].astype(str).str.startswith("主线")]
    print(current_mainline[show_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print("\n=== 潜在主线 Top 20 ===")
    potential = scored[scored["status"].isin(["潜在启动", "值得关注"])]
    print(potential.sort_values("candidate_score", ascending=False)[show_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"\n报告已写入: {args.output_dir.resolve()}")
    return paths


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
