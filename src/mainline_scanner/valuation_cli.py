from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .valuation import ValuationDataProvider, ValuationEngine, load_config


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["revenue_growth", "profit_growth", "value_deviation", "profitable_mcap_coverage", "data_coverage"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce") * 100
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="A股板块/个股多模型估值扫描器")
    ap.add_argument("--config", default="valuation_config.json")
    ap.add_argument("--cache-dir", default="data/valuation_cache")
    ap.add_argument("--state-dir", default="data/valuation_state")
    ap.add_argument("--output-dir", default="reports/valuation/latest")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    provider = ValuationDataProvider(Path(args.cache_dir), refresh=args.refresh)
    engine = ValuationEngine(provider, cfg, Path(args.state_dir))

    sector_rows = [engine.evaluate_sector(name, c) for name, c in cfg["sectors"].items()]
    stock_rows = [engine.evaluate_stock(code, info) for code, info in cfg.get("stocks", {}).items()]
    engine.save_snapshots(sector_rows)

    sectors = _format_output(pd.DataFrame(sector_rows))
    stocks = _format_output(pd.DataFrame(stock_rows))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sectors.to_csv(out / "板块估值.csv", index=False, encoding="utf-8-sig")
    stocks.to_csv(out / "个股估值.csv", index=False, encoding="utf-8-sig")

    # Easy-to-read workbook, but CSV is always written first so Excel failure is non-fatal.
    try:
        with pd.ExcelWriter(out / "A股主线估值.xlsx", engine="openpyxl") as w:
            sectors.to_excel(w, sheet_name="板块估值", index=False)
            stocks.to_excel(w, sheet_name="个股估值", index=False)
            engine.master.to_excel(w, sheet_name="底层财务数据", index=False)
    except Exception as exc:
        print(f"[WARN] Excel output failed: {exc}")

    cols_sector = [c for c in ["entity", "model", "current_primary", "fair_primary", "value_deviation", "valuation_label", "revenue_growth", "profit_growth", "data_coverage"] if c in sectors.columns]
    cols_stock = [c for c in ["entity", "code", "sector", "current_primary", "fair_primary", "value_deviation", "valuation_label", "growth_gate", "revenue_growth", "profit_growth"] if c in stocks.columns]
    print("\n=== 板块估值 ===")
    print(sectors[cols_sector].to_string(index=False))
    print("\n=== 个股估值 ===")
    print(stocks[cols_stock].to_string(index=False))
    print(f"\n输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()
