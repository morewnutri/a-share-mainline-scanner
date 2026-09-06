from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .valuation import (
    ReportSelection,
    ValuationDataProvider,
    ValuationEngine,
    apply_report_selection,
    load_config,
    select_report_periods,
)


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in [
        "revenue_growth",
        "profit_growth",
        "value_deviation",
        "profitable_mcap_coverage",
        "data_coverage",
    ]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce") * 100
    return x


def parse_stock_codes(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    codes = re.findall(r"(?<!\d)(\d{6})(?!\d)", str(text))
    return list(dict.fromkeys(codes))


def custom_stock_path(state_dir: Path) -> Path:
    return Path(state_dir) / "custom_stocks.json"


def load_custom_stocks(state_dir: Path) -> dict[str, dict[str, Any]]:
    path = custom_stock_path(state_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"无法读取自定义股票池 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"自定义股票池格式错误，顶层必须是对象: {path}")
    out: dict[str, dict[str, Any]] = {}
    for code, info in data.items():
        match = re.search(r"(\d{6})", str(code))
        if not match or not isinstance(info, dict):
            continue
        out[match.group(1)] = dict(info)
    return out


def save_custom_stocks(state_dir: Path, stocks: dict[str, dict[str, Any]]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = custom_stock_path(state_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stocks, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def merge_stock_pools(
    default_stocks: dict[str, dict[str, Any]],
    custom_stocks: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge pools without allowing a custom duplicate to overwrite a curated default."""
    merged = {str(code): dict(info) for code, info in default_stocks.items()}
    for code, info in custom_stocks.items():
        if code not in merged:
            merged[code] = dict(info)
    return merged


def _resolve_new_stock_info(
    code: str,
    provider: ValuationDataProvider,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    spot = provider.spot()
    hit = spot[spot["code"].astype(str) == str(code)]
    if hit.empty:
        return None
    name = str(hit.iloc[0]["name"])
    sector = provider.infer_stock_sector(code, cfg)
    return {
        "name": name,
        "sector": sector,
        "source": "user",
        "added_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def add_custom_stocks(
    codes: list[str],
    provider: ValuationDataProvider,
    cfg: dict[str, Any],
    state_dir: Path,
    custom: dict[str, dict[str, Any]],
    default_stocks: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    added: list[str] = []
    for code in codes:
        if code in default_stocks:
            print(f"[INFO] {code} 已在默认股票池中，无需重复添加。")
            continue
        info = _resolve_new_stock_info(code, provider, cfg)
        if info is None:
            print(f"[WARN] {code} 未在当前A股行情中找到，已跳过。")
            continue
        previous = custom.get(code)
        custom[code] = info
        added.append(code)
        action = "更新" if previous else "加入"
        print(f"[OK] {action}: {code} {info['name']} -> {info['sector']}")
    if added:
        path = save_custom_stocks(state_dir, custom)
        print(f"[INFO] 自定义股票池已保存: {path}")
    return custom, added


def _freshness_summary(
    selection: ReportSelection,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    default_count: int,
    custom_count: int,
) -> pd.DataFrame:
    min_cov = float(cfg.get("report_policy", {}).get("min_financial_coverage", 0.80))
    minimum = min(selection.current_coverage, selection.prior_coverage, selection.annual_coverage)
    status = "PASS" if minimum >= min_cov else "WARN"
    rows = [
        ("运行时间", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("是否强制刷新", "YES" if args.refresh else "NO"),
        ("行情说明", "数据源实时抓取；非交易时段的最新价通常为最近交易快照"),
        ("财报选择策略", selection.source),
        ("当前财报期", selection.current),
        ("同比财报期", selection.prior),
        ("TTM年度底稿", selection.annual),
        ("当前财报覆盖率", selection.current_coverage),
        ("同比财报覆盖率", selection.prior_coverage),
        ("年度底稿覆盖率", selection.annual_coverage),
        ("最低允许覆盖率", min_cov),
        ("数据状态", status),
        ("默认股票数", default_count),
        ("自定义股票数", custom_count),
        ("总估值股票数", default_count + custom_count),
        ("缓存目录", str(Path(args.cache_dir).resolve())),
        ("状态目录", str(Path(args.state_dir).resolve())),
        ("输出目录", str(Path(args.output_dir).resolve())),
    ]
    return pd.DataFrame(rows, columns=["项目", "当前值"])


def write_outputs(
    sector_rows: list[dict[str, Any]],
    stock_rows: list[dict[str, Any]],
    engine: ValuationEngine,
    provider: ValuationDataProvider,
    selection: ReportSelection,
    args: argparse.Namespace,
    default_count: int,
    custom_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    sectors = _format_output(pd.DataFrame(sector_rows))
    stocks = _format_output(pd.DataFrame(stock_rows))
    freshness = _freshness_summary(selection, args, engine.cfg, default_count, custom_count)
    fetch_audit = provider.freshness_frame()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sectors.to_csv(out / "板块估值.csv", index=False, encoding="utf-8-sig")
    stocks.to_csv(out / "个股估值.csv", index=False, encoding="utf-8-sig")
    freshness.to_csv(out / "数据新鲜度.csv", index=False, encoding="utf-8-sig")
    fetch_audit.to_csv(out / "数据抓取审计.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(out / "A股主线估值.xlsx", engine="openpyxl") as w:
            sectors.to_excel(w, sheet_name="板块估值", index=False)
            stocks.to_excel(w, sheet_name="个股估值", index=False)
            freshness.to_excel(w, sheet_name="数据新鲜度", index=False)
            fetch_audit.to_excel(w, sheet_name="数据抓取审计", index=False)
            engine.master.to_excel(w, sheet_name="底层财务数据", index=False)
    except Exception as exc:
        print(f"[WARN] Excel output failed: {exc}")

    return sectors, stocks, out


def print_results(sectors: pd.DataFrame, stocks: pd.DataFrame, out: Path) -> None:
    cols_sector = [
        c
        for c in [
            "entity",
            "model",
            "current_primary",
            "fair_primary",
            "value_deviation",
            "valuation_label",
            "revenue_growth",
            "profit_growth",
            "data_coverage",
        ]
        if c in sectors.columns
    ]
    cols_stock = [
        c
        for c in [
            "entity",
            "code",
            "sector",
            "stock_source",
            "current_primary",
            "fair_primary",
            "value_deviation",
            "valuation_label",
            "growth_gate",
            "revenue_growth",
            "profit_growth",
        ]
        if c in stocks.columns
    ]
    print("\n=== 板块估值 ===")
    if len(sectors):
        print(sectors[cols_sector].to_string(index=False))
    else:
        print("无板块结果")
    print("\n=== 个股估值 ===")
    if len(stocks):
        print(stocks[cols_stock].to_string(index=False))
    else:
        print("无个股结果")
    print(f"\n输出目录: {out.resolve()}")


def prompt_stock_codes() -> list[str]:
    print(
        "\n如需永久加入新的估值股票，请输入6位A股代码。\n"
        "支持多个代码，用空格/逗号分隔，例如：688256, 601138, 300274\n"
        "直接回车则结束。"
    )
    try:
        raw = input("\n新增股票代码：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] 当前运行环境没有可用交互输入，跳过新增股票。")
        return []
    return parse_stock_codes(raw)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="A股板块/个股多模型估值扫描器")
    ap.add_argument("--config", default="valuation_config.json")
    ap.add_argument("--cache-dir", default="data/valuation_cache")
    ap.add_argument("--state-dir", default="data/valuation_state")
    ap.add_argument("--output-dir", default="reports/valuation/latest")
    ap.add_argument("--refresh", action="store_true", help="本次进程内每个数据集绕过磁盘缓存抓取一次")
    ap.add_argument("--no-prompt", action="store_true", help="运行结束后不询问新增股票")
    ap.add_argument("--add-stocks", nargs="*", default=[], help="非交互加入股票，例如 --add-stocks 688256 601138")
    ap.add_argument("--remove-stocks", nargs="*", default=[], help="从自定义股票池移除；默认配置股票不会删除")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    cfg_raw = load_config(Path(args.config))
    provider = ValuationDataProvider(Path(args.cache_dir), refresh=args.refresh)

    # report_date=auto is resolved before ValuationEngine is built. The selected
    # dates are also written into the freshness sheet, so the user can verify
    # exactly which financial period powered each run.
    selection = select_report_periods(provider, cfg_raw)
    cfg = apply_report_selection(cfg_raw, selection)

    default_stocks = {str(k): dict(v) for k, v in cfg.get("stocks", {}).items()}
    custom = load_custom_stocks(state_dir)

    remove_codes = parse_stock_codes(args.remove_stocks)
    removed = []
    for code in remove_codes:
        if code in custom:
            custom.pop(code, None)
            removed.append(code)
        elif code in default_stocks:
            print(f"[WARN] {code} 属于 valuation_config.json 默认股票池，--remove-stocks 不会删除它。")
    if removed:
        save_custom_stocks(state_dir, custom)
        print(f"[OK] 已从自定义股票池移除: {', '.join(removed)}")

    pre_add = parse_stock_codes(args.add_stocks)
    if pre_add:
        custom, _ = add_custom_stocks(pre_add, provider, cfg, state_dir, custom, default_stocks)

    cfg["stocks"] = merge_stock_pools(default_stocks, custom)
    engine = ValuationEngine(provider, cfg, state_dir)

    sector_rows = [engine.evaluate_sector(name, c) for name, c in cfg["sectors"].items()]
    stock_rows = [engine.evaluate_stock(code, info) for code, info in cfg.get("stocks", {}).items()]
    engine.save_snapshots(sector_rows)

    sectors, stocks, out = write_outputs(
        sector_rows,
        stock_rows,
        engine,
        provider,
        selection,
        args,
        default_count=len(default_stocks),
        custom_count=sum(1 for code in custom if code not in default_stocks),
    )
    print_results(sectors, stocks, out)

    if args.no_prompt:
        return

    new_codes = prompt_stock_codes()
    if not new_codes:
        return

    custom, added = add_custom_stocks(new_codes, provider, cfg, state_dir, custom, default_stocks)
    if not added:
        return

    # Re-evaluate the full merged pool so the final CSV/XLSX contains the exact
    # same complete list that will be used automatically on the next run.
    cfg["stocks"] = merge_stock_pools(default_stocks, custom)
    stock_rows = [engine.evaluate_stock(code, info) for code, info in cfg["stocks"].items()]
    sectors, stocks, out = write_outputs(
        sector_rows,
        stock_rows,
        engine,
        provider,
        selection,
        args,
        default_count=len(default_stocks),
        custom_count=sum(1 for code in custom if code not in default_stocks),
    )
    print("\n=== 新增股票后已重新生成完整估值结果 ===")
    print_results(sectors, stocks, out)


if __name__ == "__main__":
    main()
