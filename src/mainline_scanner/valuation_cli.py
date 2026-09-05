from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .valuation import ValuationDataProvider, ValuationEngine, load_config

_STOCK_CODE_RE = re.compile(r"^\d{6}$")
_NEW_STOCK_PROMPT = "请输入要添加的股票，格式：代码[,名称[,板块]]（例如 600000,浦发银行,银行）"


def _cli_fallback(prompt: str) -> str | None:
    try:
        return input(f"\n{prompt}\n> ")
    except EOFError:
        return None


def _ask_new_stock(prompt: str = _NEW_STOCK_PROMPT) -> str | None:
    """Pop up a small dialog asking which stock to add.

    The repository has no existing GUI dependency, so this uses the Python
    standard library's ``tkinter`` (available on virtually every desktop
    Python install) to avoid adding a new dependency. When no display/Tk is
    available -- e.g. this repo's CI or a headless server -- it gracefully
    falls back to a plain command-line prompt so the run never crashes.
    Returns ``None`` when the user cancels (or input cannot be read at all).
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except ImportError:
        return _cli_fallback(prompt)

    try:
        root = tk.Tk()
    except tk.TclError:
        # No display/window manager available (headless environment).
        return _cli_fallback(prompt)

    try:
        root.withdraw()
        return simpledialog.askstring("添加股票", prompt, parent=root)
    finally:
        root.destroy()


def parse_new_stock_input(raw: str | None) -> tuple[str, str | None, str | None] | None:
    """Parse the free-form dialog/CLI text into ``(code, name, sector)``.

    Accepts comma (halfwidth/fullwidth) or whitespace separated fields; only
    the code is required. Returns ``None`` when the input is missing/blank,
    which covers both a cancelled dialog and an empty submission.
    """
    if raw is None:
        return None
    parts = [p.strip() for p in re.split(r"[,\uFF0C\s]+", raw.strip()) if p.strip()]
    if not parts:
        return None
    code = parts[0]
    name = parts[1] if len(parts) > 1 else None
    sector = parts[2] if len(parts) > 2 else None
    return code, name, sector


def add_stock_to_universe(
    engine: ValuationEngine, cfg: dict[str, Any], code: str, name: str | None, sector: str | None
) -> dict[str, Any]:
    """Validate, register and value a newly added stock.

    This reuses ``ValuationEngine.evaluate_stock`` so the new stock goes
    through the exact same sector-driven model dispatch (including the
    engine's own fallback model for stocks with an unconfigured sector) that
    every other stock in ``cfg["stocks"]`` already uses -- no model is
    hardcoded here. On success, ``code`` is added to ``cfg["stocks"]``;
    on any error the config is left untouched.
    """
    stocks = cfg.setdefault("stocks", {})
    if not _STOCK_CODE_RE.match(code):
        return {"error": f"股票代码 '{code}' 不合法，应为6位数字"}
    if code in stocks:
        existing_name = stocks[code].get("name", "")
        return {"error": f"股票 {code}（{existing_name}）已存在于股票列表，无需重复添加"}
    warnings: list[str] = []
    if sector and sector not in cfg.get("sectors", {}):
        msg = f"板块 '{sector}' 未在配置中定义，股票 {code} 将按项目对未配置板块的默认估值逻辑处理"
        print(f"[提示] {msg}")
        warnings.append(msg)
    if not name:
        match = engine.master.loc[engine.master["code"] == code, "name"]
        if not match.empty:
            name = str(match.iloc[0])
        else:
            name = code
            msg = f"未在行情数据中找到股票 {code} 的名称，暂以代码作为名称展示"
            print(f"[提示] {msg}")
            warnings.append(msg)
    info: dict[str, Any] = {"name": name}
    if sector:
        info["sector"] = sector
    try:
        row = engine.evaluate_stock(code, info)
    except Exception as exc:
        return {"error": f"股票 {code}（{name}）估值计算失败：{exc}"}
    if row.get("error"):
        return {"error": f"股票 {code}（{name}）{row['error']}"}
    if warnings:
        row = dict(row)
        row["warnings"] = warnings
    stocks[code] = info
    return row


def persist_config(cfg: dict[str, Any], path: Path) -> bool:
    """Persist the (possibly updated) config back to its JSON file.

    The project already persists its stock list via ``valuation_config.json``
    (loaded by ``load_config``), so newly added stocks are written back the
    same way. If writing fails for any reason, the failure is reported and
    the newly added stock simply remains in effect only for this run.
    """
    try:
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True
    except (OSError, TypeError, ValueError) as exc:
        print(f"[警告] 无法写入配置文件 {path}：{exc}（新增股票仅在本次运行内存中生效，未持久化）")
        return False


def _is_interactive_terminal() -> bool:
    """Best-effort check for a real interactive terminal on both ends, so the
    new-stock prompt is not offered (and never blocks) in CI/scheduled/piped
    runs unless explicitly requested via --add-stock."""
    return sys.stdin.isatty() and sys.stdout.isatty()


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
    ap.add_argument(
        "--add-stock", dest="add_stock", action="store_true", default=None,
        help="运行结束后强制弹出新增股票对话框/命令行输入（默认仅在交互式终端下自动开启）",
    )
    ap.add_argument(
        "--no-add-stock", dest="add_stock", action="store_false",
        help="跳过运行结束后的新增股票交互，适合定时任务/CI等非交互场景",
    )
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

    # After the normal run finishes, offer to add one more stock on the fly.
    # It is immediately valued with the same sector-driven model dispatch and
    # (if successful) appended to the stock list for future runs too. Default
    # to only doing this in an interactive terminal so scheduled/CI/scripted
    # runs never block waiting for input; --add-stock/--no-add-stock override.
    prompt_enabled = args.add_stock if args.add_stock is not None else _is_interactive_terminal()
    if not prompt_enabled:
        print("\n[提示] 当前为非交互环境，跳过新增股票交互（可加 --add-stock 强制开启）。")
        return
    raw_new_stock = _ask_new_stock()
    parsed = parse_new_stock_input(raw_new_stock)
    if parsed is None:
        print("\n[提示] 未输入股票（已取消或输入为空），跳过新增股票。")
        return
    code, name, sector = parsed
    result = add_stock_to_universe(engine, cfg, code, name, sector)
    if result.get("error"):
        print(f"\n[错误] {result['error']}，未加入股票列表。")
        return
    if persist_config(cfg, Path(args.config)):
        print(f"\n[提示] 已将股票 {code} 加入 {args.config} 并持久化，后续运行将自动纳入估值。")
    new_stock_df = _format_output(pd.DataFrame([result]))
    cols_new = [c for c in cols_stock if c in new_stock_df.columns]
    print("\n=== 新增股票估值 ===")
    print(new_stock_df[cols_new].to_string(index=False))


if __name__ == "__main__":
    main()
