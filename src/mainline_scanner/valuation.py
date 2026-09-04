from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from .data import EastmoneyAkshareProvider
except Exception:  # pragma: no cover - allows standalone import during unit tests
    EastmoneyAkshareProvider = None  # type: ignore


def _num(s: pd.Series | Any) -> pd.Series | float:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
    try:
        return float(str(s).replace(",", "").replace("%", ""))
    except Exception:
        return float("nan")


def _norm_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[\s_（）()\-—·/]+", "", x)
    return re.sub(r"(概念|板块|行业)$", "", x)


def _pick(columns: Iterable[str], *patterns: str) -> str | None:
    cols = [str(c) for c in columns]
    for p in patterns:
        if p in cols:
            return p
    for p in patterns:
        for c in cols:
            if p in c:
                return c
    return None


def _clip(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(min(max(x, lo), hi))


def _geomean_ratio(items: list[tuple[float, float]]) -> float:
    """items = [(ratio, weight)], returns weighted geometric mean ratio."""
    vals = [(r, w) for r, w in items if np.isfinite(r) and r > 0 and w > 0]
    if not vals:
        return float("nan")
    total_w = sum(w for _, w in vals)
    return float(math.exp(sum(w * math.log(r) for r, w in vals) / total_w))


def valuation_label(deviation: float) -> str:
    """Deviation > 0 means market valuation is above fair value."""
    if not np.isfinite(deviation):
        return "无法判断"
    if deviation <= -0.25:
        return "低估"
    if deviation <= -0.10:
        return "合理偏低"
    if deviation < 0.15:
        return "合理"
    if deviation < 0.35:
        return "偏高"
    if deviation < 0.70:
        return "高估"
    return "显著高估"


@dataclass
class BoardResolved:
    kind: str
    requested_alias: str
    board_name: str
    board_code: str


class ValuationDataProvider:
    """
    Valuation data layer designed to sit on top of the existing repository's
    EastmoneyAkshareProvider, thereby reusing its AKShare object, Eastmoney
    retry/fallback logic and disk cache convention.
    """

    def __init__(self, cache_dir: Path, refresh: bool = False, ttl_hours: float = 12):
        if EastmoneyAkshareProvider is None:
            raise RuntimeError("请把 valuation.py 放入原项目 src/mainline_scanner/ 下运行")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.ttl = timedelta(hours=ttl_hours)
        self.base = EastmoneyAkshareProvider(cache_dir=self.cache_dir / "market", refresh=refresh)
        self.ak = self.base.ak

    def _cached_frame(self, key: str, loader, ttl: timedelta | None = None) -> pd.DataFrame:
        p = self.cache_dir / f"{key}.csv"
        t = ttl or self.ttl
        if not self.refresh and p.exists() and datetime.now() - datetime.fromtimestamp(p.stat().st_mtime) <= t:
            return pd.read_csv(p, dtype={"代码": str, "股票代码": str}, encoding="utf-8-sig")
        df = loader()
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"{key} 数据源未返回 DataFrame")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        return df

    def spot(self) -> pd.DataFrame:
        def load_ak() -> pd.DataFrame:
            try:
                return self.ak.stock_zh_a_spot_em()
            except Exception:
                # Reuse the repository's direct Eastmoney JSON route as a fallback.
                rows: list[dict] = []
                page = 1
                while True:
                    params = {
                        "pn": page,
                        "pz": 200,
                        "po": 1,
                        "np": 1,
                        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                        "fltt": 2,
                        "invt": 2,
                        "fid": "f3",
                        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                        "fields": "f2,f3,f9,f12,f14,f20,f21,f23"
                    }
                    payload = self.base._eastmoney_json("/api/qt/clist/get", params)
                    data = payload.get("data") or {}
                    diff = data.get("diff") or []
                    rows.extend(diff)
                    if not diff or len(rows) >= int(data.get("total") or 0):
                        break
                    page += 1
                return pd.DataFrame(rows).rename(columns={
                    "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅",
                    "f9": "市盈率-动态", "f20": "总市值", "f21": "流通市值", "f23": "市净率"
                })

        df = self._cached_frame("a_spot", load_ak, timedelta(minutes=5)).copy()
        code = _pick(df.columns, "代码", "股票代码")
        name = _pick(df.columns, "名称", "股票简称")
        mv = _pick(df.columns, "总市值")
        pe = _pick(df.columns, "市盈率-动态", "市盈率")
        pb = _pick(df.columns, "市净率")
        price = _pick(df.columns, "最新价")
        out = pd.DataFrame({
            "code": df[code].astype(str).str.extract(r"(\d{6})", expand=False) if code else "",
            "name": df[name].astype(str) if name else "",
            "price": _num(df[price]) if price else np.nan,
            "market_cap": _num(df[mv]) if mv else np.nan,
            "pe_dynamic": _num(df[pe]) if pe else np.nan,
            "pb": _num(df[pb]) if pb else np.nan,
        })
        return out.drop_duplicates("code")

    def performance(self, date: str) -> pd.DataFrame:
        def load() -> pd.DataFrame:
            return self.ak.stock_yjbb_em(date=date)
        raw = self._cached_frame(f"performance_{date}", load, timedelta(hours=18)).copy()
        code_col = _pick(raw.columns, "股票代码", "代码")
        name_col = _pick(raw.columns, "股票简称", "名称")
        rev_col = _pick(raw.columns, "营业收入-营业收入", "营业收入")
        rev_yoy_col = _pick(raw.columns, "营业收入-同比增长", "营业收入同比增长")
        profit_col = _pick(raw.columns, "净利润-净利润", "归属于母公司所有者的净利润", "净利润")
        profit_yoy_col = _pick(raw.columns, "净利润-同比增长", "净利润同比增长")
        roe_col = _pick(raw.columns, "净资产收益率")
        ocfps_col = _pick(raw.columns, "每股经营现金流量", "每股经营现金流")
        eps_col = _pick(raw.columns, "每股收益")
        gross_col = _pick(raw.columns, "销售毛利率", "毛利率")
        out = pd.DataFrame({
            "code": raw[code_col].astype(str).str.extract(r"(\d{6})", expand=False) if code_col else "",
            "name_fin": raw[name_col].astype(str) if name_col else "",
            "revenue": _num(raw[rev_col]) if rev_col else np.nan,
            "revenue_yoy": _num(raw[rev_yoy_col]) if rev_yoy_col else np.nan,
            "net_profit": _num(raw[profit_col]) if profit_col else np.nan,
            "profit_yoy": _num(raw[profit_yoy_col]) if profit_yoy_col else np.nan,
            "roe_h1_pct": _num(raw[roe_col]) if roe_col else np.nan,
            "ocfps": _num(raw[ocfps_col]) if ocfps_col else np.nan,
            "eps": _num(raw[eps_col]) if eps_col else np.nan,
            "gross_margin_pct": _num(raw[gross_col]) if gross_col else np.nan,
        })
        return out.drop_duplicates("code")

    def fundamentals_ttm(self, current: str, prior_h1: str, annual: str) -> pd.DataFrame:
        cur = self.performance(current).add_suffix("_cur").rename(columns={"code_cur": "code"})
        pri = self.performance(prior_h1).add_suffix("_pri").rename(columns={"code_pri": "code"})
        ann = self.performance(annual).add_suffix("_ann").rename(columns={"code_ann": "code"})
        x = cur.merge(pri, on="code", how="outer").merge(ann, on="code", how="outer")
        x["ttm_revenue"] = x["revenue_ann"] + x["revenue_cur"] - x["revenue_pri"]
        x["ttm_profit"] = x["net_profit_ann"] + x["net_profit_cur"] - x["net_profit_pri"]
        # Safe fallback for missing annual results: annualize H1 and lower confidence later.
        x["ttm_revenue"] = x["ttm_revenue"].where(x["ttm_revenue"].notna(), x["revenue_cur"] * 2)
        x["ttm_profit"] = x["ttm_profit"].where(x["ttm_profit"].notna(), x["net_profit_cur"] * 2)
        return x

    def board_catalog(self, kind: str) -> pd.DataFrame:
        raw = self.base._direct_universe(kind).copy()
        return raw[["板块名称", "板块代码"]].drop_duplicates()

    def resolve_board(self, kind: str, aliases: list[str]) -> BoardResolved | None:
        cat = self.board_catalog(kind)
        rows = [(str(r["板块名称"]), str(r["板块代码"])) for _, r in cat.iterrows()]
        normalized = [(_norm_name(n), n, c) for n, c in rows]
        for alias in aliases:
            a = _norm_name(alias)
            exact = [x for x in normalized if x[0] == a]
            if exact:
                _, n, c = exact[0]
                return BoardResolved(kind, alias, n, c)
        for alias in aliases:
            a = _norm_name(alias)
            fuzzy = [x for x in normalized if a in x[0] or x[0] in a]
            if fuzzy:
                fuzzy.sort(key=lambda z: abs(len(z[0]) - len(a)))
                _, n, c = fuzzy[0]
                return BoardResolved(kind, alias, n, c)
        return None

    def board_constituents(self, resolved: BoardResolved) -> pd.DataFrame:
        key = f"board_cons_{resolved.kind}_{resolved.board_code}"

        def load() -> pd.DataFrame:
            # AKShare first; direct Eastmoney fallback, mirroring the repo's data strategy.
            try:
                if resolved.kind == "industry":
                    return self.ak.stock_board_industry_cons_em(symbol=resolved.board_name)
                return self.ak.stock_board_concept_cons_em(symbol=resolved.board_name)
            except Exception:
                params = {
                    "pn": 1, "pz": 1000, "po": 1, "np": 1,
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": 2, "invt": 2,
                    "fid": "f3", "fs": f"b:{resolved.board_code} f:!50",
                    "fields": "f2,f3,f9,f12,f14,f20,f23"
                }
                payload = self.base._eastmoney_json("/api/qt/clist/get", params)
                return pd.DataFrame((payload.get("data") or {}).get("diff") or []).rename(columns={
                    "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅",
                    "f9": "市盈率-动态", "f20": "总市值", "f23": "市净率"
                })

        raw = self._cached_frame(key, load, timedelta(minutes=30))
        code_col = _pick(raw.columns, "代码", "股票代码")
        name_col = _pick(raw.columns, "名称", "股票简称")
        if not code_col:
            return pd.DataFrame(columns=["code", "name", "board"])
        return pd.DataFrame({
            "code": raw[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
            "name": raw[name_col].astype(str) if name_col else "",
            "board": resolved.board_name,
        }).dropna(subset=["code"]).drop_duplicates("code")

    def sector_universe(self, sector_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[BoardResolved]]:
        chunks: list[pd.DataFrame] = []
        resolved_all: list[BoardResolved] = []
        for spec in sector_cfg.get("boards", []):
            resolved = self.resolve_board(spec["kind"], spec.get("aliases", []))
            if not resolved:
                continue
            resolved_all.append(resolved)
            c = self.board_constituents(resolved)
            c["source_kind"] = resolved.kind
            chunks.append(c)
        if not chunks:
            return pd.DataFrame(columns=["code", "name"]), resolved_all
        allc = pd.concat(chunks, ignore_index=True)
        board_count = allc.groupby("code")["board"].nunique().rename("board_hits")
        names = allc.groupby("code")["name"].first()
        return pd.concat([names, board_count], axis=1).reset_index(), resolved_all

    def stock_history_indicator(self, code: str) -> pd.DataFrame:
        """Optional AKShare LeGu index. Failure is non-fatal."""
        key = f"indicator_{code}"
        try:
            return self._cached_frame(key, lambda: self.ak.stock_a_indicator_lg(symbol=code), timedelta(hours=18))
        except Exception:
            return pd.DataFrame()


class ValuationEngine:
    def __init__(self, provider: ValuationDataProvider, config: dict[str, Any], state_dir: Path):
        self.p = provider
        self.cfg = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.spot = self.p.spot()
        self.fund = self.p.fundamentals_ttm(
            config["report_date"], config["prior_report_date"], config["annual_report_date"]
        )
        self.master = self.spot.merge(self.fund, on="code", how="left")
        self.master["pe_ttm_calc"] = np.where(
            (self.master["ttm_profit"] > 0) & (self.master["market_cap"] > 0),
            self.master["market_cap"] / self.master["ttm_profit"], np.nan
        )
        self.master["ps_ttm_calc"] = np.where(
            (self.master["ttm_revenue"] > 0) & (self.master["market_cap"] > 0),
            self.master["market_cap"] / self.master["ttm_revenue"], np.nan
        )

    def _aggregate(self, universe: pd.DataFrame) -> dict[str, float]:
        x = universe[["code"]].drop_duplicates().merge(self.master, on="code", how="left")
        x = x[x["market_cap"].notna() & (x["market_cap"] > 0)].copy()
        total_mv = x["market_cap"].sum()
        pos = x[x["ttm_profit"] > 0]
        pe = pos["market_cap"].sum() / pos["ttm_profit"].sum() if len(pos) and pos["ttm_profit"].sum() > 0 else np.nan
        revenue_sum = x.loc[x["ttm_revenue"] > 0, "ttm_revenue"].sum()
        ps = total_mv / revenue_sum if revenue_sum > 0 else np.nan
        pb_x = x[(x["pb"] > 0) & x["pb"].notna()].copy()
        book = (pb_x["market_cap"] / pb_x["pb"]).sum()
        pb = pb_x["market_cap"].sum() / book if book > 0 else np.nan

        # Aggregate H1 growth from actual current/prior numbers, avoiding arithmetic averaging of percentages.
        both_rev = x[(x["revenue_cur"] > 0) & (x["revenue_pri"] > 0)]
        rev_growth = both_rev["revenue_cur"].sum() / both_rev["revenue_pri"].sum() - 1 if len(both_rev) else np.nan
        both_p = x[x["net_profit_pri"].notna() & x["net_profit_cur"].notna()]
        prior_profit = both_p["net_profit_pri"].sum()
        profit_growth = both_p["net_profit_cur"].sum() / prior_profit - 1 if prior_profit > 0 else np.nan
        roe = float(np.nanmedian(x["roe_h1_pct_cur"])) if x["roe_h1_pct_cur"].notna().any() else np.nan
        gross = float(np.nanmedian(x["gross_margin_pct_cur"])) if x["gross_margin_pct_cur"].notna().any() else np.nan
        profitable_coverage = pos["market_cap"].sum() / total_mv if total_mv > 0 else np.nan
        data_coverage = x["ttm_revenue"].notna().mean() if len(x) else 0
        return {
            "constituents": len(x), "market_cap": total_mv, "pe": pe, "pb": pb, "ps": ps,
            "revenue_growth": rev_growth, "profit_growth": profit_growth,
            "roe_h1_pct": roe, "gross_margin_pct": gross,
            "profitable_mcap_coverage": profitable_coverage, "data_coverage": data_coverage,
        }

    @staticmethod
    def _normalized_growth_pct(m: dict[str, float]) -> float:
        rg = m.get("revenue_growth", np.nan) * 100
        pg = m.get("profit_growth", np.nan) * 100
        if np.isfinite(pg) and np.isfinite(rg):
            # Profit is more important, but cap extreme one-offs such as base-effect reversals.
            return 0.65 * _clip(pg, -30, 80) + 0.35 * _clip(rg, -30, 60)
        if np.isfinite(pg):
            return _clip(pg, -30, 80)
        return _clip(rg, -30, 60) if np.isfinite(rg) else np.nan

    def _history_fair(self, entity: str, metric: str, bootstrap: float | None) -> tuple[float, str]:
        p = self.state_dir / "valuation_snapshots.csv"
        min_points = int(self.cfg.get("history_min_points", 30))
        if p.exists():
            h = pd.read_csv(p)
            q = h[(h["entity"] == entity) & h[metric].notna()]
            if len(q) >= min_points:
                return float(q[metric].tail(252).median()), f"自建历史中位数({len(q.tail(252))}点)"
        if bootstrap is not None and np.isfinite(bootstrap):
            return float(bootstrap), "bootstrap历史中位数"
        return np.nan, "无历史基准"

    def _model_fair(self, name: str, c: dict[str, Any], m: dict[str, float]) -> tuple[dict[str, float], str]:
        model = c["model"]
        g = self._normalized_growth_pct(m)
        fair: dict[str, float] = {"pe": np.nan, "pb": np.nan, "ps": np.nan}
        note = ""

        if model in {"growth_pe", "cyclical_growth"}:
            gf = float(c.get("growth_floor_pct", 10))
            gc = float(c.get("growth_cap_pct", 40))
            g_used = _clip(g, gf, gc) if np.isfinite(g) else gf
            fair["pe"] = _clip(float(c["target_peg"]) * g_used, float(c["fair_pe_floor"]), float(c["fair_pe_cap"]))
            if model == "cyclical_growth":
                # Cyclicals should not capitalize peak growth as aggressively.
                fair["pe"] *= 0.90
            note = f"PEG/正常化增长模型(g={g_used:.1f}%)"

        elif model == "utility":
            roe_annual = m.get("roe_h1_pct", np.nan) * 2 / 100
            k = float(c.get("required_return", 0.09))
            tg = float(c.get("terminal_growth", 0.03))
            if np.isfinite(roe_annual) and roe_annual > tg and k > tg:
                fair_pb = (roe_annual - tg) / (k - tg)
                fair["pb"] = _clip(fair_pb, float(c["fair_pb_floor"]), float(c["fair_pb_cap"]))
                fair["pe"] = _clip(fair["pb"] / roe_annual, float(c["fair_pe_floor"]), float(c["fair_pe_cap"]))
            else:
                fair["pe"] = (float(c["fair_pe_floor"]) + float(c["fair_pe_cap"])) / 2
                fair["pb"] = (float(c["fair_pb_floor"]) + float(c["fair_pb_cap"])) / 2
            note = "PB-ROE/Gordon + PE约束"

        elif model in {"innovation_drug", "early_growth_ps"}:
            rg = m.get("revenue_growth", np.nan) * 100
            rg = _clip(rg, -10, 50) if np.isfinite(rg) else 0
            fair_ps = float(c.get("base_ps", 3.0)) + max(rg, 0) * float(c.get("revenue_growth_ps_slope", 0.06))
            if model == "innovation_drug" and m.get("profit_growth", -1) > 0 and m.get("profitable_mcap_coverage", 0) > 0.6:
                fair_ps += float(c.get("profitable_ps_bonus", 0.8))
            fair["ps"] = _clip(fair_ps, float(c["fair_ps_floor"]), float(c["fair_ps_cap"]))
            note = "PS/收入增长模型" if model == "early_growth_ps" else "创新药PS+盈利兑现模型"
        return fair, note

    def evaluate_sector(self, name: str, c: dict[str, Any]) -> dict[str, Any]:
        u, resolved = self.p.sector_universe(c)
        if u.empty:
            return {"entity": name, "type": "sector", "error": "未解析到板块成分"}
        m = self._aggregate(u)
        fair_model, model_note = self._model_fair(name, c, m)

        primary = c.get("bootstrap_metric", "pe")
        bootstrap = c.get("bootstrap_fair")
        hist_fair, hist_source = self._history_fair(name, primary, bootstrap)
        model_primary = fair_model.get(primary, np.nan)
        if np.isfinite(model_primary) and np.isfinite(hist_fair):
            fair_primary = 0.65 * model_primary + 0.35 * hist_fair
            fair_source = f"65%模型+35%{hist_source}"
        elif np.isfinite(model_primary):
            fair_primary, fair_source = model_primary, model_note
        else:
            fair_primary, fair_source = hist_fair, hist_source

        # Ratio from primary metric plus secondary PB where relevant.
        current_primary = m.get(primary, np.nan)
        ratios: list[tuple[float, float]] = []
        if np.isfinite(current_primary) and np.isfinite(fair_primary) and fair_primary > 0:
            ratios.append((current_primary / fair_primary, 0.8))
        if np.isfinite(m.get("pb", np.nan)) and np.isfinite(fair_model.get("pb", np.nan)) and fair_model["pb"] > 0:
            ratios.append((m["pb"] / fair_model["pb"], 0.2))
        value_ratio = _geomean_ratio(ratios)
        deviation = value_ratio - 1 if np.isfinite(value_ratio) else np.nan

        resolved_names = ";".join(f"{r.kind}:{r.board_name}" for r in resolved)
        return {
            "entity": name, "type": "sector", "model": c["model"], "resolved_boards": resolved_names,
            **m, "primary_metric": primary, "current_primary": current_primary,
            "fair_primary": fair_primary, "fair_source": fair_source, "value_deviation": deviation,
            "valuation_label": valuation_label(deviation), "model_note": model_note,
        }

    def _stock_history_median(self, code: str, metric: str) -> float:
        h = self.p.stock_history_indicator(code)
        if h.empty:
            return np.nan
        candidates = {
            "pe": ["pe_ttm", "pe"], "pb": ["pb"], "ps": ["ps_ttm", "ps"]
        }.get(metric, [metric])
        col = next((c for c in candidates if c in h.columns), None)
        if not col:
            return np.nan
        s = pd.to_numeric(h[col], errors="coerce")
        s = s[(s > 0) & np.isfinite(s)]
        return float(s.tail(750).median()) if len(s) >= 30 else np.nan

    def _blend_stock_fair(self, code: str, metric: str, model_value: float) -> tuple[float, str]:
        hist = self._stock_history_median(code, metric)
        if np.isfinite(model_value) and model_value > 0 and np.isfinite(hist) and hist > 0:
            return 0.60 * model_value + 0.40 * hist, f"60%基本面模型+40%个股近3年{metric.upper()}历史中位数"
        if np.isfinite(model_value) and model_value > 0:
            return model_value, f"{metric.upper()}基本面模型"
        if np.isfinite(hist) and hist > 0:
            return hist, f"个股近3年{metric.upper()}历史中位数"
        return np.nan, "无有效基准"

    def evaluate_stock(self, code: str, info: dict[str, Any]) -> dict[str, Any]:
        x = self.master[self.master["code"] == code]
        if x.empty:
            return {"entity": info.get("name", code), "code": code, "type": "stock", "error": "股票不存在或无行情"}
        r = x.iloc[0]
        sector = info.get("sector")
        c = self.cfg["sectors"].get(sector)
        # Medical-device / unconfigured fallback: mature growth-quality model.
        if c is None:
            c = {
                "model": "growth_pe", "target_peg": 1.45, "growth_floor_pct": 8, "growth_cap_pct": 25,
                "fair_pe_floor": 20, "fair_pe_cap": 42, "bootstrap_metric": "pe", "bootstrap_fair": np.nan
            }
        pe_value = float(r["pe_ttm_calc"]) if np.isfinite(r["pe_ttm_calc"]) else float(r["pe_dynamic"])
        m = {
            "pe": pe_value,
            "pb": float(r["pb"]), "ps": float(r["ps_ttm_calc"]),
            "revenue_growth": float(r["revenue_yoy_cur"]) / 100 if np.isfinite(r["revenue_yoy_cur"]) else np.nan,
            "profit_growth": float(r["profit_yoy_cur"]) / 100 if np.isfinite(r["profit_yoy_cur"]) else np.nan,
            "roe_h1_pct": float(r["roe_h1_pct_cur"]) if np.isfinite(r["roe_h1_pct_cur"]) else np.nan,
            "profitable_mcap_coverage": 1.0 if r["ttm_profit"] > 0 else 0.0,
        }
        fair_model, model_note = self._model_fair(info.get("name", code), c, m)
        configured_primary = c.get("bootstrap_metric", "pe")

        # Utility stocks require two valuation lenses.  PE is invalid when TTM
        # earnings <= 0, so automatically fall back to PB rather than treating
        # a negative PE as "cheap".
        ratios: list[tuple[float, float]] = []
        fair_by_metric: dict[str, float] = {}
        source_by_metric: dict[str, str] = {}
        if c.get("model") == "utility":
            for metric, weight in (("pe", 0.65), ("pb", 0.35)):
                current_metric = m.get(metric, np.nan)
                model_metric = fair_model.get(metric, np.nan)
                fair_metric, fair_src = self._blend_stock_fair(code, metric, model_metric)
                fair_by_metric[metric] = fair_metric
                source_by_metric[metric] = fair_src
                if np.isfinite(current_metric) and current_metric > 0 and np.isfinite(fair_metric) and fair_metric > 0:
                    ratios.append((current_metric / fair_metric, weight))
            primary = "pe" if np.isfinite(m["pe"]) and m["pe"] > 0 else "pb"
            current = m[primary]
            fair = fair_by_metric.get(primary, np.nan)
            fair_source = source_by_metric.get(primary, "无有效基准")
            value_ratio = _geomean_ratio(ratios)
            deviation = value_ratio - 1 if np.isfinite(value_ratio) else np.nan
        else:
            primary = configured_primary
            current = m.get(primary, np.nan)
            model_value = fair_model.get(primary, np.nan)
            fair, fair_source = self._blend_stock_fair(code, primary, model_value)
            deviation = current / fair - 1 if np.isfinite(current) and current > 0 and np.isfinite(fair) and fair > 0 else np.nan

        # Growth-quality gate: a historically cheap multiple is not enough if
        # current earnings are deteriorating.
        gate = "通过"
        pg = m["profit_growth"]
        rg = m["revenue_growth"]
        if (np.isfinite(pg) and pg < -0.10) or (np.isfinite(rg) and rg < -0.10):
            gate = "不通过-业绩下滑"
        elif np.isfinite(pg) and pg < 0:
            gate = "观察-利润未增长"
        elif np.isfinite(pg) and pg < 0.08:
            gate = "观察-利润增长偏弱"

        return {
            "entity": info.get("name", str(r["name"])), "code": code, "type": "stock", "sector": sector,
            "price": r["price"], "market_cap": r["market_cap"], "pe": m["pe"], "pb": m["pb"], "ps": m["ps"],
            "revenue_growth": m["revenue_growth"], "profit_growth": m["profit_growth"], "roe_h1_pct": m["roe_h1_pct"],
            "primary_metric": primary, "current_primary": current, "fair_primary": fair,
            "fair_source": fair_source, "value_deviation": deviation, "valuation_label": valuation_label(deviation),
            "growth_gate": gate, "model_note": model_note,
            "fair_pe": fair_by_metric.get("pe", fair if primary == "pe" else np.nan),
            "fair_pb": fair_by_metric.get("pb", fair if primary == "pb" else np.nan),
        }

    def save_snapshots(self, sector_rows: list[dict[str, Any]]) -> Path:
        p = self.state_dir / "valuation_snapshots.csv"
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for r in sector_rows:
            if r.get("error"):
                continue
            rows.append({"timestamp": now, "entity": r["entity"], "pe": r.get("pe"), "pb": r.get("pb"), "ps": r.get("ps")})
        new = pd.DataFrame(rows)
        if p.exists():
            old = pd.read_csv(p)
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(p, index=False, encoding="utf-8-sig")
        return p


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
