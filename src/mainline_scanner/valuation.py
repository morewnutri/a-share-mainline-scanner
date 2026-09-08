from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .valuation_policy import (
    HistoryStats,
    TTM_CONFIDENCE,
    confidence_label,
    data_quality_score,
    fair_value_interval,
    position_policy,
    prepare_history,
    required_margin,
    robust_z,
    shrink_log_value,
    summarize_values,
    trade_bands,
    ttm_method,
    weighted_geometric_price,
)

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


@dataclass(frozen=True)
class ReportSelection:
    current: str
    prior: str
    annual: str
    current_coverage: float
    prior_coverage: float
    annual_coverage: float
    source: str


def annualization_factor(report_date: str) -> float:
    """Convert cumulative interim figures/ROE to an approximate annual rate."""
    suffix = str(report_date)[-4:]
    return {"0331": 4.0, "0630": 2.0, "0930": 4.0 / 3.0, "1231": 1.0}.get(suffix, 1.0)


def report_period_triplet(current: str) -> tuple[str, str, str]:
    current = str(current)
    if not re.fullmatch(r"\d{8}", current):
        raise ValueError(f"非法财报日期: {current}")
    year = int(current[:4])
    suffix = current[4:]
    if suffix not in {"0331", "0630", "0930", "1231"}:
        raise ValueError(f"不支持的财报期: {current}")
    return current, f"{year - 1}{suffix}", f"{year - 1}1231"


def completed_report_candidates(today: date | None = None, limit: int = 8) -> list[str]:
    """Newest broadly-complete A-share interim periods, newest first.

    Q1 is treated as complete from Apr-30, H1 from Aug-31, Q3 from Oct-31.
    Annual reports are used as the TTM base, not as the cross-sectional current
    period, because Q1 becomes complete at the same statutory cutoff.
    """
    if today is None:
        try:
            today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        except Exception:  # pragma: no cover
            today = date.today()
    rows: list[tuple[date, str]] = []
    for year in range(today.year - 4, today.year + 1):
        rows.extend([
            (date(year, 4, 30), f"{year}0331"),
            (date(year, 8, 31), f"{year}0630"),
            (date(year, 10, 31), f"{year}0930"),
        ])
    return [period for cutoff, period in sorted(rows, reverse=True) if cutoff <= today][:limit]


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
        # --refresh means "fetch once from source in this process", not "refetch
        # every time the same logical dataset is requested". This avoids duplicate
        # network calls during report-date validation, valuation and sector inference.
        self._memory_frames: dict[str, pd.DataFrame] = {}
        self._fetch_audit: dict[str, dict[str, Any]] = {}
        self._sector_universe_memory: dict[str, tuple[pd.DataFrame, list[BoardResolved]]] = {}

    def _cached_frame(self, key: str, loader, ttl: timedelta | None = None) -> pd.DataFrame:
        if key in self._memory_frames:
            return self._memory_frames[key].copy()

        p = self.cache_dir / f"{key}.csv"
        t = ttl or self.ttl
        cache_used = False
        fetched_at = datetime.now()
        if not self.refresh and p.exists() and datetime.now() - datetime.fromtimestamp(p.stat().st_mtime) <= t:
            df = pd.read_csv(p, dtype={"代码": str, "股票代码": str}, encoding="utf-8-sig")
            cache_used = True
            fetched_at = datetime.fromtimestamp(p.stat().st_mtime)
        else:
            df = loader()
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError(f"{key} 数据源未返回 DataFrame")
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False, encoding="utf-8-sig")
            fetched_at = datetime.now()

        self._memory_frames[key] = df.copy()
        self._fetch_audit[key] = {
            "dataset": key,
            "rows": len(df),
            "cache_used": cache_used,
            "refresh_requested": self.refresh,
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "cache_path": str(p),
            "ttl_minutes": round(t.total_seconds() / 60, 1),
        }
        return df.copy()

    def freshness_frame(self) -> pd.DataFrame:
        if not self._fetch_audit:
            return pd.DataFrame(columns=["dataset", "rows", "cache_used", "refresh_requested", "fetched_at", "cache_path", "ttl_minutes"])
        return pd.DataFrame(self._fetch_audit.values()).sort_values("dataset").reset_index(drop=True)

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
        rev_col = _pick(raw.columns, "营业总收入-营业总收入", "营业总收入", "营业收入-营业收入", "营业收入")
        rev_yoy_col = _pick(raw.columns, "营业总收入-同比增长", "营业总收入同比增长", "营业收入-同比增长", "营业收入同比增长")
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
        # Every TTM-like value carries provenance. Annualised interim values are
        # retained for cautious fallback models but can never masquerade as exact TTM.
        factor = annualization_factor(current)
        for target, stem in (("ttm_revenue", "revenue"), ("ttm_profit", "net_profit"), ("ttm_eps", "eps"), ("ttm_ocfps", "ocfps")):
            exact = x[f"{stem}_ann"].notna() & x[f"{stem}_cur"].notna() & x[f"{stem}_pri"].notna()
            exact_value = x[f"{stem}_ann"] + x[f"{stem}_cur"] - x[f"{stem}_pri"]
            estimated = x[f"{stem}_cur"] * factor
            x[target] = exact_value.where(exact, estimated.where(x[f"{stem}_cur"].notna(), np.nan))
            x[f"{target}_method"] = [ttm_method(current, bool(a), bool(b)) for a, b in zip(exact, x[f"{stem}_cur"].notna())]
            x[f"{target}_confidence"] = x[f"{target}_method"].map(TTM_CONFIDENCE).fillna(0.0)
        return x

    def board_catalog(self, kind: str) -> pd.DataFrame:
        def load() -> pd.DataFrame:
            raw = self.base._direct_universe(kind).copy()
            return raw[["板块名称", "板块代码"]].drop_duplicates()
        return self._cached_frame(f"board_catalog_{kind}", load, timedelta(minutes=30)).copy()

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

    def sector_universe(self, sector_cfg: dict[str, Any], purpose: str = "inference") -> tuple[pd.DataFrame, list[BoardResolved]]:
        board_key = "valuation_boards" if purpose == "valuation" and sector_cfg.get("valuation_boards") else "boards"
        board_specs = sector_cfg.get(board_key, [])
        cache_key = purpose + ":" + json.dumps(board_specs, ensure_ascii=False, sort_keys=True)
        if cache_key in self._sector_universe_memory:
            u, r = self._sector_universe_memory[cache_key]
            return u.copy(), list(r)

        chunks: list[pd.DataFrame] = []
        resolved_all: list[BoardResolved] = []
        for spec in board_specs:
            resolved = self.resolve_board(spec["kind"], spec.get("aliases", []))
            if not resolved:
                continue
            resolved_all.append(resolved)
            c = self.board_constituents(resolved)
            c["source_kind"] = resolved.kind
            chunks.append(c)
        if not chunks:
            result = pd.DataFrame(columns=["code", "name", "board_hits"])
            self._sector_universe_memory[cache_key] = (result.copy(), resolved_all)
            return result, resolved_all
        allc = pd.concat(chunks, ignore_index=True)
        board_count = allc.groupby("code")["board"].nunique().rename("board_hits")
        names = allc.groupby("code")["name"].first()
        result = pd.concat([names, board_count], axis=1).reset_index()
        self._sector_universe_memory[cache_key] = (result.copy(), resolved_all)
        return result, resolved_all

    def infer_stock_sector(self, code: str, config: dict[str, Any]) -> str:
        """Infer the best configured valuation sector from board membership."""
        priority = list(config.get("sector_priority", config.get("sectors", {}).keys()))
        priority_rank = {name: i for i, name in enumerate(priority)}
        candidates: list[tuple[int, int, str]] = []
        for sector_name, sector_cfg in config.get("sectors", {}).items():
            try:
                universe, _ = self.sector_universe(sector_cfg, purpose="inference")
            except Exception:
                continue
            hit = universe[universe["code"] == str(code)]
            if hit.empty:
                continue
            raw_hits = pd.to_numeric(hit.iloc[0].get("board_hits", 1), errors="coerce")
            board_hits = int(raw_hits) if pd.notna(raw_hits) and float(raw_hits) > 0 else 1
            candidates.append((-board_hits, priority_rank.get(sector_name, 999), sector_name))
        if not candidates:
            return "MODEL_UNRESOLVED"
        candidates.sort()
        return candidates[0][2]

    def stock_history_indicator(self, code: str) -> pd.DataFrame:
        """Optional AKShare/LeGu history with visible status and sorted trade dates."""
        key = f"indicator_{code}"
        try:
            frame = self._cached_frame(key, lambda: self.ak.stock_a_indicator_lg(symbol=code), timedelta(hours=18))
            frame = prepare_history(frame)
            frame.attrs.update(history_status="OK" if len(frame) else "EMPTY", history_source="AKShare/LeGu", history_error="")
            return frame
        except Exception as exc:
            frame = pd.DataFrame()
            frame.attrs.update(history_status="FAILED", history_source="AKShare/LeGu", history_error=f"{type(exc).__name__}: {exc}")
            return frame


class _LegacyValuationEngine:
    def __init__(self, provider: ValuationDataProvider, config: dict[str, Any], state_dir: Path):
        self.p = provider
        self.cfg = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.spot = self.p.spot()
        self.report_date = str(config["report_date"])
        self.roe_annualization_factor = annualization_factor(self.report_date)
        self.fund = self.p.fundamentals_ttm(
            self.report_date, config["prior_report_date"], config["annual_report_date"]
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

        # Aggregate current-period growth from actual current/prior numbers, avoiding arithmetic averaging of percentages.
        both_rev = x[(x["revenue_cur"] > 0) & (x["revenue_pri"] > 0)]
        rev_growth = both_rev["revenue_cur"].sum() / both_rev["revenue_pri"].sum() - 1 if len(both_rev) else np.nan
        both_p = x[x["net_profit_pri"].notna() & x["net_profit_cur"].notna()]
        prior_profit = both_p["net_profit_pri"].sum()
        profit_growth = both_p["net_profit_cur"].sum() / prior_profit - 1 if prior_profit > 0 else np.nan
        roe = float(np.nanmedian(x["roe_h1_pct_cur"])) if x["roe_h1_pct_cur"].notna().any() else np.nan
        gross = float(np.nanmedian(x["gross_margin_pct_cur"])) if x["gross_margin_pct_cur"].notna().any() else np.nan
        profitable_coverage = pos["market_cap"].sum() / total_mv if total_mv > 0 else np.nan
        ttm_revenue_coverage = x["ttm_revenue"].notna().mean() if len(x) else 0.0
        ttm_profit_coverage = x["ttm_profit"].notna().mean() if len(x) else 0.0
        revenue_growth_coverage = len(both_rev) / len(x) if len(x) else 0.0
        profit_growth_coverage = len(both_p) / len(x) if len(x) else 0.0
        data_coverage = min(ttm_revenue_coverage, ttm_profit_coverage, revenue_growth_coverage, profit_growth_coverage)
        return {
            "constituents": len(x), "market_cap": total_mv, "pe": pe, "pb": pb, "ps": ps,
            "revenue_growth": rev_growth, "profit_growth": profit_growth,
            "roe_h1_pct": roe, "gross_margin_pct": gross,
            "profitable_mcap_coverage": profitable_coverage, "data_coverage": data_coverage,
            "ttm_revenue_coverage": ttm_revenue_coverage, "ttm_profit_coverage": ttm_profit_coverage,
            "revenue_growth_coverage": revenue_growth_coverage, "profit_growth_coverage": profit_growth_coverage,
        }

    @staticmethod
    def _normalized_growth_pct(m: dict[str, float], c: dict[str, Any] | None = None) -> float:
        rg = m.get("revenue_growth", np.nan) * 100
        pg = m.get("profit_growth", np.nan) * 100
        c = c or {}
        if np.isfinite(pg) and np.isfinite(rg):
            # Current-period profit can explode because of a low base, loss-to-profit reversals,
            # memory/commodity cycles or one-off gains. Do not capitalize that full jump into
            # a perpetual PEG multiple. Limit sustainable profit growth to a configurable
            # premium over revenue growth before weighting the two.
            premium_cap = c.get("profit_growth_premium_cap_pct")
            if premium_cap is not None and np.isfinite(float(premium_cap)):
                pg = min(pg, rg + float(premium_cap))
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
        g = self._normalized_growth_pct(m, c)
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
            roe_annual = m.get("roe_h1_pct", np.nan) * self.roe_annualization_factor / 100
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
        u, resolved = self.p.sector_universe(c, purpose="valuation")
        if u.empty:
            return {"entity": name, "type": "sector", "error": "未解析到板块成分"}
        m = self._aggregate(u)
        primary = c.get("bootstrap_metric", "pe")
        min_cov = float(c.get("min_data_coverage", self.cfg.get("report_policy", {}).get("min_sector_data_coverage", 0.75)))
        growth_required = c.get("model") in {"growth_pe", "cyclical_growth", "innovation_drug", "early_growth_ps"}
        missing_growth = growth_required and (not np.isfinite(m.get("revenue_growth", np.nan)) or not np.isfinite(m.get("profit_growth", np.nan)))
        if m.get("data_coverage", 0.0) < min_cov or missing_growth or not np.isfinite(m.get(primary, np.nan)):
            resolved_names = ";".join(f"{r.kind}:{r.board_name}" for r in resolved)
            return {
                "entity": name, "type": "sector", "model": c["model"], "resolved_boards": resolved_names,
                **m, "primary_metric": primary, "current_primary": m.get(primary, np.nan),
                "fair_primary": np.nan, "fair_source": "数据覆盖不足，拒绝估值", "value_deviation": np.nan,
                "valuation_label": "数据不足", "model_note": f"coverage={m.get('data_coverage', 0.0):.1%}, min={min_cov:.1%}",
            }

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
        if c is None:
            c = dict(self.cfg.get("fallback_stock_model", {
                "model": "growth_pe", "target_peg": 1.45, "growth_floor_pct": 8, "growth_cap_pct": 25,
                "fair_pe_floor": 20, "fair_pe_cap": 42, "bootstrap_metric": "pe", "bootstrap_fair": np.nan
            }))
        if np.isfinite(r["pe_ttm_calc"]) and float(r["pe_ttm_calc"]) > 0:
            pe_value = float(r["pe_ttm_calc"])
        elif pd.isna(r["ttm_profit"]) and np.isfinite(r["pe_dynamic"]) and float(r["pe_dynamic"]) > 0:
            # Only use the quote-provider PE when TTM profit itself is missing.
            # A known loss must never be converted into a seemingly cheap PE.
            pe_value = float(r["pe_dynamic"])
        else:
            pe_value = np.nan
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
            "growth_gate": gate, "model_note": model_note, "stock_source": info.get("source", "config"),
            "fair_pe": fair_by_metric.get("pe", fair if primary == "pe" else np.nan),
            "fair_pb": fair_by_metric.get("pb", fair if primary == "pb" else np.nan),
        }

    def save_snapshots(self, sector_rows: list[dict[str, Any]]) -> Path:
        """Persist one valuation observation per sector per Shanghai calendar day.

        Re-running a notebook many times on the same day must not manufacture
        enough "history" to satisfy history_min_points. The latest run of the
        day replaces the previous same-day observation.
        """
        p = self.state_dir / "valuation_snapshots.csv"
        try:
            now_dt = datetime.now(ZoneInfo("Asia/Shanghai"))
        except Exception:  # pragma: no cover
            now_dt = datetime.now().astimezone()
        now = now_dt.isoformat(timespec="seconds")
        day = now_dt.date().isoformat()
        rows = []
        for r in sector_rows:
            if r.get("error"):
                continue
            rows.append({
                "timestamp": now, "date": day, "entity": r["entity"],
                "pe": r.get("pe"), "pb": r.get("pb"), "ps": r.get("ps")
            })
        new = pd.DataFrame(rows)
        if p.exists():
            old = pd.read_csv(p)
            if "date" not in old.columns and "timestamp" in old.columns:
                old["date"] = pd.to_datetime(old["timestamp"], errors="coerce").dt.date.astype("string")
            if len(new):
                entities = set(new["entity"].astype(str))
                old = old[~((old["date"].astype(str) == day) & old["entity"].astype(str).isin(entities))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(p, index=False, encoding="utf-8-sig")
        return p


class ValuationEngine(_LegacyValuationEngine):
    """Risk-aware valuation pipeline with provenance, uncertainty and position policy."""

    def __init__(self, provider: ValuationDataProvider, config: dict[str, Any], state_dir: Path):
        self.p = provider
        self.cfg = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.spot = self.p.spot()
        self.report_date = str(config["report_date"])
        self.roe_annualization_factor = annualization_factor(self.report_date)
        self.fund = self.p.fundamentals_ttm(
            self.report_date, config["prior_report_date"], config["annual_report_date"]
        )
        self.master = self.spot.merge(self.fund, on="code", how="left")
        self.master["pe_ttm_calc"] = np.where(
            (self.master["ttm_profit"] > 0) & (self.master["market_cap"] > 0),
            self.master["market_cap"] / self.master["ttm_profit"], np.nan,
        )
        self.master["ps_ttm_calc"] = np.where(
            (self.master["ttm_revenue"] > 0) & (self.master["market_cap"] > 0),
            self.master["market_cap"] / self.master["ttm_revenue"], np.nan,
        )
        book = np.where(
            (self.master["pb"] > 0) & (self.master["market_cap"] > 0),
            self.master["market_cap"] / self.master["pb"], np.nan,
        )
        self.master["ttm_roe_pct"] = np.where(
            (np.asarray(book) > 0) & self.master["ttm_profit"].notna(),
            self.master["ttm_profit"] / book * 100.0, np.nan,
        )
        self.master["ttm_roe_method"] = np.where(
            self.master["ttm_roe_pct"].notna(), "TTM_NI_OVER_CURRENT_EQUITY", "ANNUALIZED_REPORTED_ROE"
        )
        fallback_roe = self.master["roe_h1_pct_cur"] * self.roe_annualization_factor
        self.master["ttm_roe_pct"] = self.master["ttm_roe_pct"].where(self.master["ttm_roe_pct"].notna(), fallback_roe)
        self.master["ttm_roe_confidence"] = np.where(
            self.master["ttm_roe_method"] == "TTM_NI_OVER_CURRENT_EQUITY",
            np.minimum(self.master["ttm_profit_confidence"].fillna(0.0), 0.75),
            np.minimum(self.master["ttm_profit_confidence"].fillna(0.0), 0.45),
        )
        shares = self.master["ttm_profit"] / self.master["ttm_eps"]
        shares = shares.where((shares > 0) & np.isfinite(shares), np.nan)
        self.master["shares_proxy"] = shares
        self.master["sps_ttm"] = self.master["ttm_revenue"] / shares
        self.master["bvps"] = (pd.Series(book, index=self.master.index) / shares).where(shares > 0)
        self.master["cash_conversion"] = np.where(
            (self.master["ttm_eps"] > 0) & self.master["ttm_ocfps"].notna(),
            self.master["ttm_ocfps"] / self.master["ttm_eps"], np.nan,
        )
        self.master["gross_margin_trend_pct"] = self.master["gross_margin_pct_cur"] - self.master["gross_margin_pct_pri"]
        self.master["roe_trend_pct"] = self.master["roe_h1_pct_cur"] - self.master["roe_h1_pct_pri"]
        self._history_cache: dict[tuple[str, str], HistoryStats] = {}

    def _aggregate(self, universe: pd.DataFrame) -> dict[str, float | str]:
        x = universe[["code"]].drop_duplicates().merge(self.master, on="code", how="left")
        x = x[x["market_cap"].notna() & (x["market_cap"] > 0)].copy()
        total_mv = float(x["market_cap"].sum())
        if total_mv <= 0:
            return {"constituents": 0, "data_coverage": 0.0}
        pos = x[x["ttm_profit"] > 0]
        profit_valid = x[x["ttm_profit"].notna()]
        total_profit = float(profit_valid["ttm_profit"].sum())
        positive_profit_pe = (
            float(pos["market_cap"].sum() / pos["ttm_profit"].sum())
            if len(pos) and pos["ttm_profit"].sum() > 0 else np.nan
        )
        aggregate_pe = total_mv / total_profit if total_profit > 0 else np.nan
        profitable_coverage = float(pos["market_cap"].sum() / total_mv)
        loss_mcap_share = float(x.loc[x["ttm_profit"] <= 0, "market_cap"].sum() / total_mv)

        ps_x = x[(x["ttm_revenue"] > 0) & x["ttm_revenue"].notna()]
        ps = float(ps_x["market_cap"].sum() / ps_x["ttm_revenue"].sum()) if len(ps_x) else np.nan
        total_revenue = float(ps_x["ttm_revenue"].sum()) if len(ps_x) else np.nan
        ps_cov = float(ps_x["market_cap"].sum() / total_mv)
        pb_x = x[(x["pb"] > 0) & x["pb"].notna()].copy()
        book = (pb_x["market_cap"] / pb_x["pb"]).sum()
        pb = float(pb_x["market_cap"].sum() / book) if book > 0 else np.nan
        pb_cov = float(pb_x["market_cap"].sum() / total_mv)

        both_rev = x[(x["revenue_cur"] > 0) & (x["revenue_pri"] > 0)]
        rev_growth = both_rev["revenue_cur"].sum() / both_rev["revenue_pri"].sum() - 1 if len(both_rev) else np.nan
        both_p = x[x["net_profit_pri"].notna() & x["net_profit_cur"].notna()]
        prior_profit = both_p["net_profit_pri"].sum()
        profit_growth = both_p["net_profit_cur"].sum() / prior_profit - 1 if prior_profit > 0 else np.nan
        ttm_profit = x["ttm_profit"].sum(min_count=1)
        total_book = (pb_x["market_cap"] / pb_x["pb"]).sum(min_count=1)
        roe_ttm = float(ttm_profit / total_book * 100) if total_book > 0 and np.isfinite(ttm_profit) else np.nan
        gross = float(np.nanmedian(x["gross_margin_pct_cur"])) if x["gross_margin_pct_cur"].notna().any() else np.nan
        ttm_revenue_cov = float(x.loc[x["ttm_revenue"].notna(), "market_cap"].sum() / total_mv)
        ttm_profit_cov = float(x.loc[x["ttm_profit"].notna(), "market_cap"].sum() / total_mv)
        rev_growth_cov = float(both_rev["market_cap"].sum() / total_mv)
        profit_growth_cov = float(both_p["market_cap"].sum() / total_mv)
        data_cov = min(ttm_revenue_cov, ttm_profit_cov, rev_growth_cov, profit_growth_cov)
        min_pe_cov = float(self.cfg.get("valuation_policy", {}).get("min_profitable_mcap_coverage", 0.70))
        pe = aggregate_pe if profitable_coverage >= min_pe_cov else np.nan
        return {
            "constituents": len(x), "market_cap": total_mv,
            "pe": pe, "aggregate_pe": aggregate_pe, "positive_profit_pe": positive_profit_pe,
            "pb": pb, "ps": ps, "revenue_growth": rev_growth, "profit_growth": profit_growth,
            "net_margin": total_profit / total_revenue if total_revenue > 0 else np.nan,
            "roe_ttm_pct": roe_ttm, "roe_h1_pct": roe_ttm, "gross_margin_pct": gross,
            "profitable_mcap_coverage": profitable_coverage, "loss_mcap_share": loss_mcap_share,
            "pe_mcap_coverage": profitable_coverage, "pb_mcap_coverage": pb_cov, "ps_mcap_coverage": ps_cov,
            "pe_metric_source": "aggregate earnings incl. losses" if np.isfinite(pe) else "DISABLED_LOW_PROFITABLE_COVERAGE",
            "pb_metric_source": "market-cap weighted aggregate book value",
            "ps_metric_source": "market-cap weighted positive-revenue constituents",
            "pe_metric_confidence": profitable_coverage if np.isfinite(pe) else 0.0,
            "pb_metric_confidence": pb_cov, "ps_metric_confidence": ps_cov,
            "data_coverage": data_cov, "ttm_revenue_coverage": ttm_revenue_cov,
            "ttm_profit_coverage": ttm_profit_cov, "revenue_growth_coverage": rev_growth_cov,
            "profit_growth_coverage": profit_growth_cov,
        }

    def _model_fair(self, name: str, c: dict[str, Any], m: dict[str, float]) -> tuple[dict[str, float], str]:
        model = c["model"]
        g = self._normalized_growth_pct(m, c)
        fair = {"pe": np.nan, "pb": np.nan, "ps": np.nan}
        note = ""
        if model in {"growth_pe", "cyclical_growth", "quality_growth", "mature_consumer"}:
            gf, gc = float(c.get("growth_floor_pct", 5)), float(c.get("growth_cap_pct", 40))
            g_used = _clip(g, gf, gc) if np.isfinite(g) else gf
            fair_pe = _clip(float(c.get("target_peg", 1.3)) * g_used, float(c["fair_pe_floor"]), float(c["fair_pe_cap"]))
            if model == "cyclical_growth":
                fair_pe *= 0.90
            cash = m.get("cash_conversion", np.nan)
            if model in {"quality_growth", "mature_consumer"} and np.isfinite(cash):
                fair_pe *= _clip(0.85 + 0.15 * cash, 0.80, 1.10)
            fair["pe"] = float(fair_pe)
            margin = m.get("net_margin", np.nan)
            roe = m.get("roe_ttm_pct", np.nan) / 100.0
            if np.isfinite(margin) and margin > 0:
                fair["ps"] = fair_pe * margin
            if np.isfinite(roe) and roe > 0:
                fair["pb"] = fair_pe * roe
            note = f"{model}: 正常化增长+现金质量(g={g_used:.1f}%)"
        elif model == "utility":
            roe = m.get("roe_ttm_pct", np.nan) / 100.0
            k, tg = float(c.get("required_return", 0.09)), float(c.get("terminal_growth", 0.03))
            if np.isfinite(roe) and roe > tg and k > tg:
                fair["pb"] = _clip((roe - tg) / (k - tg), float(c["fair_pb_floor"]), float(c["fair_pb_cap"]))
                fair["pe"] = _clip(fair["pb"] / roe, float(c["fair_pe_floor"]), float(c["fair_pe_cap"]))
            note = "PB-ROE/Gordon（TTM利润/权益代理）+ PE约束"
        elif model in {"innovation_drug", "early_growth_ps"}:
            rg = m.get("revenue_growth", np.nan) * 100
            rg = _clip(rg, -10, 50) if np.isfinite(rg) else 0
            value = float(c.get("base_ps", 3.0)) + max(rg, 0) * float(c.get("revenue_growth_ps_slope", 0.06))
            if model == "innovation_drug" and m.get("profit_growth", -1) > 0 and m.get("profitable_mcap_coverage", 0) > 0.6:
                value += float(c.get("profitable_ps_bonus", 0.8))
            fair["ps"] = _clip(value, float(c["fair_ps_floor"]), float(c["fair_ps_cap"]))
            note = "PS/收入增长+盈利兑现模型"
        return fair, note

    def _sector_history(self, entity: str, metric: str) -> HistoryStats:
        path = self.state_dir / "valuation_snapshots.csv"
        if not path.exists():
            return HistoryStats(status="MISSING", source="SELF_SECTOR_SNAPSHOTS")
        h = pd.read_csv(path)
        if not {"entity", metric}.issubset(h.columns):
            return HistoryStats(status="MISSING", source="SELF_SECTOR_SNAPSHOTS")
        q = h[(h["entity"].astype(str) == entity) & h[metric].notna()].copy()
        q = prepare_history(q)
        return summarize_values(q[metric], dates=q.get("trade_date"), source="SELF_SECTOR_SNAPSHOTS", minimum=int(self.cfg.get("history_min_points", 30)))

    def _stock_metric_history(self, code: str, metric: str) -> HistoryStats:
        key = (code, metric)
        if key in self._history_cache:
            return self._history_cache[key]
        h = self.p.stock_history_indicator(code)
        source = str(h.attrs.get("history_source", "AKShare/LeGu"))
        if h.attrs.get("history_status") == "FAILED":
            stats = HistoryStats(status="FAILED", source=source)
            self._history_cache[key] = stats
            return stats
        candidates = {"pe": ["pe_ttm", "pe"], "pb": ["pb"], "ps": ["ps_ttm", "ps"]}.get(metric, [metric])
        col = next((candidate for candidate in candidates if candidate in h.columns), None)
        if col is None:
            stats = HistoryStats(status="METRIC_MISSING", source=source)
        else:
            q = prepare_history(h).tail(750)
            values = pd.to_numeric(q[col], errors="coerce")
            q = q[(values > 0) & np.isfinite(values)].copy()
            values = pd.to_numeric(q[col], errors="coerce")
            stats = summarize_values(values, dates=q.get("trade_date"), source=source, minimum=int(self.cfg.get("history_min_points", 30)))
        self._history_cache[key] = stats
        return stats

    def _blend_multiple(self, code: str, metric: str, model_value: float) -> tuple[float, str, HistoryStats, float]:
        stats = self._stock_metric_history(code, metric)
        kappa = float(self.cfg.get("valuation_policy", {}).get("history_shrinkage_kappa", 60))
        value, weight = shrink_log_value(model_value, stats, kappa)
        source = f"log-shrinkage model/history; w_hist={weight:.3f}; {stats.status}"
        return value, source, stats, weight

    def evaluate_sector(self, name: str, c: dict[str, Any]) -> dict[str, Any]:
        u, resolved = self.p.sector_universe(c, purpose="valuation")
        if u.empty:
            return {"entity": name, "type": "sector", "valuation_status": "DATA_UNAVAILABLE", "error": "未解析到估值样本"}
        m = self._aggregate(u)
        configured = str(c.get("bootstrap_metric", "pe"))
        min_metric_cov = float(self.cfg.get("valuation_policy", {}).get("min_metric_mcap_coverage", 0.70))
        primary = configured
        if configured == "pe" and not np.isfinite(m.get("pe", np.nan)):
            primary = next((metric for metric in ("ps", "pb") if m.get(f"{metric}_mcap_coverage", 0) >= min_metric_cov), "pe")
        min_cov = float(c.get("min_data_coverage", self.cfg.get("report_policy", {}).get("min_sector_data_coverage", 0.75)))
        fair_model, note = self._model_fair(name, c, m) if m.get("data_coverage", 0) >= min_cov else ({"pe": np.nan, "pb": np.nan, "ps": np.nan}, "数据覆盖不足")
        history = self._sector_history(name, primary)
        bootstrap = c.get("bootstrap_fair") if primary == configured else None
        if history.status != "OK" and bootstrap is not None and np.isfinite(bootstrap):
            history = HistoryStats(status="OK", points=0, effective_points=1.0, source="BOOTSTRAP", confidence=0.15, median=float(bootstrap))
        fair, w_hist = shrink_log_value(fair_model.get(primary, np.nan), history, float(self.cfg.get("valuation_policy", {}).get("history_shrinkage_kappa", 60)))
        current = m.get(primary, np.nan)
        deviation = current / fair - 1 if np.isfinite(current) and current > 0 and np.isfinite(fair) and fair > 0 else np.nan
        status = "OK" if np.isfinite(deviation) and m.get("data_coverage", 0) >= min_cov else "INSUFFICIENT_DATA"
        return {
            "entity": name, "type": "sector", "model": c["model"],
            "resolved_boards": ";".join(f"{r.kind}:{r.board_name}" for r in resolved), **m,
            "primary_metric": primary, "configured_primary_metric": configured,
            "current_primary": current, "fair_primary": fair,
            "fair_source": f"log-shrinkage w_hist={w_hist:.3f}; {history.source}",
            "value_deviation": deviation, "valuation_label": valuation_label(deviation),
            "valuation_status": status, "history_status": history.status,
            "history_points": history.points, "history_effective_points": history.effective_points,
            "history_last_date": history.last_date, "history_source": history.source,
            "history_confidence": history.confidence, "model_note": note,
        }

    @staticmethod
    def _growth_gate(model: str, revenue_growth: float, profit_growth: float, cash_conversion: float) -> str:
        if model == "utility":
            if np.isfinite(profit_growth) and profit_growth < -0.25:
                return "观察-公用事业利润显著下滑"
            return "通过-公用事业阈值"
        if model == "cyclical_growth":
            if np.isfinite(cash_conversion) and cash_conversion < 0:
                return "不通过-周期利润现金背离"
            return "通过-周期正常化判断"
        threshold = 0.10 if model in {"growth_pe", "quality_growth", "early_growth_ps", "innovation_drug"} else 0.03
        if (np.isfinite(profit_growth) and profit_growth < -0.10) or (np.isfinite(revenue_growth) and revenue_growth < -0.10):
            return "不通过-业绩下滑"
        if np.isfinite(profit_growth) and profit_growth < threshold:
            return "观察-低于模型族增长阈值"
        return "通过"

    def _residual_history(self, code: str) -> HistoryStats:
        path = self.state_dir / "stock_valuation_snapshots.csv"
        if not path.exists():
            return HistoryStats(status="MISSING", source="POINT_IN_TIME_STOCK_SNAPSHOTS")
        h = pd.read_csv(path, dtype={"code": str})
        if not {"code", "valuation_residual"}.issubset(h.columns):
            return HistoryStats(status="MISSING", source="POINT_IN_TIME_STOCK_SNAPSHOTS")
        q = h[(h["code"].astype(str).str.zfill(6) == code) & h["valuation_residual"].notna()].copy()
        q = prepare_history(q)
        return summarize_values(q["valuation_residual"], dates=q.get("trade_date"), source="POINT_IN_TIME_STOCK_SNAPSHOTS", minimum=int(self.cfg.get("history_min_points", 30)))

    def _previous_target(self, code: str) -> float | None:
        path = self.state_dir / "stock_valuation_snapshots.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path, dtype={"code": str})
        if not {"code", "target_position_pct"}.issubset(frame.columns):
            return None
        rows = prepare_history(frame[frame["code"].astype(str).str.zfill(6) == code])
        if rows.empty:
            return None
        value = pd.to_numeric(rows.iloc[-1]["target_position_pct"], errors="coerce")
        return float(value) if pd.notna(value) else None

    def evaluate_stock(self, code: str, info: dict[str, Any]) -> dict[str, Any]:
        x = self.master[self.master["code"].astype(str) == str(code)]
        config_name = str(info.get("name", code))
        if x.empty:
            return {"entity": config_name, "code": code, "type": "stock", "valuation_status": "DATA_UNAVAILABLE", "action": "NO_TRADE", "error": "股票不存在或无行情"}
        r = x.iloc[0]
        market_name = str(r.get("name", ""))
        name_match = _norm_name(config_name) == _norm_name(market_name)
        sector = str(info.get("sector", ""))
        c = self.cfg.get("sectors", {}).get(sector)
        if c is None and isinstance(info.get("model_override"), dict):
            c = dict(info["model_override"])
        if c is None:
            return {
                "entity": config_name, "config_name": config_name, "market_name": market_name,
                "name_match": name_match, "code": code, "type": "stock", "sector": sector,
                "model_family": "UNRESOLVED", "valuation_status": "MODEL_UNRESOLVED",
                "data_quality": 0.0, "valuation_confidence": "NO_VALUATION",
                "target_position_pct": 0, "action": "NO_TRADE",
                "action_reason": "行业未配置且无model_override，禁止通用模型静默回退",
            }
        model = str(c["model"])
        profit_method = str(r.get("ttm_profit_method", "MISSING"))
        profit_conf = float(r.get("ttm_profit_confidence", 0) or 0)
        if np.isfinite(r.get("pe_ttm_calc", np.nan)) and float(r["pe_ttm_calc"]) > 0:
            pe, pe_method, pe_conf = float(r["pe_ttm_calc"]), profit_method, profit_conf
        elif pd.isna(r.get("ttm_profit")) and np.isfinite(r.get("pe_dynamic", np.nan)) and float(r["pe_dynamic"]) > 0:
            pe, pe_method, pe_conf = float(r["pe_dynamic"]), "QUOTE_PROVIDER_PE", TTM_CONFIDENCE["QUOTE_PROVIDER_PE"]
        else:
            pe, pe_method, pe_conf = np.nan, "MISSING", 0.0
        def finite(name: str) -> float:
            value = r.get(name, np.nan)
            return float(value) if np.isfinite(value) else np.nan
        revenue_growth = finite("revenue_yoy_cur") / 100 if np.isfinite(finite("revenue_yoy_cur")) else np.nan
        profit_growth = finite("profit_yoy_cur") / 100 if np.isfinite(finite("profit_yoy_cur")) else np.nan
        net_margin = finite("ttm_profit") / finite("ttm_revenue") if finite("ttm_revenue") > 0 else np.nan
        m = {
            "pe": pe, "pb": finite("pb"), "ps": finite("ps_ttm_calc"),
            "revenue_growth": revenue_growth, "profit_growth": profit_growth,
            "roe_ttm_pct": finite("ttm_roe_pct"), "roe_h1_pct": finite("roe_h1_pct_cur"),
            "cash_conversion": finite("cash_conversion"), "net_margin": net_margin,
            "profitable_mcap_coverage": 1.0 if finite("ttm_profit") > 0 else 0.0,
        }
        fair_model, model_note = self._model_fair(config_name, c, m)
        default_weights = {
            "utility": {"pb": 0.40, "pe": 0.30, "ocf_yield": 0.30},
            "mature_consumer": {"ocf_yield": 0.40, "pe": 0.40, "pb": 0.20},
            "quality_growth": {"pe": 0.50, "ocf_yield": 0.30, "ps": 0.20},
            "cyclical_growth": {"pe": 0.40, "pb": 0.30, "ps": 0.30},
            "innovation_drug": {"ps": 1.00}, "early_growth_ps": {"ps": 1.00},
            "growth_pe": {"pe": 0.70, "ps": 0.30},
        }
        weights = c.get("model_weights", default_weights.get(model, {str(c.get("bootstrap_metric", "pe")): 1.0}))
        implied: list[tuple[float, float, str]] = []
        fair_by_metric: dict[str, float] = {"pe": np.nan, "pb": np.nan, "ps": np.nan}
        history_by_metric: dict[str, HistoryStats] = {}
        source_by_metric: dict[str, str] = {}
        current_by_metric = {"pe": pe, "pb": m["pb"], "ps": m["ps"]}
        per_share = {"pe": finite("ttm_eps"), "pb": finite("bvps"), "ps": finite("sps_ttm")}
        for metric in ("pe", "pb", "ps"):
            if float(weights.get(metric, 0)) <= 0:
                continue
            fair_metric, source, stats, _ = self._blend_multiple(code, metric, fair_model.get(metric, np.nan))
            fair_by_metric[metric] = fair_metric
            history_by_metric[metric] = stats
            source_by_metric[metric] = source
            basis = per_share[metric]
            price_value = basis * fair_metric if np.isfinite(basis) and basis > 0 and np.isfinite(fair_metric) else np.nan
            implied.append((price_value, float(weights[metric]), metric.upper()))
        if float(weights.get("ocf_yield", 0)) > 0:
            ocfps = finite("ttm_ocfps")
            target_yield = float(c.get("target_ocf_yield", 0.06))
            ocf_price = ocfps / target_yield if ocfps > 0 and target_yield > 0 else np.nan
            implied.append((ocf_price, float(weights["ocf_yield"]), "OCF_YIELD_PROXY"))
        center, disagreement, models_used = weighted_geometric_price(implied)
        history_conf = max((s.confidence for s in history_by_metric.values()), default=0.0)
        ttm_revenue_conf = finite("ttm_revenue_confidence")
        ttm_score = max(0.0, np.nanmean([profit_conf, ttm_revenue_conf]) if np.isfinite(ttm_revenue_conf) else profit_conf)
        growth_score = float(np.mean([np.isfinite(revenue_growth), np.isfinite(profit_growth)]))
        cash_score = finite("ttm_ocfps_confidence") if np.isfinite(finite("cash_conversion")) else 0.0
        report_cov = float(self.cfg.get("_report_min_coverage", 1.0))
        quality = data_quality_score({
            "ttm": ttm_score, "growth": growth_score, "history": history_conf,
            "cashflow": cash_score, "sector": 1.0, "freshness": report_cov,
        })
        if not name_match:
            quality = max(0.0, quality - 0.05)
        q_label = confidence_label(quality)
        base_uncertainty = float(c.get("base_uncertainty", 0.14))
        low, high, sigma = fair_value_interval(center, disagreement, quality, base_uncertainty)
        margin = required_margin(float(c.get("base_margin", 0.20)), disagreement, quality, float(c.get("cycle_penalty", 0.0)))
        residual_stats = self._residual_history(code)
        bands = trade_bands(center, low, margin, residual_stats, sigma)
        price = finite("price")
        residual = math.log(price / center) if price > 0 and center > 0 else np.nan
        if residual_stats.status == "OK" and np.isfinite(residual):
            residual_values_path = self.state_dir / "stock_valuation_snapshots.csv"
            residual_frame = prepare_history(pd.read_csv(residual_values_path, dtype={"code": str}))
            rv = pd.to_numeric(residual_frame.loc[residual_frame["code"].astype(str).str.zfill(6) == code, "valuation_residual"], errors="coerce").dropna()
            quantile = float((rv <= residual).mean()) if len(rv) else np.nan
            rz = robust_z(residual, residual_stats)
        else:
            quantile, rz = np.nan, np.nan
        bands["valuation_quantile"] = quantile
        bands["valuation_robust_z"] = rz
        gate = self._growth_gate(model, revenue_growth, profit_growth, finite("cash_conversion"))
        stage = str(info.get("mainline_stage") or self.cfg.get("_mainline_stages", {}).get(sector, "UNKNOWN"))
        status = "OK" if q_label != "NO_VALUATION" and np.isfinite(center) else "INSUFFICIENT_DATA"
        target, action, reason = position_policy(
            price, bands, stage, status == "OK", gate, previous_target=self._previous_target(code)
        )
        volatility_scale = _clip(0.22 / sigma, 0.50, 1.00) if np.isfinite(sigma) and sigma > 0 else 0.50
        if target > 0 and action != "HOLD_HYSTERESIS":
            target = int(5 * round((target * volatility_scale) / 5))
        if np.isfinite(quantile):
            reason += f"；估值Q{quantile * 100:.0f}"
        reason += f"；安全边际{margin:.0%}；数据{q_label}"
        if volatility_scale < 0.999:
            reason += f"；不确定性仓位系数{volatility_scale:.0%}"
        primary = str(c.get("bootstrap_metric", next(iter(weights))))
        primary_history = history_by_metric.get(primary, HistoryStats(status="NOT_USED", source="NONE"))
        current_primary = current_by_metric.get(primary, np.nan)
        fair_primary = fair_by_metric.get(primary, np.nan)
        deviation = price / center - 1 if price > 0 and center > 0 else np.nan
        return {
            "entity": config_name, "config_name": config_name, "market_name": market_name, "name_match": name_match,
            "code": code, "type": "stock", "sector": sector, "model_family": model,
            "price": price, "market_cap": finite("market_cap"), "pe": pe, "pb": m["pb"], "ps": m["ps"],
            "fair_pe": fair_by_metric["pe"], "fair_pb": fair_by_metric["pb"], "fair_ps": fair_by_metric["ps"],
            "primary_metric": primary, "current_primary": current_primary, "fair_primary": fair_primary,
            "fair_source": "; ".join(source_by_metric.values()), "models_used": ";".join(models_used),
            "fair_price_low": low, "fair_price_center": center, "fair_price_high": high,
            "model_disagreement": disagreement, "sigma_total": sigma, **bands,
            "margin_of_safety": margin, "value_deviation": deviation, "valuation_label": valuation_label(deviation),
            "revenue_growth": revenue_growth, "profit_growth": profit_growth,
            "cash_conversion": finite("cash_conversion"), "roe": finite("ttm_roe_pct"),
            "gross_margin": finite("gross_margin_pct_cur"), "gross_margin_trend": finite("gross_margin_trend_pct"),
            "roe_trend": finite("roe_trend_pct"), "data_quality": quality,
            "valuation_confidence": q_label, "valuation_status": status,
            "history_status": primary_history.status, "history_points": primary_history.points,
            "history_effective_points": primary_history.effective_points,
            "history_last_date": primary_history.last_date, "history_source": primary_history.source,
            "history_confidence": primary_history.confidence,
            "residual_history_status": residual_stats.status, "residual_history_points": residual_stats.points,
            "valuation_residual": residual, "ttm_method": profit_method,
            "ttm_profit_method": profit_method, "ttm_profit_confidence": profit_conf,
            "ttm_revenue_method": r.get("ttm_revenue_method", "MISSING"),
            "ttm_revenue_confidence": finite("ttm_revenue_confidence"),
            "pe_method": pe_method, "pe_confidence": pe_conf,
            "roe_method": r.get("ttm_roe_method", "MISSING"), "roe_confidence": finite("ttm_roe_confidence"),
            "mainline_stage": stage, "growth_gate": gate, "target_position_pct": target,
            "volatility_position_scale": volatility_scale,
            "action": action, "action_reason": reason, "model_note": model_note,
            "stock_source": info.get("source", "config"),
        }

    @staticmethod
    def _latest_weekday(today: date) -> date:
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        return today

    def save_snapshots(self, sector_rows: list[dict[str, Any]], stock_rows: list[dict[str, Any]] | None = None) -> Path:
        try:
            now_dt = datetime.now(ZoneInfo("Asia/Shanghai"))
        except Exception:  # pragma: no cover
            now_dt = datetime.now().astimezone()
        timestamp = now_dt.isoformat(timespec="seconds")
        trade_day = self._latest_weekday(now_dt.date()).isoformat()
        sector_path = self.state_dir / "valuation_snapshots.csv"
        sector_columns = ["timestamp", "trade_date", "date", "entity", "pe", "pb", "ps"]
        sector_new = pd.DataFrame([{
            "timestamp": timestamp, "trade_date": trade_day, "date": trade_day,
            "entity": row["entity"], "pe": row.get("pe"), "pb": row.get("pb"), "ps": row.get("ps"),
        } for row in sector_rows if not row.get("error")], columns=sector_columns)
        if sector_path.exists():
            old = pd.read_csv(sector_path)
            if "trade_date" not in old:
                old["trade_date"] = old.get("date", pd.to_datetime(old.get("timestamp"), errors="coerce").dt.date.astype("string"))
            sector_new = pd.concat([old, sector_new], ignore_index=True)
        if len(sector_new):
            sector_new = sector_new.sort_values("timestamp").drop_duplicates(["trade_date", "entity"], keep="last")
        sector_new.to_csv(sector_path, index=False, encoding="utf-8-sig")

        if stock_rows is not None:
            stock_path = self.state_dir / "stock_valuation_snapshots.csv"
            stock_columns = [
                "timestamp", "trade_date", "code", "entity", "price", "fair_price_center",
                "valuation_residual", "data_quality", "model_family", "target_position_pct", "action",
            ]
            stock_new = pd.DataFrame([{
                "timestamp": timestamp, "trade_date": row.get("history_last_date") or trade_day,
                "code": row.get("code"), "entity": row.get("entity"), "price": row.get("price"),
                "fair_price_center": row.get("fair_price_center"), "valuation_residual": row.get("valuation_residual"),
                "data_quality": row.get("data_quality"), "model_family": row.get("model_family"),
                "target_position_pct": row.get("target_position_pct"), "action": row.get("action"),
            } for row in stock_rows if row.get("valuation_status") == "OK"], columns=stock_columns)
            if stock_path.exists():
                old = pd.read_csv(stock_path, dtype={"code": str})
                stock_new = pd.concat([old, stock_new], ignore_index=True)
            if len(stock_new):
                stock_new["code"] = stock_new["code"].astype(str).str.zfill(6)
                stock_new = stock_new.sort_values("timestamp").drop_duplicates(["trade_date", "code"], keep="last")
            stock_new.to_csv(stock_path, index=False, encoding="utf-8-sig")
        return sector_path


def _coverage(reference_codes: set[str], available_codes: set[str]) -> float:
    if not reference_codes:
        return 0.0
    return len(reference_codes & available_codes) / len(reference_codes)


def select_report_periods(
    provider: ValuationDataProvider,
    config: dict[str, Any],
    today: date | None = None,
) -> ReportSelection:
    """Resolve the latest sufficiently complete report period.

    Explicit dates in config are respected. With report_date="auto", the
    newest completed interim period is tried first and the function falls back
    if cross-sectional coverage is below the configured threshold.
    """
    min_cov = float(config.get("report_policy", {}).get("min_financial_coverage", 0.80))
    spot = provider.spot()
    spot_codes = set(spot["code"].dropna().astype(str))

    explicit = str(config.get("report_date", "auto"))
    candidates = [explicit] if explicit.lower() != "auto" else completed_report_candidates(today=today)
    errors: list[str] = []

    for current in candidates:
        try:
            _, auto_prior, auto_annual = report_period_triplet(current)
            prior_cfg = str(config.get("prior_report_date", "auto"))
            annual_cfg = str(config.get("annual_report_date", "auto"))
            prior = prior_cfg if explicit.lower() != "auto" and prior_cfg.lower() != "auto" else auto_prior
            annual = annual_cfg if explicit.lower() != "auto" and annual_cfg.lower() != "auto" else auto_annual

            cur = provider.performance(current)
            pri = provider.performance(prior)
            ann = provider.performance(annual)
            cur_codes = set(cur["code"].dropna().astype(str))
            pri_codes = set(pri["code"].dropna().astype(str))
            ann_codes = set(ann["code"].dropna().astype(str))
            current_cov = _coverage(spot_codes, cur_codes)
            # Compare old periods against current reporters rather than today's
            # whole market, so recent IPOs do not falsely fail the audit.
            prior_cov = _coverage(cur_codes, pri_codes)
            annual_cov = _coverage(cur_codes, ann_codes)
            if explicit.lower() != "auto" or (current_cov >= min_cov and prior_cov >= min_cov and annual_cov >= min_cov):
                return ReportSelection(
                    current=current, prior=prior, annual=annual,
                    current_coverage=current_cov, prior_coverage=prior_cov, annual_coverage=annual_cov,
                    source="config" if explicit.lower() != "auto" else "auto-latest-complete",
                )
            errors.append(
                f"{current}: current={current_cov:.1%}, prior={prior_cov:.1%}, annual={annual_cov:.1%}"
            )
        except Exception as exc:
            errors.append(f"{current}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "无法找到覆盖率达标的财报期。尝试结果: " + " | ".join(errors)
    )


def apply_report_selection(config: dict[str, Any], selection: ReportSelection) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config, ensure_ascii=False))
    cfg["report_date"] = selection.current
    cfg["prior_report_date"] = selection.prior
    cfg["annual_report_date"] = selection.annual
    cfg["_report_min_coverage"] = min(
        selection.current_coverage, selection.prior_coverage, selection.annual_coverage
    )
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"估值配置不是合法JSON: {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(cfg.get("sectors"), dict) or not cfg["sectors"]:
        raise ValueError("valuation_config.json: sectors 必须是非空对象")
    if not isinstance(cfg.get("stocks"), dict):
        raise ValueError("valuation_config.json: stocks 必须是对象")
    supported = {"growth_pe", "cyclical_growth", "quality_growth", "mature_consumer", "utility", "innovation_drug", "early_growth_ps"}
    errors: list[str] = []
    for sector, model in cfg["sectors"].items():
        if not isinstance(model, dict) or model.get("model") not in supported:
            errors.append(f"sector {sector!r} 的 model 无效")
        if "boards" not in model:
            errors.append(f"sector {sector!r} 缺少 boards")
    for raw_code, stock in cfg["stocks"].items():
        code = str(raw_code)
        if not re.fullmatch(r"\d{6}", code):
            errors.append(f"股票代码必须是6位数字: {code!r}")
        if not isinstance(stock, dict) or not str(stock.get("name", "")).strip():
            errors.append(f"股票 {code} 缺少 name")
            continue
        if not str(stock.get("sector", "")).strip():
            errors.append(f"股票 {code} 缺少 sector")
        if stock.get("sector") not in cfg["sectors"] and not isinstance(stock.get("model_override"), dict):
            errors.append(f"股票 {code} 的 sector={stock.get('sector')!r} 未配置，且无 model_override")
    if errors:
        raise ValueError("估值配置完整性校验失败:\n- " + "\n- ".join(errors))
    return cfg
