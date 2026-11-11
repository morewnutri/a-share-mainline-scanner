"""Optional same-session constituent validation for shortlisted boards."""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo
import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _codes(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    column = next((c for c in candidates if c in frame), None)
    return frame[column].astype("string").str.extract(r"(\d{6})", expand=False) if column else pd.Series(dtype="string")


def calculate_structure_metrics(members: pd.DataFrame, stock_quotes: pd.DataFrame,
                                limit_up_pool: pd.DataFrame, market_date: pd.Timestamp,
                                min_coverage: float = .7) -> dict[str, object]:
    result: dict[str, object] = {"structure_score": np.nan, "structure_status": "成分结构未验证",
                                 "structure_coverage": 0.0, "mapping_coverage": 0.0,
                                 "structure_breadth": np.nan, "limit_up_density": np.nan,
                                 "strong_stock_ratio": np.nan, "consecutive_limit_up_count": np.nan,
                                 "leader_follower_gap": np.nan, "structure_valid_stocks": 0}
    if members.empty or stock_quotes.empty:
        return result
    day = pd.Timestamp(market_date).normalize()
    if "market_date" not in stock_quotes or pd.to_datetime(stock_quotes["market_date"], errors="coerce").dt.normalize().ne(day).any():
        result["structure_status"] = "个股行情日期不一致"
        return result
    if "fetched_at" in members and pd.to_datetime(members["fetched_at"], errors="coerce").dt.normalize().ne(day).any():
        result["structure_status"] = "成分股映射非当日版本"
        return result
    member_codes = _codes(members, ("代码", "code", "股票代码")).dropna().drop_duplicates()
    expected = len(members)
    result["mapping_coverage"] = len(member_codes) / expected if expected else 0.0
    if expected == 0 or member_codes.empty:
        return result
    quotes = stock_quotes.copy()
    quotes["code"] = _codes(quotes, ("代码", "code", "股票代码"))
    returns_col = next((c for c in ("涨跌幅", "pct_change", "return_pct") if c in quotes), None)
    if returns_col is None:
        return result
    quotes["return_pct"] = pd.to_numeric(quotes[returns_col], errors="coerce")
    valid = quotes[quotes["code"].isin(member_codes) & quotes["return_pct"].notna()].drop_duplicates("code")
    result["structure_valid_stocks"] = len(valid)
    result["structure_coverage"] = len(valid) / expected
    if result["mapping_coverage"] < min_coverage or result["structure_coverage"] < min_coverage:
        result["structure_status"] = "成分股覆盖不足"
        return result
    returns = valid["return_pct"]
    result["structure_breadth"] = float((returns > 0).mean())
    result["strong_stock_ratio"] = float((returns >= 5).mean())
    if len(returns) >= 10:
        result["leader_follower_gap"] = float(returns.nlargest(max(1, len(returns) // 10)).mean()
                                              - returns.median())
    if not limit_up_pool.empty:
        if "market_date" not in limit_up_pool or pd.to_datetime(limit_up_pool["market_date"], errors="coerce").dt.normalize().ne(day).any():
            result["structure_status"] = "涨停池日期不一致"
            return result
        limit_codes = set(_codes(limit_up_pool, ("代码", "code", "股票代码")).dropna())
        result["limit_up_density"] = len(set(valid["code"]) & limit_codes) / len(valid)
        board_pool = limit_up_pool[_codes(limit_up_pool, ("代码", "code", "股票代码")).isin(valid["code"])]
        streak_col = next((c for c in ("连板数", "连板", "streak") if c in board_pool), None)
        if streak_col:
            result["consecutive_limit_up_count"] = int((pd.to_numeric(board_pool[streak_col], errors="coerce") >= 2).sum())
    breadth = result["structure_breadth"]
    strong = min(result["strong_stock_ratio"] / .15, 1.0)
    components = [breadth, strong]
    if pd.notna(result["limit_up_density"]):
        components.append(min(result["limit_up_density"] / .08, 1.0))
    result["structure_score"] = float(np.mean(components) * 100)
    result["structure_status"] = "结构已验证" if len(components) == 3 else "广度已验证/涨停池缺失"
    return result


def enrich_candidate_structure(scored: pd.DataFrame, provider: object, candidate_count: int = 20,
                               now: datetime | None = None) -> pd.DataFrame:
    out = scored.copy()
    for key, value in calculate_structure_metrics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp.now()).items():
        out[key] = value
    out["constituent_source"] = ""
    out["constituent_fetched_at"] = pd.NaT
    out["quote_date_basis"] = ""
    if out.empty or candidate_count <= 0:
        return out
    captured = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    captured_stamp = pd.Timestamp(captured).tz_localize(None) if captured.tzinfo else pd.Timestamp(captured)
    day = pd.to_datetime(out["as_of"]).dt.normalize().mode().iloc[0]
    if pd.Timestamp(captured.date()) != day or captured.time() < time(15, 15):
        out["structure_status"] = "非当日收盘/成分结构未验证"
        return out
    selected = set(out.nlargest(candidate_count, "ignition_score").index) | set(out.nlargest(candidate_count, "mainline_score").index)
    try:
        quotes = provider.ak.stock_zh_a_spot_em()
        quotes["market_date"] = day
    except Exception as exc:
        LOG.warning("全市场个股行情不可用，跳过结构验证: %s", exc)
        return out
    try:
        pool = provider.ak.stock_zt_pool_em(date=day.strftime("%Y%m%d"))
        pool["market_date"] = day
    except Exception as exc:
        LOG.warning("涨停池不可用，结构分只使用上涨广度和强势股比例: %s", exc)
        pool = pd.DataFrame()
    for idx in selected:
        row = out.loc[idx]
        try:
            func = provider.ak.stock_board_industry_cons_em if row["kind"] == "industry" else provider.ak.stock_board_concept_cons_em
            members = func(symbol=str(row["name"]))
            members["fetched_at"] = captured_stamp
            members["mapping_source"] = func.__name__
            cache_root = getattr(provider, "cache_dir", None)
            if cache_root is not None:
                path = cache_root / "constituents" / day.strftime("%Y-%m-%d") / f"{row['kind']}_{row['code']}.csv.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                members.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
            values = calculate_structure_metrics(members, quotes, pool, day)
            for key, value in values.items():
                out.at[idx, key] = value
            out.at[idx, "constituent_source"] = func.__name__
            out.at[idx, "constituent_fetched_at"] = captured_stamp
            out.at[idx, "quote_date_basis"] = "当日收盘后抓取"
        except Exception as exc:
            LOG.warning("%s 成分结构验证失败: %s", row["name"], exc)
    strong = (out["structure_score"] >= 70) & (out["ret_5d"] < 3)
    out.loc[strong, "miss_diagnosis"] = "结构强、指数弱"
    return out
