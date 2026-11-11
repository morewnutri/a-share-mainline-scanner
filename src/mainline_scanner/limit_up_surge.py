"""Compare board limit-up breadth across adjacent completed market sessions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from .stock_structure import _codes
from .trading_calendar import expected_close_session

LOG = logging.getLogger(__name__)


def surge_rule(min_today: int, min_increase: int, min_ratio: float) -> str:
    return (f"今日涨停数 ≥ {min_today}，较前一交易日增加 ≥ {min_increase}，"
            f"且前日为 0 或今日/前日 ≥ {min_ratio:g} 倍；两日均使用本次抓取的成分股名单。")


def _pool_codes(pool: pd.DataFrame) -> set[str]:
    if pool.empty:
        raise ValueError("涨停池为空，无法核实涨停数")
    codes = _codes(pool, ("代码", "code", "股票代码")).dropna()
    if codes.empty:
        raise ValueError("涨停池缺少有效股票代码")
    return set(codes.astype(str))


def enrich_limit_up_surge(scored: pd.DataFrame, provider: object, captured_at: datetime,
                          min_today: int = 3, min_increase: int = 3,
                          min_ratio: float = 2.0, workers: int = 3) -> pd.DataFrame:
    """Attach an independent, auditable screen to every scored industry/concept board."""
    if min_today < 1 or min_increase < 1 or min_ratio < 1 or workers < 1:
        raise ValueError("涨停突增门槛和并发数必须为正，倍数至少为 1")
    out = scored.copy()
    out["limit_up_today"] = np.nan
    out["limit_up_previous"] = np.nan
    out["limit_up_increase"] = np.nan
    out["limit_up_ratio"] = np.nan
    out["limit_up_surge"] = False
    out["limit_up_status"] = "未核实"
    out["limit_up_previous_date"] = ""
    out["limit_up_membership_date"] = ""
    if out.empty:
        return out

    days = pd.to_datetime(out["as_of"], errors="coerce").dt.normalize()
    day = expected_close_session(pd.Timestamp(captured_at.date()), captured_at)
    current = days.eq(day)
    out.loc[~current, "limit_up_status"] = "板块行情非本次收盘日"
    if not current.any():
        return out
    previous = expected_close_session(day - pd.Timedelta(days=1), captured_at)
    out["limit_up_previous_date"] = previous.strftime("%Y-%m-%d")
    out["limit_up_membership_date"] = captured_at.date().isoformat()
    try:
        today_codes = _pool_codes(provider.ak.stock_zt_pool_em(date=day.strftime("%Y%m%d")))
        previous_codes = _pool_codes(provider.ak.stock_zt_pool_em(date=previous.strftime("%Y%m%d")))
    except Exception as exc:
        LOG.warning("相邻交易日涨停池不可用: %s", exc)
        out["limit_up_status"] = f"涨停池不可用: {exc}"
        return out

    def count_board(idx: object) -> tuple[object, dict[str, object]]:
        row = out.loc[idx]
        kind, code = str(row["kind"]), str(row["code"])
        cache_root = getattr(provider, "cache_dir", None)
        path = (cache_root / "constituents" / captured_at.date().isoformat() /
                f"{kind}_{code}.csv.gz") if cache_root is not None else None
        try:
            if path is not None and path.is_file() and not getattr(provider, "refresh", False):
                members = pd.read_csv(path, dtype={"代码": str, "code": str, "股票代码": str})
            else:
                func = (provider.ak.stock_board_industry_cons_em if kind == "industry"
                        else provider.ak.stock_board_concept_cons_em)
                members = func(symbol=str(row["name"]))
                if path is not None and not members.empty:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    members.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
            codes = set(_codes(members, ("代码", "code", "股票代码")).dropna().astype(str))
            if not codes:
                raise ValueError("成分股名单为空或缺少有效代码")
            today = len(codes & today_codes)
            prior = len(codes & previous_codes)
            increase = today - prior
            ratio = today / prior if prior else np.nan
            surge = (today >= min_today and increase >= min_increase
                     and (prior == 0 or ratio >= min_ratio))
            return idx, {"limit_up_today": today, "limit_up_previous": prior,
                         "limit_up_increase": increase, "limit_up_ratio": ratio,
                         "limit_up_surge": bool(surge), "limit_up_status": "已核实"}
        except Exception as exc:
            LOG.warning("%s 成分股涨停数核实失败: %s", row["name"], exc)
            return idx, {"limit_up_status": f"成分股不可用: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(count_board, idx) for idx in out.index[current]]
        for future in as_completed(futures):
            idx, values = future.result()
            for key, value in values.items():
                out.at[idx, key] = value
    return out
