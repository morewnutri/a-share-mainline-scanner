"""Market-only, point-in-time metrics. No subjective evidence enters scoring."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd

def load_market(path: str | None, reconciliation_tolerance: float = .05) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not 0 <= reconciliation_tolerance < 1:
        raise ValueError("对账容忍度必须在 0..1 之间")
    market = pd.read_csv(path)
    if not {"date", "market_amount"}.issubset(market):
        raise ValueError("全市场成交额文件需要 date, market_amount")
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market["market_amount"] = pd.to_numeric(market["market_amount"], errors="raise")
    if (market["market_amount"] <= 0).any() or market["date"].duplicated().any():
        raise ValueError("全市场成交额必须为正，日期不能重复")
    if "benchmark_close" in market:
        market["benchmark_close"] = pd.to_numeric(market["benchmark_close"], errors="raise")
    audit_ratios = {
        "price_coverage": (("valid_quote_count", "suspended_count"), "expected_stock_count"),
        "theme_coverage": (("theme_mapped_count",), "expected_stock_count"),
        "financial_coverage": (("financial_report_available_count",), "financial_report_due_count"),
        "consensus_coverage": (("consensus_covered_count",), "expected_stock_count"),
    }
    for output, (numerators, denominator) in audit_ratios.items():
        if all(col in market for col in (*numerators, denominator)):
            values = sum(pd.to_numeric(market[col], errors="raise") for col in numerators)
            base = pd.to_numeric(market[denominator], errors="raise").replace(0, np.nan)
            market[output] = values / base
    if "stock_amount_sum" in market:
        stock_sum = pd.to_numeric(market["stock_amount_sum"], errors="raise")
        market["reconciliation_error"] = (stock_sum - market["market_amount"]).abs() / market["market_amount"]
        if (market["reconciliation_error"] > reconciliation_tolerance).any():
            bad = market.loc[market["reconciliation_error"] > reconciliation_tolerance, "date"].dt.strftime("%Y-%m-%d")
            raise ValueError(f"个股与全市场成交额对账误差超过 {reconciliation_tolerance:.1%}: {', '.join(bad.head(5))}")
    return market.sort_values("date")

def enrich_market_history(
    metrics: pd.DataFrame, histories: Mapping[tuple[str, str], pd.DataFrame], market: pd.DataFrame,
) -> pd.DataFrame:
    out = metrics.copy()
    out["history_source"] = out.get("history_source", pd.Series("原始板块日线", index=out.index)).fillna("原始板块日线")
    for col in ("turnover_share", "turnover_share_z60", "turnover_share_delta_5d",
                "turnover_share_delta_10d", "rs_market_5d", "rs_market_10d",
                "rs_market_20d", "amount_z60", "amount_growth_5d",
                "turnover_efficiency", "turnover_efficiency_change", "top_rank_days_10",
                "top_rank_valid_days_10", "historical_rank_change_3d"):
        out[col] = np.nan
    if out.empty:
        return out
    m = market.set_index("date") if not market.empty else None
    for idx, row in out.iterrows():
        h = histories.get((str(row["kind"]), str(row["code"])))
        if h is None or h.empty:
            continue
        h = h.copy().sort_values("date")
        h["date"] = pd.to_datetime(h["date"]).dt.normalize()
        amount = pd.to_numeric(h.get("amount"), errors="coerce")
        if amount.notna().sum() >= 10:
            prior = amount.iloc[-min(60, len(amount)):-1]
            std = prior.std(ddof=0)
            out.at[idx, "amount_z60"] = (amount.iloc[-1] - prior.mean()) / std if std > 0 else 0.0
            old = amount.iloc[-10:-5].mean()
            out.at[idx, "amount_growth_5d"] = amount.iloc[-5:].mean() / old - 1 if old > 0 else np.nan
        if m is None:
            continue
        aligned = h.set_index("date").join(m[["market_amount"] + (["benchmark_close"] if "benchmark_close" in m else [])], how="left")
        share = (pd.to_numeric(aligned["amount"], errors="coerce") / aligned["market_amount"]).replace([np.inf, -np.inf], np.nan)
        if share.notna().any():
            out.at[idx, "turnover_share"] = share.iloc[-1]
            valid = share.dropna()
            if len(valid) >= 6 and pd.notna(share.iloc[-1]):
                out.at[idx, "turnover_share_delta_5d"] = share.iloc[-1] - share.iloc[-6] if pd.notna(share.iloc[-6]) else np.nan
            if len(valid) >= 11 and pd.notna(share.iloc[-11]):
                out.at[idx, "turnover_share_delta_10d"] = share.iloc[-1] - share.iloc[-11]
            prior = share.iloc[-61:-1].dropna()
            if len(prior) >= 10 and pd.notna(share.iloc[-1]):
                sd = prior.std(ddof=0)
                out.at[idx, "turnover_share_z60"] = (share.iloc[-1] - prior.mean()) / sd if sd > 0 else 0.0
            ret = pd.to_numeric(aligned["close"], errors="coerce").pct_change(fill_method=None)
            if "benchmark_close" in aligned:
                ret = ret - aligned["benchmark_close"].pct_change(fill_method=None)
            efficiency = ret / share.replace(0, np.nan)
            if len(efficiency) >= 6:
                out.at[idx, "turnover_efficiency"] = efficiency.iloc[-1]
                out.at[idx, "turnover_efficiency_change"] = efficiency.iloc[-5:].mean() - efficiency.iloc[-10:-5].mean() if len(efficiency) >= 10 else np.nan
        if "benchmark_close" in aligned:
            close = pd.to_numeric(aligned["close"], errors="coerce")
            benchmark = aligned["benchmark_close"]
            for days in (5, 10, 20):
                if len(aligned) > days and close.iloc[-days-1] > 0 and benchmark.iloc[-days-1] > 0 and pd.notna(benchmark.iloc[-1]):
                    out.at[idx, f"rs_market_{days}d"] = (close.iloc[-1] / close.iloc[-days-1] - benchmark.iloc[-1] / benchmark.iloc[-days-1]) * 100
    # A daily cross-sectional persistence measure, computed only from information
    # available on each of the last ten trading dates. No future board rank leaks.
    for (kind, _source), group in out.groupby(["kind", "history_source"]):
        daily = {}
        for idx, row in group.iterrows():
            h = histories.get((str(kind), str(row["code"])))
            if h is None or len(h) < 2:
                continue
            s = h.assign(date=pd.to_datetime(h["date"]).dt.normalize()).set_index("date")["close"]
            daily[idx] = pd.to_numeric(s, errors="coerce").pct_change(fill_method=None)
        if len(daily) < 3:
            continue
        common_as_of = pd.to_datetime(group["as_of"]).dt.normalize().mode().iloc[0]
        returns = pd.DataFrame(daily).loc[:common_as_of].tail(10)
        ranks = returns.rank(axis=1, pct=True)
        for idx in group.index:
            valid = ranks[idx].dropna() if idx in ranks else pd.Series(dtype=float)
            out.at[idx, "top_rank_valid_days_10"] = len(valid)
            if len(valid) >= 8 and pd.Timestamp(valid.index[-1]) == common_as_of:
                out.at[idx, "top_rank_days_10"] = float((valid >= .8).mean())
        closes = {}
        for idx, row in group.iterrows():
            h = histories.get((str(kind), str(row["code"])))
            if h is not None and not h.empty:
                closes[idx] = pd.Series(pd.to_numeric(h["close"], errors="coerce").values,
                                        index=pd.to_datetime(h["date"]).dt.normalize())
        if len(closes) >= 3:
            five_day = pd.DataFrame(closes).loc[:common_as_of].pct_change(5, fill_method=None)
            if len(five_day) >= 9:
                current_rank = five_day.iloc[-1].rank(pct=True)
                prior_rank = five_day.iloc[-4].rank(pct=True)
                for idx in group.index:
                    if pd.notna(current_rank.get(idx)) and pd.notna(prior_rank.get(idx)):
                        out.at[idx, "historical_rank_change_3d"] = current_rank[idx] - prior_rank[idx]
    return out
