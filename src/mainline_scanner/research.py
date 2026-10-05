"""Research radar: point-in-time fundamentals, market confirmation and exhaustion.

All boards stay in the result. Missing optional data stays missing; a percentile
or a neutral value must never masquerade as observed earnings or policy data.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


# Optional observations are evidence scores in [0, 1], with .5 neutral.
# Keep the individual observations so an analyst can audit every component.
SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "earnings": ("eps_revision_fy1", "eps_revision_fy2", "revision_breadth",
                 "revenue_revision", "profit_growth", "gross_margin_trend"),
    "industry": ("industry_demand", "industry_price", "industry_orders",
                 "industry_capacity", "industry_inventory"),
    "policy": ("policy_level", "policy_specificity", "policy_implementation",
               "policy_economic_size"),
    "catalyst": ("catalyst_chain", "catalyst_frequency", "catalyst_forward"),
    "expectation_gap": ("earnings_surprise", "guidance_surprise"),
    "leadership": ("leader_strength", "leader_turnover", "leader_capacity",
                   "leader_sector_link", "leader_lead_lag", "leader_recovery",
                   "down_market_resilience", "rebound_leadership", "middle_army",
                   "follower_tier", "upstream_diffusion", "institutional_participation"),
    "breadth_extra": ("breadth_ma20", "breadth_high20", "median_constituent_return"),
    "attention": ("attention_rank", "news_attention", "search_attention", "dragon_tiger_seat"),
    "index": ("index_resonance",),
    "tape": ("volume_price_alignment", "aggressive_buying", "tape_acceptance",
             "orderbook_liquidity"),
    "sentiment": ("limit_up_breadth", "popularity_rank", "relay_success_rate"),
    "exhaustion_extra": ("leader_divergence", "breakout_failure_rate",
                         "relay_failure_rate", "edge_catchup", "rotation_speed",
                         "high_volume_stall", "catalyst_exhaustion"),
    "geopolitics": ("export_restriction", "tariff_exposure", "entity_risk",
                    "customer_exposure", "supplier_exposure", "substitution_benefit"),
}
ALL_SIGNALS = tuple(dict.fromkeys(c for group in SIGNAL_GROUPS.values() for c in group))
POTENTIAL_WEIGHTS = {"earnings": .30, "industry": .25, "policy": .20,
                     "catalyst": .15, "expectation_gap": .10}
CONFIRMATION_WEIGHTS = {"turnover": .25, "relative_strength": .20, "breadth": .15,
                        "persistence": .15, "leadership": .10, "liquidity": .10,
                        "attention": .05}
EXHAUSTION_WEIGHTS = {"breadth_divergence": .25, "leader_divergence": .20,
                      "turnover_efficiency_loss": .20, "failure_rate": .20,
                      "catalyst_exhaustion": .15}


def _china_local_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("Asia/Shanghai").tz_localize(None) if stamp.tzinfo else stamp


def load_observations(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"kind": str, "code": str})
    required = {"kind", "code", "available_at"}
    if not required.issubset(frame):
        raise ValueError(f"信号文件缺少字段: {sorted(required - set(frame))}")
    frame["available_at"] = frame["available_at"].map(_china_local_timestamp)
    if "published_at" in frame:
        frame["published_at"] = frame["published_at"].map(_china_local_timestamp)
        if (frame["available_at"] < frame["published_at"]).any():
            raise ValueError("available_at 不能早于 published_at")
    for col in set(frame).intersection(ALL_SIGNALS):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame[col].dropna().between(0, 1).all():
            raise ValueError(f"{col} 必须在 0..1 之间")
    return frame.sort_values("available_at")


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


def _latest_observations(scored: pd.DataFrame, observations: pd.DataFrame,
                         known_at: pd.Timestamp | None = None) -> pd.DataFrame:
    result = scored.copy()
    for col in ALL_SIGNALS:
        if col not in result:
            result[col] = np.nan
    result["evidence_latest_at"] = pd.NaT
    if observations.empty:
        return result
    # Independently select the latest published value of each field. A newer
    # sparse event must not erase an earlier, still known observation.
    for idx, row in result.iterrows():
        cutoff = pd.Timestamp(row["as_of"]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if known_at is not None:
            cutoff = min(cutoff, pd.Timestamp(known_at))
        known = observations.loc[
            (observations["kind"] == str(row["kind"]))
            & (observations["code"] == str(row["code"]))
            & (observations["available_at"] <= cutoff)
        ]
        if known.empty:
            continue
        result.at[idx, "evidence_latest_at"] = known["available_at"].max()
        for col in set(known).intersection(ALL_SIGNALS):
            valid = known[col].dropna()
            if not valid.empty:
                result.at[idx, col] = valid.iloc[-1]
    return result


def _mean_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _weighted_available(frame: pd.DataFrame, weights: Mapping[str, float]) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for col, weight in weights.items():
        v = pd.to_numeric(frame[col], errors="coerce").clip(0, 1)
        numerator += v.fillna(0) * weight
        denominator += v.notna().astype(float) * weight
    return (numerator / denominator.replace(0, np.nan) * 100, denominator / sum(weights.values()))


def _rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 3 or numeric.nunique() < 2:
        return pd.Series(np.where(numeric.notna(), .5, np.nan), index=series.index, dtype=float)
    return numeric.rank(pct=True, method="average")


def _mean_available(*series: pd.Series) -> pd.Series:
    return pd.concat(series, axis=1).mean(axis=1)


def enrich_market_history(
    metrics: pd.DataFrame, histories: Mapping[tuple[str, str], pd.DataFrame], market: pd.DataFrame,
) -> pd.DataFrame:
    out = metrics.copy()
    for col in ("turnover_share", "turnover_share_z60", "turnover_share_delta_5d",
                "turnover_share_delta_10d", "rs_market_5d", "rs_market_10d",
                "rs_market_20d", "amount_z60", "amount_growth_5d",
                "turnover_efficiency", "turnover_efficiency_change", "top_rank_days_10"):
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
    for kind, group in out.groupby("kind"):
        daily = {}
        for idx, row in group.iterrows():
            h = histories.get((str(kind), str(row["code"])))
            if h is None or len(h) < 2:
                continue
            s = h.assign(date=pd.to_datetime(h["date"]).dt.normalize()).set_index("date")["close"]
            daily[idx] = pd.to_numeric(s, errors="coerce").pct_change(fill_method=None)
        if len(daily) < 3:
            continue
        returns = pd.DataFrame(daily).tail(10)
        ranks = returns.rank(axis=1, pct=True)
        for idx in group.index:
            valid = ranks[idx].dropna() if idx in ranks else pd.Series(dtype=float)
            if not valid.empty:
                out.at[idx, "top_rank_days_10"] = float((valid >= .8).mean())
    return out


def add_research_scores(scored: pd.DataFrame, observations: pd.DataFrame | None = None,
                        known_at: pd.Timestamp | None = None) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = _latest_observations(scored, observations if observations is not None else pd.DataFrame(), known_at)
    optional_market = ("turnover_share", "turnover_share_z60", "turnover_share_delta_5d",
                       "turnover_share_delta_10d", "amount_z60", "amount_growth_5d",
                       "rs_market_5d", "rs_market_10d", "rs_market_20d",
                       "top_rank_days_10", "turnover_efficiency_change", "rs_5d",
                       "rs_10d", "rs_20d", "breadth", "positive_days_10", "last_amount")
    for col in optional_market:
        if col not in out:
            out[col] = np.nan
    for group, cols in SIGNAL_GROUPS.items():
        out[f"signal_{group}"] = _mean_columns(out, cols)
    # Policy execution is conjunctive: aspirational statements score below a
    # funded and specific implementation. Sparse policies remain partially known.
    policy = out[list(SIGNAL_GROUPS["policy"])].apply(pd.to_numeric, errors="coerce")
    policy_complete = policy.notna().all(axis=1)
    out["signal_policy"] = policy.mean(axis=1)
    out.loc[policy_complete, "signal_policy"] = policy.loc[policy_complete].prod(axis=1) ** .25
    negative_geo = _mean_available(*(out[c] for c in SIGNAL_GROUPS["geopolitics"][:-1]))
    out["geo_net_exposure"] = out["substitution_benefit"] - negative_geo
    # Geo exposure is a separate signed diagnostic, never inferred from keywords.
    out["potential_score"], out["potential_coverage"] = _weighted_available(
        out.rename(columns={f"signal_{k}": k for k in POTENTIAL_WEIGHTS}), POTENTIAL_WEIGHTS,
    )
    out["potential_rank_score"] = 50 + (out["potential_score"] - 50) * out["potential_coverage"]
    out["potential_state"] = np.where(out["potential_coverage"] > 0, "有时点证据", "基本面资料待补")
    result = []
    for _, group in out.groupby("kind", sort=False):
        x = group.copy()
        turnover = _mean_available(_rank(x["turnover_share"]), _rank(x["turnover_share_z60"]),
                                   _rank(x["turnover_share_delta_5d"]), _rank(x["turnover_share_delta_10d"]))
        x["confirm_turnover"] = turnover.combine_first(_mean_available(_rank(x["amount_z60"]), _rank(x["amount_growth_5d"])))
        rs = _mean_available(*(_rank(x[f"rs_market_{d}d"]) for d in (5, 10, 20)))
        x["confirm_relative_strength"] = _mean_available(
            rs.combine_first(_mean_available(_rank(x["rs_5d"]), _rank(x["rs_10d"]), _rank(x["rs_20d"]))),
            x["signal_index"],
        )
        x["confirm_breadth"] = _mean_available(x["breadth"], x["breadth_ma20"], x["breadth_high20"],
                                                x["median_constituent_return"], x["signal_breadth_extra"])
        x["confirm_persistence"] = _mean_available(x["positive_days_10"], x.get("top_rank_days_10", pd.Series(np.nan, index=x.index)))
        x["confirm_leadership"] = _mean_available(x["signal_leadership"], x["signal_tape"])
        x["confirm_liquidity"] = _mean_available(_rank(x["last_amount"]), x["orderbook_liquidity"])
        x["confirm_attention"] = _mean_available(x["signal_attention"], x["signal_sentiment"])
        confirmation = pd.DataFrame({name: x[f"confirm_{name}"] for name in CONFIRMATION_WEIGHTS}, index=x.index)
        x["market_confirmation_score"], x["confirmation_coverage"] = _weighted_available(
            confirmation, CONFIRMATION_WEIGHTS,
        )
        x["market_confirmation_rank_score"] = 50 + (x["market_confirmation_score"] - 50) * x["confirmation_coverage"]
        prior_breadth = pd.to_numeric(x.get("breadth_prev_1d", np.nan), errors="coerce")
        prior_breadth = pd.Series(prior_breadth, index=x.index)
        x["risk_breadth_divergence"] = np.where(
            (x["ret_5d"] > 0) & (x["breadth"] < prior_breadth),
            (prior_breadth - x["breadth"]).clip(0, 1), np.nan,
        )
        x["risk_leader_divergence"] = x["leader_divergence"]
        x["risk_turnover_efficiency_loss"] = np.where(
            (x["turnover_share_delta_5d"] > 0) & (x["turnover_efficiency_change"] < 0),
            .5 + .5 * _rank(-x["turnover_efficiency_change"]), np.nan,
        )
        x["risk_failure_rate"] = _mean_available(x["breakout_failure_rate"], x["relay_failure_rate"],
                                                 x["edge_catchup"], x["rotation_speed"],
                                                 x["high_volume_stall"], 1 - x["relay_success_rate"])
        x["risk_catalyst_exhaustion"] = x["catalyst_exhaustion"]
        risks = pd.DataFrame({name: x[f"risk_{name}"] for name in EXHAUSTION_WEIGHTS}, index=x.index)
        x["exhaustion_score"], x["exhaustion_coverage"] = _weighted_available(
            risks, EXHAUSTION_WEIGHTS,
        )
        x["exhaustion_rank_score"] = 50 + (x["exhaustion_score"] - 50) * x["exhaustion_coverage"]
        result.append(x)
    return pd.concat(result, ignore_index=True)


def add_switch_signals(scored: pd.DataFrame) -> pd.DataFrame:
    """Rank all challengers; no fixed cutoff or requirement that the old leader crashes."""
    out = scored.copy()
    out["confirmation_change"] = pd.to_numeric(out.get("market_confirmation_score_delta_1d", np.nan), errors="coerce")
    out["turnover_change"] = pd.to_numeric(out.get("turnover_share_delta_1d", np.nan), errors="coerce")
    out["confirmation_change_3d"] = pd.to_numeric(out.get("market_confirmation_score_delta_3d", np.nan), errors="coerce")
    out["turnover_change_3d"] = pd.to_numeric(out.get("turnover_share_delta_3d", np.nan), errors="coerce")
    out["switch_score"] = np.nan
    out["switch_from"] = ""
    for _, g in out.groupby("kind"):
        previous = pd.to_numeric(g.get("market_confirmation_score_prev_1d", np.nan), errors="coerce")
        previous = pd.Series(previous, index=g.index)
        has_history = previous.notna().any()
        incumbent = previous.idxmax() if has_history else g["market_confirmation_score"].idxmax()
        old = out.loc[incumbent]
        out.loc[g.index, "switch_from"] = str(old["name"])
        if not has_history:
            continue
        for idx in g.index:
            if idx == incumbent:
                continue
            challenger = out.loc[idx]
            change = challenger["confirmation_change"] - old["confirmation_change"]
            share = challenger["turnover_change"] - old["turnover_change"]
            change3 = challenger["confirmation_change_3d"] - old["confirmation_change_3d"]
            share3 = challenger["turnover_change_3d"] - old["turnover_change_3d"]
            gap = challenger["market_confirmation_score"] - old["market_confirmation_score"]
            pieces = [v for v in (change, share * 100 if pd.notna(share) else np.nan,
                                  change3, share3 * 100 if pd.notna(share3) else np.nan,
                                  gap / 10) if pd.notna(v)]
            if len(pieces) >= 2:
                out.at[idx, "switch_score"] = float(np.mean(pieces))
    out["research_phase"] = "基本面资料待补"
    for _, g in out.groupby("kind"):
        p = pd.to_numeric(g["potential_rank_score"], errors="coerce")
        c = pd.to_numeric(g["market_confirmation_rank_score"], errors="coerce")
        if not c.notna().any():
            continue
        c_high = c >= c.median()
        p_high = p >= p.median() if p.notna().any() else pd.Series(False, index=g.index)
        out.loc[g.index[c_high & p.isna()], "research_phase"] = "市场热/基本面待补"
        out.loc[g.index[p.notna() & ~p_high & ~c_high], "research_phase"] = "逻辑与市场均未确认"
        out.loc[g.index[p.notna() & ~p_high & c_high], "research_phase"] = "情绪热/基本面偏弱"
        out.loc[g.index[p_high & ~c_high], "research_phase"] = "产业先行/潜在主线"
        out.loc[g.index[p_high & c_high], "research_phase"] = "逻辑与市场共振"
        fading = (out.loc[g.index, "confirmation_change"] < 0) & (out.loc[g.index, "exhaustion_rank_score"] > 50)
        out.loc[g.index[fading], "research_phase"] = "退潮风险上升"
    return out
