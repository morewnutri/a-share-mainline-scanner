"""Objective market confirmation, rotation and decay signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CONFIRMATION_WEIGHTS = {"confirm_trend": .30, "confirm_relative_strength": .20,
                               "confirm_persistence": .15, "confirm_volume": .20,
                               "confirm_breadth": .15}
EXHAUSTION_WEIGHTS = {"risk_trend": .45, "risk_relative": .35, "risk_volume": .20}


def _rank(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 3:
        return pd.Series(np.where(v.notna(), .5, np.nan), index=v.index)
    return v.rank(pct=True)


def _combine(frame: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    values = pd.DataFrame({key: pd.to_numeric(frame[key], errors="coerce") for key in weights})
    weight = pd.Series(weights)
    available = values.notna().mul(weight).sum(axis=1)
    raw = values.fillna(0).mul(weight).sum(axis=1).div(available.replace(0, np.nan)) * 100
    coverage = available / weight.sum()
    return 50 + (raw - 50) * coverage, coverage


def add_quant_radar(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    blocks = []
    for _, g in scored.groupby(["kind", "history_source"], sort=False):
        x = g.copy()
        x["confirm_trend"] = pd.to_numeric(x["mainline_score"], errors="coerce") / 100
        x["confirm_relative_strength"] = _rank(x["rs_market_5d"] if "rs_market_5d" in x else x["rs_5d"])
        x["confirm_persistence"] = pd.to_numeric(x.get("top_rank_days_10", np.nan), errors="coerce")
        x["confirm_volume"] = _rank(x["amount_ratio_5_20"])
        x["confirm_breadth"] = pd.to_numeric(x.get("breadth", np.nan), errors="coerce")
        x["market_confirmation_score"], x["market_confirmation_coverage"] = _combine(x, MARKET_CONFIRMATION_WEIGHTS)
        x["market_confirmation_rank_score"] = x["market_confirmation_score"]
        x["risk_trend"] = pd.Series(np.where(x["slope_3d"] < 0, np.minimum(-x["slope_3d"] / 2, 1), 0), index=x.index).where(x["slope_3d"].notna())
        x["risk_relative"] = pd.Series(np.where(x["rs_5d"] < 0, np.minimum(-x["rs_5d"] / 10, 1), 0), index=x.index).where(x["rs_5d"].notna())
        x["risk_volume"] = pd.Series(np.where(x["amount_ratio_5_20"] < .8, np.minimum((.8 - x["amount_ratio_5_20"]) / .8, 1), 0), index=x.index).where(x["amount_ratio_5_20"].notna())
        x["exhaustion_score"], x["exhaustion_coverage"] = _combine(x, EXHAUSTION_WEIGHTS)
        x["exhaustion_rank_score"] = x["exhaustion_score"]
        x["quant_phase"] = np.select(
            [x["lifecycle"].eq("Mainline"), x["lifecycle"].isin(["Seed", "Ignition"]),
             x["lifecycle"].eq("Decay")], ["确认主线", "量化火种", "退潮风险"], default="量化观察")
        blocks.append(x)
    return pd.concat(blocks, ignore_index=True)


def add_switch_signals(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["confirmation_change"] = pd.to_numeric(out.get("market_confirmation_score_delta_1d", np.nan), errors="coerce")
    out["turnover_change"] = pd.to_numeric(out.get("amount_share_delta_1d", np.nan), errors="coerce")
    out["switch_score"] = np.nan
    out["switch_from"] = ""
    for _, group in out.groupby(["kind", "history_source"]):
        prior = pd.to_numeric(group.get("market_confirmation_score_prev_1d", np.nan), errors="coerce")
        prior = pd.Series(prior, index=group.index)
        if prior.notna().sum() < 2:
            continue
        leader = prior.idxmax()
        out.loc[group.index, "switch_from"] = str(out.loc[leader, "name"])
        for idx in group.index.difference([leader]):
            change = out.at[idx, "confirmation_change"] - out.at[leader, "confirmation_change"]
            share = out.at[idx, "turnover_change"] - out.at[leader, "turnover_change"]
            if pd.notna(change):
                out.at[idx, "switch_score"] = change + (share * 100 if pd.notna(share) else 0)
    return out
