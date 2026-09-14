from __future__ import annotations

import numpy as np
import pandas as pd

from .snapshot_store import add_amount_share


def _return(close: pd.Series, days: int) -> float:
    if len(close) <= days or close.iloc[-days - 1] <= 0:
        return np.nan
    return (close.iloc[-1] / close.iloc[-days - 1] - 1.0) * 100.0


def _log_slope(close: pd.Series, days: int, offset: int = 0) -> tuple[float, float]:
    end = len(close) - offset
    start = end - days
    if start < 0 or days < 2:
        return np.nan, np.nan
    y = np.log(close.iloc[start:end].astype(float).values)
    if not np.isfinite(y).all():
        return np.nan, np.nan
    x = np.arange(days, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(slope * 100.0), float(np.clip(r2, 0, 1))


def calculate_board_metrics(history: pd.DataFrame) -> dict[str, float | str | pd.Timestamp]:
    h = history.dropna(subset=["close"]).sort_values("date").copy()
    close = h["close"]
    if len(h) < 12:
        raise ValueError("至少需要 12 个交易日")
    slope3, r2_3 = _log_slope(close, 3)
    slope5, r2_5 = _log_slope(close, 5)
    slope10, r2_10 = _log_slope(close, 10)
    slope20, r2_20 = _log_slope(close, min(20, len(close)))
    prior_slope5, _ = _log_slope(close, 5, offset=3)
    daily = close.pct_change()
    ma20 = close.tail(20).mean()
    high20 = close.tail(20).max()
    amount = h.get("amount", pd.Series(index=h.index, dtype=float))
    turnover = h.get("turnover", pd.Series(index=h.index, dtype=float))
    amount20 = amount.tail(20).mean()
    turnover20 = turnover.tail(20).mean()
    high_series = pd.to_numeric(h.get("high", close), errors="coerce").fillna(close)
    low_series = pd.to_numeric(h.get("low", close), errors="coerce").fillna(close)
    box_high20 = float(high_series.tail(20).max())
    box_low20 = float(low_series.tail(20).min())
    level_window = min(60, len(h))
    level_high = float(high_series.tail(level_window).max())
    level_low = float(low_series.tail(level_window).min())
    level_span = level_high - level_low
    vol20 = float(daily.tail(20).std(ddof=0) * 100)
    vol5 = float(daily.tail(5).std(ddof=0) * 100)
    result: dict[str, float | str | pd.Timestamp] = {
        "as_of": h["date"].iloc[-1],
        "last_close": close.iloc[-1],
        "last_amount": float(pd.to_numeric(amount, errors="coerce").iloc[-1]) if len(amount) else np.nan,
        "ret_1d": _return(close, 1), "ret_3d": _return(close, 3),
        "ret_5d": _return(close, 5), "ret_10d": _return(close, 10),
        "ret_20d": _return(close, 20),
        "slope_3d": slope3, "slope_5d": slope5, "slope_10d": slope10,
        "slope_20d": slope20, "trend_r2_20d": r2_20,
        "acceleration": slope3 - prior_slope5 if np.isfinite(prior_slope5) else slope3 - slope10,
        "trend_r2_5d": r2_5, "trend_r2_10d": r2_10,
        "positive_days_10": float((daily.tail(10) > 0).mean()),
        "volatility_10d": float(daily.tail(10).std(ddof=0) * 100),
        "volatility_5d": vol5, "volatility_20d": vol20,
        "volatility_ratio_5_20": vol5 / vol20 if vol20 > 0 else np.nan,
        "distance_ma20": float((close.iloc[-1] / ma20 - 1) * 100) if ma20 else np.nan,
        "distance_high20": float((close.iloc[-1] / high20 - 1) * 100) if high20 else np.nan,
        "box_range_20d_pct": (box_high20 / box_low20 - 1) * 100 if box_low20 > 0 else np.nan,
        "distance_low_20d_pct": (float(close.iloc[-1]) / box_low20 - 1) * 100 if box_low20 > 0 else np.nan,
        "range_position_60d_pct": (float(close.iloc[-1]) - level_low) / level_span * 100 if level_span > 0 else 50.0,
        "distance_high_60d_pct": (float(close.iloc[-1]) / level_high - 1) * 100 if level_high > 0 else np.nan,
        "level_lookback_days": level_window,
        "amount_ratio_5_20": float(amount.tail(5).mean() / amount20) if amount20 and np.isfinite(amount20) else np.nan,
        "turnover_ratio_5_20": float(turnover.tail(5).mean() / turnover20) if turnover20 and np.isfinite(turnover20) else np.nan,
        "drawdown_10d": float((close.iloc[-1] / close.tail(10).cummax() - 1).min() * 100),
        "history_days": len(h),
    }
    # 当外部主力资金接口不可用时，用 Chaikin Money Flow 衡量量价资金压力。
    # 这是代理指标，不等同于交易所/行情商口径的主力净流入。
    if {"high", "low", "amount"}.issubset(h.columns):
        high = pd.to_numeric(h["high"], errors="coerce")
        low = pd.to_numeric(h["low"], errors="coerce")
        amount_numeric = pd.to_numeric(h["amount"], errors="coerce")
        spread = (high - low).replace(0, np.nan)
        multiplier = ((2 * close - high - low) / spread).clip(-1, 1).fillna(0)
        for window in (1, 5, 10):
            denominator = amount_numeric.tail(window).sum(min_count=1)
            result[f"flow_proxy_{window}d_pct"] = (
                float((multiplier.tail(window) * amount_numeric.tail(window)).sum() / denominator * 100)
                if denominator and np.isfinite(denominator) else np.nan
            )
    return result


def build_metric_table(
    boards: pd.DataFrame,
    histories: dict[tuple[str, str], pd.DataFrame],
    flows: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for row in boards.to_dict("records"):
        key = (str(row["kind"]), str(row["code"]))
        if key not in histories:
            continue
        try:
            metrics = calculate_board_metrics(histories[key])
            records.append({**row, **metrics})
        except (ValueError, TypeError, IndexError):
            continue
    out = pd.DataFrame(records)
    if out.empty:
        return out
    if not flows.empty:
        out = out.merge(flows, on=["kind", "name"], how="left")
    for window in (1, 5, 10):
        flow_col = f"flow_{window}d_pct"
        proxy_col = f"flow_proxy_{window}d_pct"
        if flow_col not in out:
            out[flow_col] = np.nan
        out[f"flow_{window}d_direct_pct"] = out[flow_col]
        direct = out[flow_col].notna()
        out[f"flow_{window}d_source"] = np.where(direct, "东方财富主力资金", "量价代理CMF")
        if proxy_col in out:
            out[flow_col] = out[flow_col].fillna(out[proxy_col])
        out.loc[out[flow_col].isna(), f"flow_{window}d_source"] = "缺失"
        out[f"flow_{window}d_confidence"] = np.select(
            [direct, out[flow_col].notna()], [1.0, 0.55], default=0.0,
        )
    out["rs_5d"] = out["ret_5d"] - out.groupby("kind")["ret_5d"].transform("median")
    out["rs_10d"] = out["ret_10d"] - out.groupby("kind")["ret_10d"].transform("median")
    direct_acceleration = out["flow_1d_direct_pct"] - out["flow_5d_direct_pct"] / 5.0
    proxy_acceleration = out.get("flow_proxy_1d_pct", np.nan) - out.get("flow_proxy_5d_pct", np.nan)
    same_direct_source = out["flow_1d_direct_pct"].notna() & out["flow_5d_direct_pct"].notna()
    same_proxy_source = (
        out["flow_1d_direct_pct"].isna() & out["flow_5d_direct_pct"].isna()
        & pd.Series(proxy_acceleration, index=out.index).notna()
    )
    out["flow_acceleration"] = np.select(
        [same_direct_source, same_proxy_source], [direct_acceleration, proxy_acceleration], default=np.nan,
    )
    out["flow_acceleration_source"] = np.select(
        [same_direct_source, same_proxy_source], ["东方财富同口径(日值-5日均值)", "CMF同口径变化"], default="不可比/缺失",
    )
    return add_amount_share(out)


def _rank01(s: pd.Series, higher_is_better: bool = True, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    ranked = numeric.rank(pct=True, method="average", ascending=higher_is_better)
    return ranked.fillna(neutral)


def _weighted_score(
    group: pd.DataFrame,
    weights: dict[str, float],
    pre_ranked: set[str] | None = None,
) -> pd.Series:
    score = pd.Series(0.0, index=group.index)
    weight_sum = 0.0
    pre_ranked = pre_ranked or set()
    for col, weight in weights.items():
        if col in group:
            signal = pd.to_numeric(group[col], errors="coerce").fillna(.5) if col in pre_ranked else _rank01(group[col])
            score += signal * weight
            weight_sum += weight
    return score / weight_sum * 100 if weight_sum else score


def _source_adjusted_flow_rank(group: pd.DataFrame, window: int) -> pd.Series:
    value_col = f"flow_{window}d_pct"
    source_col = f"flow_{window}d_source"
    confidence_col = f"flow_{window}d_confidence"
    result = pd.Series(.5, index=group.index, dtype=float)
    if value_col not in group:
        return result
    sources = group[source_col] if source_col in group else pd.Series("未知", index=group.index)
    raw_confidence = group[confidence_col] if confidence_col in group else pd.Series(1.0, index=group.index)
    confidence = pd.to_numeric(raw_confidence, errors="coerce").fillna(0)
    for source in sources.dropna().unique():
        mask = sources == source
        ranked = _rank01(group.loc[mask, value_col]) if int(mask.sum()) >= 3 else pd.Series(.5, index=group.index[mask])
        result.loc[mask] = .5 + (ranked - .5) * confidence.loc[mask]
    return result


def score_boards(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frames = []
    main_weights = {
        "ret_5d": .11, "ret_10d": .12, "slope_5d": .14, "slope_10d": .07,
        "trend_r2_10d": .07, "rs_10d": .10, "flow_5d_signal": .12,
        "flow_10d_signal": .06, "breadth": .08, "amount_ratio_5_20": .07,
        "positive_days_10": .06,
    }
    candidate_weights = {
        "acceleration": .18, "slope_3d": .13, "rs_5d": .09,
        "flow_1d_signal": .14, "flow_acceleration": .10, "amount_ratio_5_20": .11,
        "turnover_ratio_5_20": .05, "breadth": .08, "trend_r2_5d": .07,
        "distance_high20": .05,
    }
    ignition_weights = {
        "confirmation_score_rank_velocity_1d": .12,
        "confirmation_score_rank_velocity_3d": .08,
        "confirmation_score_delta_1d": .10,
        "breadth_delta_1d": .15,
        "breadth_delta_intraday": .05,
        "amount_share_delta_1d": .15,
        "amount_share_delta_intraday": .05,
        "flow_acceleration": .10,
        "acceleration": .10,
        "amount_ratio_5_20": .06,
        "ret_1d": .04,
    }
    for _, g in metrics.groupby("kind", sort=False):
        x = g.copy()
        for window in (1, 5, 10):
            x[f"flow_{window}d_signal"] = _source_adjusted_flow_rank(x, window)
        x["mainline_score"] = _weighted_score(x, main_weights, {"flow_5d_signal", "flow_10d_signal"})
        x["confirmation_score"] = _weighted_score(x, candidate_weights, {"flow_1d_signal"})
        # 对已经极度偏离均线/短期暴涨的板块降温，避免把末端加速误判成“即将启动”。
        crowding = ((x["distance_ma20"] - 12).clip(lower=0) * 0.7 + (x["ret_10d"] - 18).clip(lower=0) * 0.5).clip(upper=18)
        x["crowding_penalty"] = crowding.fillna(0)
        x["confirmation_score"] = (x["confirmation_score"] - x["crowding_penalty"]).clip(0, 100)
        x["candidate_score"] = x["confirmation_score"]  # 兼容 1.x 输出字段
        x["ignition_score"] = _weighted_score(x, ignition_weights)
        early_crowding = ((x["ret_5d"] - 8).clip(lower=0) * .9 + (x["ret_10d"] - 15).clip(lower=0) * .45).clip(upper=22)
        x["ignition_score"] = (x["ignition_score"] - early_crowding.fillna(0)).clip(0, 100)
        # “横盘火种”是另一条正交于动量点火的路径：低位、窄箱体、斜率平、
        # 波动收缩且没有继续破位。这里使用绝对阈值，避免弱市里按相对排名
        # 把所有下跌板块都抬成高分。
        tightness = ((18 - x["box_range_20d_pct"]) / 12).clip(0, 1)
        flatness = (1 - x["slope_20d"].abs() / .50).clip(0, 1)
        low_level = ((55 - x["range_position_60d_pct"]) / 55).clip(0, 1)
        near_support = (1 - x["distance_low_20d_pct"] / 12).clip(0, 1)
        contraction = ((1.20 - x["volatility_ratio_5_20"]) / .80).clip(0, 1)
        stable = ((x["drawdown_10d"] >= -8) & (x["slope_5d"] >= -.60)).astype(float)
        x["sideways_seed_score"] = (
            .25 * tightness + .20 * flatness + .20 * low_level + .15 * near_support
            + .10 * contraction.fillna(.5) + .10 * stable
        ) * 100
        enough_history = x["level_lookback_days"] >= 40
        box_ok = (
            enough_history & (x["box_range_20d_pct"] <= 15) & (x["slope_20d"].abs() <= .35)
            & (x["range_position_60d_pct"] <= 55) & (x["distance_high_60d_pct"] <= -8)
        )
        x["sideways_seed_status"] = "非横盘"
        x.loc[~enough_history, "sideways_seed_status"] = "数据不足"
        x.loc[box_ok & (x["sideways_seed_score"] >= 60), "sideways_seed_status"] = "横盘观察"
        x.loc[box_ok & (x["sideways_seed_score"] >= 70), "sideways_seed_status"] = "横盘火种"
        main_ok = (x["slope_5d"] > 0) & (x["ret_10d"] > 0) & (x.get("breadth", .5).fillna(.5) >= .45)
        candidate_ok = (x["slope_3d"] > 0) & (x["acceleration"] > 0) & (x["ret_10d"] < 18)
        x["status"] = "普通"
        x.loc[(x["confirmation_score"] >= 65) & candidate_ok, "status"] = "值得关注"
        x.loc[(x["confirmation_score"] >= 75) & candidate_ok, "status"] = "潜在启动"
        x.loc[(x["mainline_score"] >= 68) & main_ok, "status"] = "主线观察"
        x.loc[(x["mainline_score"] >= 80) & main_ok, "status"] = "主线核心"
        x["lifecycle"] = "Dormant"
        x.loc[(x["mainline_score"] >= 62) & main_ok, "lifecycle"] = "Diffusion"
        x.loc[(x["ignition_score"] >= 60) & (x["ret_5d"] < 8), "lifecycle"] = "Seed"
        x.loc[(x["ignition_score"] >= 72) & (x["ret_5d"] < 8), "lifecycle"] = "Ignition"
        x.loc[(x["mainline_score"] >= 80) & main_ok, "lifecycle"] = "Mainline"
        x.loc[(x["crowding_penalty"] >= 8) & (x["mainline_score"] >= 65), "lifecycle"] = "Crowded"
        x.loc[(x["mainline_score"] >= 60) & (x["slope_3d"] < 0) & (x["acceleration"] < 0), "lifecycle"] = "Decay"
        frames.append(x)
    return pd.concat(frames, ignore_index=True).sort_values("mainline_score", ascending=False)
