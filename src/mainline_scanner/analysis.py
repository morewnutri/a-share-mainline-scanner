from __future__ import annotations

import numpy as np
import pandas as pd


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
    prior_slope5, _ = _log_slope(close, 5, offset=3)
    daily = close.pct_change()
    ma20 = close.tail(20).mean()
    high20 = close.tail(20).max()
    amount = h.get("amount", pd.Series(index=h.index, dtype=float))
    turnover = h.get("turnover", pd.Series(index=h.index, dtype=float))
    amount20 = amount.tail(20).mean()
    turnover20 = turnover.tail(20).mean()
    result: dict[str, float | str | pd.Timestamp] = {
        "as_of": h["date"].iloc[-1],
        "last_close": close.iloc[-1],
        "ret_1d": _return(close, 1), "ret_3d": _return(close, 3),
        "ret_5d": _return(close, 5), "ret_10d": _return(close, 10),
        "ret_20d": _return(close, 20),
        "slope_3d": slope3, "slope_5d": slope5, "slope_10d": slope10,
        "acceleration": slope3 - prior_slope5 if np.isfinite(prior_slope5) else slope3 - slope10,
        "trend_r2_5d": r2_5, "trend_r2_10d": r2_10,
        "positive_days_10": float((daily.tail(10) > 0).mean()),
        "volatility_10d": float(daily.tail(10).std(ddof=0) * 100),
        "distance_ma20": float((close.iloc[-1] / ma20 - 1) * 100) if ma20 else np.nan,
        "distance_high20": float((close.iloc[-1] / high20 - 1) * 100) if high20 else np.nan,
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
        direct = out[flow_col].notna()
        out[f"flow_{window}d_source"] = np.where(direct, "东方财富主力资金", "量价代理CMF")
        if proxy_col in out:
            out[flow_col] = out[flow_col].fillna(out[proxy_col])
        out.loc[out[flow_col].isna(), f"flow_{window}d_source"] = "缺失"
    out["rs_5d"] = out["ret_5d"] - out.groupby("kind")["ret_5d"].transform("median")
    out["rs_10d"] = out["ret_10d"] - out.groupby("kind")["ret_10d"].transform("median")
    out["flow_acceleration"] = out.get("flow_1d_pct", np.nan) - out.get("flow_5d_pct", np.nan)
    return out


def _rank01(s: pd.Series, higher_is_better: bool = True, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    ranked = numeric.rank(pct=True, method="average", ascending=higher_is_better)
    return ranked.fillna(neutral)


def _weighted_score(group: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=group.index)
    weight_sum = 0.0
    for col, weight in weights.items():
        if col in group:
            score += _rank01(group[col], higher_is_better=True) * weight
            weight_sum += weight
    return score / weight_sum * 100 if weight_sum else score


def score_boards(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frames = []
    main_weights = {
        "ret_5d": .11, "ret_10d": .12, "slope_5d": .14, "slope_10d": .07,
        "trend_r2_10d": .07, "rs_10d": .10, "flow_5d_pct": .12,
        "flow_10d_pct": .06, "breadth": .08, "amount_ratio_5_20": .07,
        "positive_days_10": .06,
    }
    candidate_weights = {
        "acceleration": .18, "slope_3d": .13, "rs_5d": .09,
        "flow_1d_pct": .14, "flow_acceleration": .10, "amount_ratio_5_20": .11,
        "turnover_ratio_5_20": .05, "breadth": .08, "trend_r2_5d": .07,
        "distance_high20": .05,
    }
    for _, g in metrics.groupby("kind", sort=False):
        x = g.copy()
        x["mainline_score"] = _weighted_score(x, main_weights)
        x["candidate_score"] = _weighted_score(x, candidate_weights)
        # 对已经极度偏离均线/短期暴涨的板块降温，避免把末端加速误判成“即将启动”。
        crowding = ((x["distance_ma20"] - 12).clip(lower=0) * 0.7 + (x["ret_10d"] - 18).clip(lower=0) * 0.5).clip(upper=18)
        x["crowding_penalty"] = crowding.fillna(0)
        x["candidate_score"] = (x["candidate_score"] - x["crowding_penalty"]).clip(0, 100)
        main_ok = (x["slope_5d"] > 0) & (x["ret_10d"] > 0) & (x.get("breadth", .5).fillna(.5) >= .45)
        candidate_ok = (x["slope_3d"] > 0) & (x["acceleration"] > 0) & (x["ret_10d"] < 18)
        x["status"] = "普通"
        x.loc[(x["candidate_score"] >= 65) & candidate_ok, "status"] = "值得关注"
        x.loc[(x["candidate_score"] >= 75) & candidate_ok, "status"] = "潜在启动"
        x.loc[(x["mainline_score"] >= 68) & main_ok, "status"] = "主线观察"
        x.loc[(x["mainline_score"] >= 80) & main_ok, "status"] = "主线核心"
        frames.append(x)
    return pd.concat(frames, ignore_index=True).sort_values("mainline_score", ascending=False)
