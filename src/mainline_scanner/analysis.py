from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib

from .snapshot_store import add_amount_share, MODEL_VERSION
from .quant_scores import MARKET_CONFIRMATION_WEIGHTS, EXHAUSTION_WEIGHTS


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
        "history_source": str(h["data_source"].iloc[-1]) if "data_source" in h else "原始板块日线",
    }
    # 外部主力资金接口不可用时，CMF 只作为量价代理，不冒充真实主力净流入。
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
        if "code" in flows:
            flow_frame = flows.copy()
            flow_frame["code"] = flow_frame["code"].astype(str)
            flow_frame = flow_frame.drop(columns=["name"], errors="ignore").drop_duplicates(["kind", "code"])
            out = out.merge(flow_frame, on=["kind", "code"], how="left")
            out["flow_match_method"] = "verified_code"
            if "目录来源" in out:
                incompatible = ~out["目录来源"].fillna("东方财富").astype(str).eq("东方财富")
                for col in [c for c in out if c.startswith("flow_") and (c.endswith("_pct") or c.endswith("_amount"))]:
                    out.loc[incompatible, col] = np.nan
                out.loc[incompatible, "flow_match_method"] = "source_namespace_mismatch"
        else:
            unique_flows = flows.drop_duplicates(["kind", "name"], keep=False)
            out = out.merge(unique_flows, on=["kind", "name"], how="left")
            out["flow_match_method"] = "unique_name_unverified"
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
    out["rs_20d"] = out["ret_20d"] - out.groupby("kind")["ret_20d"].transform("median")
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
        [same_direct_source, same_proxy_source],
        ["东方财富同口径(日值-5日均值)", "CMF同口径变化"],
        default="不可比/缺失",
    )
    return add_amount_share(out)


def _rank01(s: pd.Series, higher_is_better: bool = True, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(s, errors="coerce")
    ranked = numeric.rank(pct=True, method="average", ascending=higher_is_better)
    return ranked


def _weighted_score(
    group: pd.DataFrame,
    weights: dict[str, float],
    pre_ranked: set[str] | None = None,
) -> pd.DataFrame:
    score = pd.Series(0.0, index=group.index)
    available = pd.Series(0.0, index=group.index)
    contributions = {}
    pre_ranked = pre_ranked or set()
    for col, weight in weights.items():
        if col in group:
            signal = pd.to_numeric(group[col], errors="coerce") if col in pre_ranked else _rank01(group[col])
            contributions[f"{col}_contribution"] = signal * weight * 100
            score += signal.fillna(0) * weight
            available += signal.notna().astype(float) * weight
    raw = score.div(available.replace(0, np.nan)) * 100
    coverage = available / sum(weights.values())
    result = pd.DataFrame({"raw_score": raw, "coverage": coverage,
                           "rank_score": 50 + (raw - 50) * coverage}, index=group.index)
    return result.assign(**contributions)


def _grouped_score(group: pd.DataFrame, groups: dict[str, tuple[float, dict[str, float]]],
                   pre_ranked: set[str] | None = None) -> pd.DataFrame:
    numerator = pd.Series(0.0, index=group.index)
    covered = pd.Series(0.0, index=group.index)
    parts: dict[str, pd.Series] = {}
    for name, (weight, factors) in groups.items():
        component = _weighted_score(group, factors, pre_ranked)
        effective = weight * component["coverage"]
        numerator += component["raw_score"].fillna(0) * effective
        covered += effective
        parts[f"{name}_contribution"] = component["raw_score"] * effective
    raw = numerator.div(covered.replace(0, np.nan))
    return pd.DataFrame({"raw_score": raw, "coverage": covered,
                         "rank_score": 50 + (raw - 50) * covered, **parts}, index=group.index)


def _source_adjusted_flow_rank(group: pd.DataFrame, window: int) -> pd.Series:
    value_col = f"flow_{window}d_pct"
    source_col = f"flow_{window}d_source"
    confidence_col = f"flow_{window}d_confidence"
    result = pd.Series(np.nan, index=group.index, dtype=float)
    if value_col not in group:
        return result
    sources = group[source_col] if source_col in group else pd.Series("未知", index=group.index)
    raw_confidence = group[confidence_col] if confidence_col in group else pd.Series(1.0, index=group.index)
    confidence = pd.to_numeric(raw_confidence, errors="coerce").fillna(0)
    for source in sources.dropna().unique():
        mask = sources == source
        ranked = _rank01(group.loc[mask, value_col]) if int(mask.sum()) >= 3 else pd.Series(.5, index=group.index[mask]).where(group.loc[mask, value_col].notna())
        result.loc[mask] = .5 + (ranked - .5) * confidence.loc[mask]
    return result


def _source_adjusted_acceleration_rank(group: pd.DataFrame) -> pd.Series:
    """资金加速度也必须按同一数据来源内部比较，避免 CMF 与东方财富口径混排。"""
    result = pd.Series(np.nan, index=group.index, dtype=float)
    if "flow_acceleration" not in group:
        return result
    sources = group.get("flow_acceleration_source", pd.Series("未知", index=group.index)).astype(str)
    c1 = pd.to_numeric(group.get("flow_1d_confidence", 0.0), errors="coerce")
    c5 = pd.to_numeric(group.get("flow_5d_confidence", 0.0), errors="coerce")
    confidence = pd.concat([pd.Series(c1, index=group.index), pd.Series(c5, index=group.index)], axis=1).min(axis=1).fillna(0.0)
    values = pd.to_numeric(group["flow_acceleration"], errors="coerce")
    for source in sources.dropna().unique():
        if source in {"不可比/缺失", "未知"}:
            continue
        mask = (sources == source) & values.notna()
        if not mask.any():
            continue
        ranked = _rank01(values.loc[mask]) if int(mask.sum()) >= 3 else pd.Series(.5, index=group.index[mask])
        result.loc[mask] = .5 + (ranked - .5) * confidence.loc[mask]
    return result


def _ignition_history_coverage(frame: pd.DataFrame) -> pd.Series:
    trajectory_cols = [
        "confirmation_score_rank_velocity_1d",
        "confirmation_score_rank_velocity_3d",
        "confirmation_score_delta_1d",
        "breadth_delta_1d",
        "breadth_delta_intraday",
        "amount_share_delta_1d",
        "amount_share_delta_intraday",
    ]
    present = [col for col in trajectory_cols if col in frame]
    if not present:
        return pd.Series(0.0, index=frame.index)
    return frame[present].apply(pd.to_numeric, errors="coerce").notna().mean(axis=1)


def score_boards(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    metrics = metrics.copy()
    if "history_source" not in metrics:
        metrics["history_source"] = "来源未标记"
    frames = []
    main_groups = {
        "trend": (.28, {"ret_5d": .25, "ret_10d": .25, "slope_5d": .25,
                          "slope_10d": .15, "trend_r2_10d": .10}),
        "relative_strength": (.20, {"rs_10d": 1.0}),
        "volume": (.22, {"flow_5d_signal": .45, "flow_10d_signal": .25,
                           "amount_ratio_5_20": .30}),
        "breadth": (.15, {"breadth": 1.0}),
        "persistence": (.15, {"positive_days_10": 1.0}),
    }
    candidate_weights = {
        "acceleration": .18, "slope_3d": .13, "rs_5d": .09,
        "flow_1d_signal": .14, "flow_acceleration_signal": .10, "amount_ratio_5_20": .11,
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
        "flow_acceleration_signal": .10,
        "acceleration": .10,
        "amount_ratio_5_20": .06,
        "ret_1d": .04,
        "historical_rank_change_3d": .10,
        "top_rank_days_10": .08,
    }
    config_hash = hashlib.sha256(repr((main_groups, candidate_weights, ignition_weights,
                                       MARKET_CONFIRMATION_WEIGHTS, EXHAUSTION_WEIGHTS,
                                       .65, .45, 80, 72, "same-source-v1")).encode()).hexdigest()[:16]
    for _, g in metrics.groupby(["kind", "history_source"], sort=False):
        x = g.copy()
        x["comparison_peer_count"] = len(x)
        x["model_version"] = MODEL_VERSION
        x["config_hash"] = config_hash
        if "breadth" not in x:
            x["breadth"] = np.nan
        if "history_source" not in x:
            x["history_source"] = "来源未标记"
        if "history_days" not in x:
            x["history_days"] = 0
        for days in (5, 10, 20):
            ret_col = f"ret_{days}d"
            if ret_col in x:
                x[f"rs_{days}d"] = pd.to_numeric(x[ret_col], errors="coerce") - pd.to_numeric(x[ret_col], errors="coerce").median()
        for window in (1, 5, 10):
            x[f"flow_{window}d_signal"] = _source_adjusted_flow_rank(x, window)
        x["flow_acceleration_signal"] = _source_adjusted_acceleration_rank(x)
        main = _grouped_score(x, main_groups, {"flow_5d_signal", "flow_10d_signal"})
        x["mainline_raw_score"], x["mainline_coverage"], x["mainline_score"] = main["raw_score"], main["coverage"], main["rank_score"]
        confirmation = _weighted_score(x, candidate_weights, {"flow_1d_signal", "flow_acceleration_signal"})
        x["confirmation_raw_score"], x["confirmation_coverage"], x["confirmation_score"] = confirmation["raw_score"], confirmation["coverage"], confirmation["rank_score"]
        for col in main.filter(like="_contribution"):
            x[f"mainline_{col}"] = main[col]
        crowding = ((x["distance_ma20"] - 12).clip(lower=0) * 0.7 + (x["ret_10d"] - 18).clip(lower=0) * 0.5).clip(upper=18)
        x["crowding_penalty"] = crowding.fillna(0)
        x["confirmation_score"] = (x["confirmation_score"] - x["crowding_penalty"]).clip(0, 100)
        x["candidate_score"] = x["confirmation_score"]
        ignition = _weighted_score(x, ignition_weights, {"flow_acceleration_signal"})
        x["ignition_raw_score"], x["ignition_coverage"], x["ignition_score"] = ignition["raw_score"], ignition["coverage"], ignition["rank_score"]
        early_crowding = ((x["ret_5d"] - 8).clip(lower=0) * .9 + (x["ret_10d"] - 15).clip(lower=0) * .45).clip(upper=22)
        x["ignition_score"] = (x["ignition_score"] - early_crowding.fillna(0)).clip(0, 100)
        x["ignition_history_coverage"] = _ignition_history_coverage(x)

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

        breadth = pd.to_numeric(x.get("breadth", pd.Series(np.nan, index=x.index)), errors="coerce")
        synthetic = x["history_source"].astype(str).str.contains("合成")
        main_ok = ((x["slope_5d"] > 0) & (x["ret_10d"] > 0)
                   & (breadth.isna() | (breadth >= .45)) & (x["mainline_coverage"] >= .65)
                   & ~synthetic & (x["comparison_peer_count"] >= 3))
        x["mainline_gate_passed"] = main_ok
        x["mainline_blockers"] = ["；".join(reason for reason, failed in (
            ("5日趋势未上行", row.slope_5d <= 0), ("10日绝对收益未转正", row.ret_10d <= 0),
            ("上涨广度不足", pd.notna(row.breadth) and row.breadth < .45),
            ("评分覆盖率不足", row.mainline_coverage < .65),
            ("合成指数不可直接确认", "合成" in str(row.history_source)),
            ("同源可比板块不足3个", row.comparison_peer_count < 3),
            ("确认分不足80", row.mainline_score < 80),
        ) if failed) for row in x.itertuples()]
        x["structure_status"] = np.where(breadth.notna(), "广度已验证", "成分结构未验证")
        candidate_ok = (x["slope_3d"] > 0) & (x["acceleration"] > 0) & (x["ret_10d"] < 18)
        x["absolute_strength_state"] = np.select(
            [(x["ret_5d"] > 0) & (x["slope_5d"] > 0),
             (x["ret_5d"] <= 0) & (x["rs_5d"] > 0)],
            ["绝对上行", "防御强势"], default="尚未上行")
        x["status"] = "普通"
        x.loc[(x["confirmation_score"] >= 65) & candidate_ok, "status"] = "值得关注"
        x.loc[(x["confirmation_score"] >= 75) & candidate_ok, "status"] = "潜在启动"
        x.loc[(x["mainline_score"] >= 68) & main_ok, "status"] = "主线观察"
        x.loc[(x["mainline_score"] >= 80) & main_ok, "status"] = "主线核心"

        # Seed/Ignition 是轨迹型标签；至少 40% 的轨迹字段可用才允许进入。
        trajectory_ok = (x["ignition_history_coverage"] >= .40) | (x["history_days"] >= 20)
        x["lifecycle"] = "Dormant"
        x.loc[(x["mainline_score"] >= 62) & main_ok, "lifecycle"] = "Diffusion"
        x.loc[(x["ignition_score"] >= 60) & (x["ret_5d"] < 8) & trajectory_ok, "lifecycle"] = "Seed"
        x.loc[(x["ignition_score"] >= 72) & (x["ret_5d"] < 8) & trajectory_ok, "lifecycle"] = "Ignition"
        x.loc[(x["mainline_score"] >= 80) & main_ok, "lifecycle"] = "Mainline"
        x.loc[(x["crowding_penalty"] >= 8) & (x["mainline_score"] >= 65), "lifecycle"] = "Crowded"
        x.loc[(x["mainline_score"] >= 60) & (x["slope_3d"] < 0) & (x["acceleration"] < 0), "lifecycle"] = "Decay"
        historical_jump = pd.to_numeric(x.get("historical_rank_change_3d", pd.Series(np.nan, index=x.index)), errors="coerce")
        x["miss_diagnosis"] = np.select(
            [(x["ignition_score"] >= 70) & ~x["lifecycle"].eq("Mainline"),
             historical_jump >= .25,
             (x["amount_ratio_5_20"] >= 1.3) & (x["ret_5d"] < 3),
             x["mainline_coverage"] < .65],
            ["高火种未确认", "排名快速上升", "量能领先", "数据不足"], default="")
        frames.append(x)
    return pd.concat(frames, ignore_index=True).sort_values("mainline_score", ascending=False)
