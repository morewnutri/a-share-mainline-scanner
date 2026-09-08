from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


TTM_CONFIDENCE = {
    "EXACT_TTM": 1.00,
    "ANNUALIZED_Q3": 0.70,
    "ANNUALIZED_H1": 0.55,
    "ANNUALIZED_Q1": 0.35,
    "QUOTE_PROVIDER_PE": 0.30,
    "MISSING": 0.00,
}


@dataclass(frozen=True)
class HistoryStats:
    status: str = "MISSING"
    points: int = 0
    effective_points: float = 0.0
    last_date: str = ""
    source: str = "NONE"
    confidence: float = 0.0
    median: float = float("nan")
    q10: float = float("nan")
    q25: float = float("nan")
    q50: float = float("nan")
    q75: float = float("nan")
    q90: float = float("nan")
    mad: float = float("nan")


def ttm_method(report_date: str, exact: bool, current_available: bool) -> str:
    if exact:
        return "EXACT_TTM"
    if not current_available:
        return "MISSING"
    suffix = str(report_date)[-4:]
    return {"0331": "ANNUALIZED_Q1", "0630": "ANNUALIZED_H1", "0930": "ANNUALIZED_Q3", "1231": "EXACT_TTM"}.get(
        suffix, "MISSING"
    )


def confidence_label(score: float) -> str:
    if not np.isfinite(score) or score < 0.40:
        return "NO_VALUATION"
    if score < 0.60:
        return "LOW"
    if score < 0.80:
        return "MEDIUM"
    return "HIGH"


def effective_sample_size(values: Iterable[float]) -> float:
    s = pd.Series(list(values), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 3:
        return float(n)
    rho = float(s.autocorr(lag=1))
    if not np.isfinite(rho):
        return float(n)
    rho = min(max(rho, -0.95), 0.95)
    return float(min(n, max(1.0, n * (1.0 - rho) / (1.0 + rho))))


def prepare_history(frame: pd.DataFrame, date_candidates: Iterable[str] = ("trade_date", "date", "日期")) -> pd.DataFrame:
    """Sort observations and remove repeated calendar/weekend snapshots by actual trade date."""
    out = frame.copy()
    date_col = next((c for c in date_candidates if c in out.columns), None)
    if date_col is None:
        out["trade_date"] = pd.NaT
        return out.reset_index(drop=True)
    out["trade_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
    return out.drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def summarize_values(
    values: Iterable[float],
    *,
    dates: Iterable[Any] | None = None,
    source: str,
    minimum: int = 30,
) -> HistoryStats:
    raw = pd.DataFrame({"value": list(values)})
    if dates is not None:
        raw["trade_date"] = list(dates)
        raw = prepare_history(raw)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw[np.isfinite(raw["value"])]
    points = len(raw)
    last_date = ""
    if "trade_date" in raw and raw["trade_date"].notna().any():
        last_date = pd.Timestamp(raw["trade_date"].max()).date().isoformat()
    if not points:
        return HistoryStats(status="MISSING", source=source)
    neff = effective_sample_size(raw["value"])
    confidence = min(1.0, neff / max(float(minimum), 1.0))
    required_effective = max(5.0, float(minimum) * 0.20)
    if points < minimum:
        status = "INSUFFICIENT_POINTS"
    elif neff < required_effective:
        status = "INSUFFICIENT_EFFECTIVE_POINTS"
    else:
        status = "OK"
    v = raw["value"]
    median = float(v.median())
    mad = float((v - median).abs().median())
    return HistoryStats(
        status=status,
        points=points,
        effective_points=neff,
        last_date=last_date,
        source=source,
        confidence=confidence,
        median=median,
        q10=float(v.quantile(0.10)),
        q25=float(v.quantile(0.25)),
        q50=float(v.quantile(0.50)),
        q75=float(v.quantile(0.75)),
        q90=float(v.quantile(0.90)),
        mad=mad,
    )


def shrink_log_value(model_value: float, history: HistoryStats, kappa: float = 60.0) -> tuple[float, float]:
    """Empirical-Bayes blend in log space; autocorrelated days count as fewer observations."""
    model_ok = np.isfinite(model_value) and model_value > 0
    history_ok = history.status == "OK" and np.isfinite(history.median) and history.median > 0
    if model_ok and history_ok:
        w_hist = history.effective_points / (history.effective_points + max(kappa, 1e-9))
        w_hist *= history.confidence
        value = math.exp((1.0 - w_hist) * math.log(model_value) + w_hist * math.log(history.median))
        return float(value), float(w_hist)
    if model_ok:
        return float(model_value), 0.0
    if history_ok:
        return float(history.median), 1.0
    return float("nan"), 0.0


def weighted_geometric_price(items: Iterable[tuple[float, float, str]]) -> tuple[float, float, list[str]]:
    valid = [(float(v), float(w), str(name)) for v, w, name in items if np.isfinite(v) and v > 0 and w > 0]
    if not valid:
        return float("nan"), float("nan"), []
    total = sum(w for _, w, _ in valid)
    logs = np.array([math.log(v) for v, _, _ in valid], dtype=float)
    weights = np.array([w / total for _, w, _ in valid], dtype=float)
    center_log = float(np.sum(weights * logs))
    disagreement = float(math.sqrt(np.sum(weights * np.square(logs - center_log))))
    return float(math.exp(center_log)), disagreement, [name for _, _, name in valid]


def data_quality_score(parts: dict[str, float]) -> float:
    weights = {"ttm": 0.30, "growth": 0.15, "history": 0.15, "cashflow": 0.15, "sector": 0.10, "freshness": 0.15}
    score = 0.0
    for key, weight in weights.items():
        value = float(parts.get(key, 0.0))
        score += weight * min(1.0, max(0.0, value if np.isfinite(value) else 0.0))
    return float(min(1.0, max(0.0, score)))


def required_margin(
    base_margin: float,
    model_disagreement: float,
    data_quality: float,
    cycle_penalty: float = 0.0,
) -> float:
    disagreement = model_disagreement if np.isfinite(model_disagreement) else 0.12
    value = base_margin + 0.35 * disagreement + 0.18 * (1.0 - data_quality) + cycle_penalty
    return float(min(0.45, max(0.10, value)))


def fair_value_interval(
    center: float,
    model_disagreement: float,
    data_quality: float,
    base_uncertainty: float,
) -> tuple[float, float, float]:
    if not np.isfinite(center) or center <= 0:
        return float("nan"), float("nan"), float("nan")
    disagreement = model_disagreement if np.isfinite(model_disagreement) else base_uncertainty
    sigma = math.sqrt(disagreement**2 + base_uncertainty**2 + (0.25 * (1.0 - data_quality)) ** 2)
    return float(center * math.exp(-sigma)), float(center * math.exp(sigma)), float(sigma)


def robust_z(value: float, stats: HistoryStats) -> float:
    scale = 1.4826 * stats.mad
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 1e-12:
        return float("nan")
    return float((value - stats.median) / scale)


def trade_bands(
    fair_center: float,
    fair_low: float,
    margin: float,
    residuals: HistoryStats,
    sigma_total: float,
) -> dict[str, float | str]:
    if not np.isfinite(fair_center) or fair_center <= 0:
        return {k: float("nan") for k in ("deep_buy_price", "buy_price", "buy_exit_price", "trim_price", "exit_price", "valuation_quantile", "valuation_robust_z")} | {"trade_band_source": "NONE"}
    if residuals.status == "OK":
        q10, q25, q75, q90 = residuals.q10, residuals.q25, residuals.q75, residuals.q90
        q35 = residuals.q25 + 0.40 * (residuals.q50 - residuals.q25)
        source = "POINT_IN_TIME_RESIDUAL_QUANTILES"
    else:
        scale = max(float(sigma_total) if np.isfinite(sigma_total) else 0.20, 0.08)
        q10, q25, q75, q90 = -1.2816 * scale, -0.6745 * scale, 0.6745 * scale, 1.2816 * scale
        q35 = -0.3853 * scale
        source = "MODEL_PRIOR_BANDS"
    safety = min(fair_low, fair_center * (1.0 - margin))
    deep = min(fair_center * math.exp(q10), safety * 0.92)
    buy = min(fair_center * math.exp(q25), safety)
    trim = max(fair_center * math.exp(q75), fair_center * 1.02)
    exit_price = max(fair_center * math.exp(q90), trim * 1.02)
    buy_exit = max(buy * 1.02, fair_center * math.exp(q35))
    # Numerical monotonicity is a contract used by downstream automation.
    deep = min(deep, buy * 0.98)
    trim = max(trim, buy * 1.02)
    buy_exit = min(max(buy_exit, buy * 1.02), trim * 0.98)
    exit_price = max(exit_price, trim * 1.02)
    return {
        "deep_buy_price": float(deep),
        "buy_price": float(buy),
        "buy_exit_price": float(buy_exit),
        "trim_price": float(trim),
        "exit_price": float(exit_price),
        "trade_band_source": source,
    }


def position_policy(
    price: float,
    bands: dict[str, Any],
    stage: str,
    tradable: bool,
    gate: str,
    previous_target: float | None = None,
) -> tuple[int, str, str]:
    if not tradable or not np.isfinite(price):
        return 0, "NO_TRADE", "模型或数据门槛未通过"
    deep, buy, trim, exit_price = (float(bands[k]) for k in ("deep_buy_price", "buy_price", "trim_price", "exit_price"))
    if price <= deep:
        target, zone = 100, "深度低估"
    elif price <= buy:
        target, zone = 70, "低估"
    elif price < trim:
        target, zone = 40, "合理"
    elif price < exit_price:
        target, zone = 15, "偏贵"
    else:
        target, zone = 0, "极端高估"

    if previous_target is not None and np.isfinite(previous_target) and not gate.startswith("不通过") and stage != "Decay":
        buy_exit = float(bands.get("buy_exit_price", buy))
        if previous_target >= 70 and buy < price <= buy_exit and target < previous_target:
            return int(previous_target), "HOLD_HYSTERESIS", "已处建仓状态，价格尚未越过Q35退出阈值"

    stage = stage or "UNKNOWN"
    if gate.startswith("不通过") and target > 40:
        return 0, "NO_TRADE", f"{zone}但{gate}，防止价值陷阱"
    if zone in {"深度低估", "低估"}:
        if stage in {"Seed", "Ignition"}:
            return target, "BUILD_POSITION", f"{zone}+{stage}，允许分批建仓"
        if stage in {"Mainline", "Diffusion"}:
            return target, "ADD_OR_HOLD", f"{zone}+{stage}，加仓或持有"
        if stage == "Decay":
            return min(target, 20), "WAIT", f"{zone}但主线Decay，防价值陷阱"
        return min(target, 30), "WAIT_MAINLINE_CONFIRMATION", f"{zone}但缺少主线确认"
    if zone == "合理":
        return target, "HOLD_OR_WAIT", f"估值合理，主线阶段={stage}"
    if zone == "偏贵":
        return target, "TRIM", f"偏贵，主线阶段={stage}"
    action = "EXIT_PRIORITY" if stage == "Decay" else "AVOID_OR_EXIT"
    return 0, action, f"极端高估，主线阶段={stage}"
