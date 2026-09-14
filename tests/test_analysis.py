import numpy as np
import pandas as pd

from mainline_scanner.analysis import build_metric_table, calculate_board_metrics, score_boards


def make_history(slope=.01, acceleration=0.0, n=45):
    x = np.arange(n, dtype=float)
    log_close = 5 + slope * x + acceleration * np.maximum(x - (n - 8), 0) ** 2
    close = np.exp(log_close)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="B"),
        "close": close,
        "amount": np.linspace(1e9, 2e9, n),
        "turnover": np.linspace(1, 2, n),
    })


def test_metrics_detect_positive_slope_and_acceleration():
    m = calculate_board_metrics(make_history(.003, .0008))
    assert m["slope_3d"] > m["slope_10d"] > 0
    assert m["acceleration"] > 0
    assert m["amount_ratio_5_20"] > 1


def test_scoring_prefers_stronger_board():
    boards = pd.DataFrame([
        {"kind": "industry", "code": "A", "name": "强", "breadth": .8},
        {"kind": "industry", "code": "B", "name": "中", "breadth": .6},
        {"kind": "industry", "code": "C", "name": "弱", "breadth": .2},
    ])
    histories = {
        ("industry", "A"): make_history(.006, .0003),
        ("industry", "B"): make_history(.002, 0),
        ("industry", "C"): make_history(-.003, 0),
    }
    flows = pd.DataFrame({
        "kind": ["industry"] * 3, "name": ["强", "中", "弱"],
        "flow_1d_pct": [10, 2, -5], "flow_5d_pct": [8, 1, -4], "flow_10d_pct": [6, 0, -3],
    })
    scored = score_boards(build_metric_table(boards, histories, flows)).set_index("name")
    assert scored.loc["强", "mainline_score"] > scored.loc["中", "mainline_score"] > scored.loc["弱", "mainline_score"]


def test_cmf_proxy_fills_missing_fund_flow_and_is_labeled():
    history = make_history(.002, 0)
    history["high"] = history["close"] * 1.01
    history["low"] = history["close"] * 0.98
    boards = pd.DataFrame([{"kind": "industry", "code": "A", "name": "测试", "breadth": .6}])
    metrics = build_metric_table(boards, {("industry", "A"): history}, pd.DataFrame())
    assert pd.notna(metrics.loc[0, "flow_5d_pct"])
    assert metrics.loc[0, "flow_5d_source"] == "量价代理CMF"
    assert metrics.loc[0, "flow_5d_confidence"] == .55


def test_flow_acceleration_compares_only_same_source_and_daily_intensity():
    boards = pd.DataFrame([
        {"kind": "industry", "code": "A", "name": "真实", "breadth": .6},
        {"kind": "industry", "code": "B", "name": "代理", "breadth": .6},
    ])
    histories = {("industry", "A"): make_history(), ("industry", "B"): make_history()}
    for history in histories.values():
        history["high"] = history["close"] * 1.01
        history["low"] = history["close"] * .98
    flows = pd.DataFrame({
        "kind": ["industry"], "name": ["真实"],
        "flow_1d_pct": [6.0], "flow_5d_pct": [10.0], "flow_10d_pct": [12.0],
    })
    metrics = build_metric_table(boards, histories, flows).set_index("name")
    assert metrics.loc["真实", "flow_acceleration"] == 4.0
    assert metrics.loc["真实", "flow_acceleration_source"].startswith("东方财富同口径")
    assert metrics.loc["代理", "flow_acceleration_source"] == "CMF同口径变化"


def test_sideways_seed_finds_low_tight_box_but_not_uptrend():
    dates = pd.date_range("2026-01-01", periods=65, freq="B")
    early = np.linspace(120, 92, 25)
    box = 92 + 1.2 * np.sin(np.linspace(0, 6 * np.pi, 40))
    low_box = pd.DataFrame({
        "date": dates, "close": np.r_[early, box],
        "high": np.r_[early, box] * 1.01, "low": np.r_[early, box] * .99,
        "amount": np.full(65, 1e9), "turnover": np.full(65, 1.0),
    })
    uptrend = make_history(.008, 0, n=65)
    boards = pd.DataFrame([
        {"kind": "industry", "code": "BOX", "name": "低位箱体", "breadth": .5},
        {"kind": "industry", "code": "UP", "name": "上涨趋势", "breadth": .7},
    ])
    scored = score_boards(build_metric_table(
        boards, {("industry", "BOX"): low_box, ("industry", "UP"): uptrend}, pd.DataFrame()
    )).set_index("name")
    assert scored.loc["低位箱体", "sideways_seed_status"] in {"横盘火种", "横盘观察"}
    assert scored.loc["低位箱体", "sideways_seed_score"] > scored.loc["上涨趋势", "sideways_seed_score"]
    assert scored.loc["上涨趋势", "sideways_seed_status"] == "非横盘"
