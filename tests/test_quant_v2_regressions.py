from datetime import datetime
from datetime import timedelta
import json

import numpy as np
import pandas as pd

from mainline_scanner.analysis import _weighted_score
from mainline_scanner.backtest import evaluate_snapshots, replay_daily_histories
from mainline_scanner.snapshot_store import SnapshotStore
from mainline_scanner.stock_structure import calculate_structure_metrics
from mainline_scanner.trading_calendar import expected_close_session
from mainline_scanner.data import EastmoneyAkshareProvider


def test_missing_factor_is_not_neutral():
    frame = pd.DataFrame({"price": [10, 5, 0], "breadth": [1.0, np.nan, 0.0]})
    scored = _weighted_score(frame, {"price": .5, "breadth": .5})
    assert scored.loc[1, "coverage"] == .5
    assert scored.loc[1, "rank_score"] == 50 + (scored.loc[1, "raw_score"] - 50) * .5
    assert pd.isna(scored.loc[1, "breadth_contribution"])


def test_same_trade_date_does_not_create_fake_velocity(tmp_path):
    store = SnapshotStore(tmp_path)
    row = pd.DataFrame([{"kind": "industry", "code": "001234", "name": "测试",
                         "as_of": pd.Timestamp("2026-09-30"), "mainline_score": 60,
                         "confirmation_score": 55, "breadth": .6}])
    store.save(row, datetime(2026, 10, 5, 16))
    changed = row.assign(mainline_score=90)
    out = store.enrich(changed, datetime(2026, 10, 6, 16))
    assert "mainline_score_delta_1d" not in out


def test_holiday_recheck_not_selected_as_daily_close(tmp_path):
    store = SnapshotStore(tmp_path)
    row = pd.DataFrame([{"kind": "industry", "code": "A", "name": "甲",
                         "as_of": pd.Timestamp("2026-09-30"), "run_mode": "historical_replay",
                         "mainline_score": 75}])
    store.save(row, datetime(2026, 10, 5, 16))
    assert store.daily_closes() == []


def test_future_window_censored_and_lifecycle_required(tmp_path):
    store = SnapshotStore(tmp_path)
    for day, score, lifecycle in [("2026-09-28", 60, "Seed"),
                                  ("2026-09-29", 82, "Dormant"),
                                  ("2026-09-30", 84, "Mainline")]:
        frame = pd.DataFrame([{"kind": "industry", "code": "A", "name": "甲",
                               "as_of": pd.Timestamp(day), "ignition_score": 75,
                               "mainline_score": score, "lifecycle": lifecycle, "ret_1d": 1}])
        store.save(frame, datetime.fromisoformat(day + "T16:00:00"))
    detail, summary = evaluate_snapshots(tmp_path, top_k=1, horizon=2)
    first = detail.loc[detail["signal_date"] == pd.Timestamp("2026-09-28")].iloc[0]
    assert first["outcome"] == "success" and first["lead_time_sessions"] == 2
    assert summary.loc[0, "censored_signals"] == 2


def test_holiday_session_and_stock_coverage():
    assert expected_close_session(pd.Timestamp("2026-10-07"), datetime(2026, 10, 7, 16)) == pd.Timestamp("2026-09-30")
    day = pd.Timestamp("2026-09-30")
    members = pd.DataFrame({"代码": [f"00000{i}" for i in range(10)], "fetched_at": [day] * 10})
    quotes = pd.DataFrame({"代码": [f"00000{i}" for i in range(5)], "涨跌幅": [1, 2, 3, 4, 5], "market_date": [day] * 5})
    result = calculate_structure_metrics(members, quotes, pd.DataFrame(), day)
    assert result["structure_status"] == "成分股覆盖不足"
    assert pd.isna(result["structure_score"])


def test_cache_requires_requested_interval_and_expected_session(tmp_path):
    provider = object.__new__(EastmoneyAkshareProvider)
    provider.refresh = False
    provider.ttl = timedelta(hours=24)
    path = tmp_path / "history.csv"
    pd.DataFrame({"日期": ["2026-09-29", "2026-09-30"], "收盘": [100, 101]}).to_csv(path, index=False)
    meta = {"requested_start": "20260901", "expected_trade_date": "2026-09-30",
            "last_actual_date": "2026-09-30", "adjustment": "none", "source": "东方财富"}
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    assert provider._history_cache_valid(path, "20260901", "20261007")
    assert not provider._history_cache_valid(path, "20260801", "20261007")
    assert not provider._history_cache_valid(path, "20260901", "20261008")


def test_historical_replay_does_not_copy_current_breadth(tmp_path):
    dates = pd.bdate_range("2026-08-03", periods=25)
    boards = pd.DataFrame([{"kind": "industry", "code": code, "name": code, "breadth": 1.0}
                           for code in ("A", "B", "C")])
    histories = {("industry", code): pd.DataFrame({
        "date": dates, "close": 100 * np.exp(slope * np.arange(25)),
        "amount": np.full(25, 100.0), "turnover": np.ones(25),
    }) for code, slope in (("A", .01), ("B", .005), ("C", -.002))}
    paths = replay_daily_histories(boards, histories, pd.DataFrame(), tmp_path, horizon=2)
    store = SnapshotStore(paths["replay_snapshots"])
    frames = store.daily_closes()
    assert frames
    assert frames[0]["breadth"].isna().all()
