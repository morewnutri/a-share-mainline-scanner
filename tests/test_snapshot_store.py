from datetime import datetime

import pandas as pd

from mainline_scanner.backtest import evaluate_snapshots
from mainline_scanner.snapshot_store import SnapshotStore


def frame(a_score, b_score, a_breadth=.4, b_breadth=.6):
    return pd.DataFrame([
        {"kind": "industry", "code": "A", "name": "甲", "mainline_score": 60, "confirmation_score": 55,
         "candidate_score": 55, "ignition_score": a_score, "breadth": a_breadth, "amount_share": .1,
         "ret_1d": 1, "ret_5d": 2},
        {"kind": "industry", "code": "B", "name": "乙", "mainline_score": 65, "confirmation_score": 60,
         "candidate_score": 60, "ignition_score": b_score, "breadth": b_breadth, "amount_share": .2,
         "ret_1d": 0, "ret_5d": 3},
    ])


def test_snapshot_enrichment_calculates_positive_rank_velocity(tmp_path):
    store = SnapshotStore(tmp_path)
    prior = frame(40, 80).assign(as_of=pd.Timestamp("2026-08-28"))
    store.save(prior, datetime(2026, 8, 28, 15, 1))
    current = frame(85, 50, a_breadth=.7)
    current["as_of"] = pd.Timestamp("2026-08-31")
    current.loc[current["code"] == "A", ["confirmation_score", "candidate_score"]] = 70
    enriched = store.enrich(current, datetime(2026, 8, 31, 15, 30)).set_index("code")
    assert enriched.loc["A", "confirmation_score_rank_velocity_1d"] == 1
    assert abs(enriched.loc["A", "breadth_delta_1d"] - .3) < 1e-12
    assert enriched.loc["A", "snapshot_history_coverage"] > 0


def test_snapshot_backtest_reports_mainline_conversion(tmp_path):
    store = SnapshotStore(tmp_path)
    first = frame(80, 40)
    first["as_of"] = pd.Timestamp("2026-08-28")
    second = frame(75, 40)
    second["as_of"] = pd.Timestamp("2026-08-31")
    second.loc[second["code"] == "A", "mainline_score"] = 85
    second["lifecycle"] = ["Mainline", "Dormant"]
    store.save(first, datetime(2026, 8, 28, 15, 1))
    store.save(second, datetime(2026, 8, 31, 15, 1))
    detail, summary = evaluate_snapshots(tmp_path, top_k=1, horizon=1)
    assert detail.iloc[0]["became_mainline"]
    assert detail.iloc[0]["lead_time_sessions"] == 1
    assert summary.iloc[0]["precision_at_1"] == 1


def test_numeric_board_codes_survive_snapshot_reload_and_merge(tmp_path):
    store = SnapshotStore(tmp_path)
    first = frame(40, 80)
    first["code"] = ["881001", "001234"]
    first["as_of"] = pd.Timestamp("2026-09-30")
    store.save(first, datetime(2026, 10, 5, 15, 1))
    reloaded = store._read(store.list()[0])
    assert reloaded["code"].tolist() == ["881001", "001234"]
    enriched = store.enrich(first, datetime(2026, 10, 6, 15, 1))
    assert "breadth_delta_1d" not in enriched
