import numpy as np
import pandas as pd
import pytest

from mainline_scanner.research import (
    add_research_scores, add_switch_signals, enrich_market_history,
    load_market, load_observations,
)
from mainline_scanner.backtest import evaluate_future_dominance
from mainline_scanner.snapshot_store import SnapshotStore


def _history(amount=100.0, slope=.01, n=65):
    dates = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"date": dates, "close": 100 * np.exp(slope * np.arange(n)),
                         "amount": np.full(n, amount)})


def _metrics():
    date = _history()["date"].iloc[-1]
    return pd.DataFrame([
        {"kind": "concept", "code": "A", "name": "甲", "as_of": date,
         "breadth": .7, "positive_days_10": .7, "last_amount": 100,
         "rs_5d": 3, "rs_10d": 6, "ret_20d": 10, "ret_5d": 5},
        {"kind": "concept", "code": "B", "name": "乙", "as_of": date,
         "breadth": .5, "positive_days_10": .5, "last_amount": 100,
         "rs_5d": 2, "rs_10d": 4, "ret_20d": 8, "ret_5d": 3},
        {"kind": "concept", "code": "C", "name": "丙", "as_of": date,
         "breadth": .3, "positive_days_10": .3, "last_amount": 100,
         "rs_5d": 1, "rs_10d": 2, "ret_20d": 5, "ret_5d": 1},
    ])


def test_point_in_time_sparse_evidence_and_missing_potential(tmp_path):
    today = _metrics()["as_of"].iloc[0]
    path = tmp_path / "signals.csv"
    pd.DataFrame([
        {"kind": "concept", "code": "A", "published_at": today - pd.Timedelta(days=2),
         "available_at": today - pd.Timedelta(days=1), "eps_revision_fy1": .8},
        {"kind": "concept", "code": "A", "published_at": today,
         "available_at": today + pd.Timedelta(hours=10), "industry_orders": .9},
        {"kind": "concept", "code": "A", "published_at": today + pd.Timedelta(days=1),
         "available_at": today + pd.Timedelta(days=1), "eps_revision_fy1": .1},
    ]).to_csv(path, index=False)
    scored = add_research_scores(_metrics(), load_observations(path), today + pd.Timedelta(hours=9)).set_index("code")
    assert scored.loc["A", "eps_revision_fy1"] == .8
    assert pd.isna(scored.loc["A", "industry_orders"])
    assert pd.isna(scored.loc["B", "potential_score"])
    assert scored.loc["B", "potential_state"] == "基本面资料待补"
    assert 50 < scored.loc["A", "potential_rank_score"] < scored.loc["A", "potential_score"]
    assert len(scored) == 3


def test_market_share_uses_independent_market_total_even_for_overlapping_concepts(tmp_path):
    metrics = _metrics()
    dates = _history()["date"]
    market_path = tmp_path / "market.csv"
    pd.DataFrame({"date": dates, "market_amount": np.full(len(dates), 1000),
                  "benchmark_close": 100 * np.exp(.005 * np.arange(len(dates)))}).to_csv(market_path, index=False)
    histories = {("concept", code): _history(amount=amount) for code, amount in [("A", 100), ("B", 100), ("C", 50)]}
    enriched = enrich_market_history(metrics, histories, load_market(market_path)).set_index("code")
    assert enriched.loc["A", "turnover_share"] == pytest.approx(.1)
    assert enriched.loc["B", "turnover_share"] == pytest.approx(.1)
    assert enriched.loc["A", "rs_market_5d"] > 0
    assert enriched["top_rank_days_10"].notna().all()


def test_risk_and_switch_remain_separate_from_potential():
    metrics = _metrics()
    metrics["leader_divergence"] = [.9, np.nan, np.nan]
    metrics["breakout_failure_rate"] = [.8, np.nan, np.nan]
    metrics["market_confirmation_score_prev_1d"] = [70, 60, 40]
    metrics["market_confirmation_score_delta_1d"] = [-3, 5, 1]
    metrics["turnover_share_delta_1d"] = [-.01, .02, 0]
    scored = add_switch_signals(add_research_scores(metrics)).set_index("code")
    assert scored.loc["A", "exhaustion_score"] > 70
    assert pd.isna(scored.loc["B", "exhaustion_score"])
    assert scored.loc["B", "switch_score"] > scored.loc["C", "switch_score"]
    assert scored.loc["B", "switch_from"] == "甲"


def test_invalid_evidence_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"kind": "industry", "code": "A", "available_at": "2026-01-01",
                   "policy_level": 1.2}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="0..1"):
        load_observations(path)


def test_market_audit_reconciles_amount_and_counts_suspensions(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame([{"date": "2026-01-01", "market_amount": 1000,
                   "stock_amount_sum": 990, "expected_stock_count": 100,
                   "valid_quote_count": 93, "suspended_count": 5,
                   "theme_mapped_count": 90, "consensus_covered_count": 40,
                   "financial_report_due_count": 80,
                   "financial_report_available_count": 76}]).to_csv(path, index=False)
    audited = load_market(path)
    assert audited.loc[0, "price_coverage"] == pytest.approx(.98)
    assert audited.loc[0, "theme_coverage"] == pytest.approx(.9)
    assert audited.loc[0, "financial_coverage"] == pytest.approx(.95)
    assert audited.loc[0, "consensus_coverage"] == pytest.approx(.4)
    assert audited.loc[0, "reconciliation_error"] == pytest.approx(.01)
    with pytest.raises(ValueError, match="对账误差"):
        load_market(path, reconciliation_tolerance=.005)


def test_future_dominance_uses_only_a_complete_forward_window(tmp_path):
    store = SnapshotStore(tmp_path)
    for day in range(22):
        frame = pd.DataFrame([
            {"kind": "industry", "code": "A", "name": "甲", "potential_score": 85,
             "ret_1d": 2, "turnover_share": .20, "breadth": .8, "last_close": 100 + 2 * day},
            {"kind": "industry", "code": "B", "name": "乙", "potential_score": 45,
             "ret_1d": 0, "turnover_share": .10, "breadth": .5, "last_close": 100},
            {"kind": "industry", "code": "C", "name": "丙", "potential_score": 30,
             "ret_1d": -1, "turnover_share": .05, "breadth": .2, "last_close": 100 - day},
        ])
        store.save(frame, pd.Timestamp("2026-01-01") + pd.Timedelta(days=day))
    detail, summary = evaluate_future_dominance(tmp_path, horizon=20, top_k=1)
    assert detail["signal_date"].nunique() == 2
    assert detail.loc[detail["code"] == "A", "future_mainline_label"].all()
    assert summary.iloc[0]["potential_precision_at_1"] == 1
