import numpy as np
import pandas as pd
import pytest

from mainline_scanner.analysis import build_metric_table, score_boards
from mainline_scanner.market_metrics import enrich_market_history, load_market
from mainline_scanner.quant_scores import add_quant_radar
from mainline_scanner.manual_checklist import attach_manual_checklist


def _history(slope=.01, n=65):
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = 100 * np.exp(slope * np.arange(n))
    return pd.DataFrame({"date": dates, "close": close, "amount": np.full(n, 100.0)})


def _boards():
    boards = pd.DataFrame([{"kind": "concept", "code": code, "name": code, "breadth": .6}
                           for code in ("A", "B", "C")])
    histories = {("concept", code): _history(slope) for code, slope in (("A", .01), ("B", .005), ("C", -.002))}
    return boards, histories


def test_market_share_uses_independent_market_total(tmp_path):
    boards, histories = _boards()
    dates = next(iter(histories.values()))["date"]
    path = tmp_path / "market.csv"
    pd.DataFrame({"date": dates, "market_amount": np.full(len(dates), 1000),
                  "benchmark_close": 100 * np.exp(.005 * np.arange(len(dates)))}).to_csv(path, index=False)
    metrics = enrich_market_history(build_metric_table(boards, histories, pd.DataFrame()), histories, load_market(path))
    assert metrics["turnover_share"].eq(.1).all()
    assert metrics["top_rank_valid_days_10"].eq(10).all()
    assert metrics["top_rank_days_10"].notna().all()
    assert metrics.set_index("code").loc["A", "rs_market_5d"] > 0


def test_subjective_fields_cannot_change_quantitative_scores():
    boards, histories = _boards()
    metrics = enrich_market_history(build_metric_table(boards, histories, pd.DataFrame()), histories, pd.DataFrame())
    base = attach_manual_checklist(add_quant_radar(score_boards(metrics))).set_index("code")
    changed = metrics.assign(policy_level=1.0, industry_orders=1.0, catalyst_frequency=1.0)
    altered = attach_manual_checklist(add_quant_radar(score_boards(changed))).set_index("code")
    for col in ("mainline_score", "ignition_score", "market_confirmation_score", "lifecycle"):
        assert base[col].equals(altered[col])
    assert "未检查" in altered.loc["A", "人工核查提醒"]
    assert "potential_score" not in altered


def test_insufficient_daily_rank_coverage_stays_missing():
    boards, histories = _boards()
    histories[("concept", "C")] = histories[("concept", "C")].tail(5)
    metrics = build_metric_table(boards.iloc[:2], {k: v for k, v in histories.items() if k[1] != "C"}, pd.DataFrame())
    metrics = pd.concat([metrics, pd.DataFrame([{"kind": "concept", "code": "C", "name": "C",
                                                "as_of": histories[("concept", "C")]["date"].iloc[-1]}])], ignore_index=True)
    enriched = enrich_market_history(metrics, histories, pd.DataFrame()).set_index("code")
    assert enriched.loc["C", "top_rank_valid_days_10"] < 8
    assert pd.isna(enriched.loc["C", "top_rank_days_10"])


def test_market_reconciliation_rejects_bad_amount(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame([{"date": "2026-01-01", "market_amount": 1000, "stock_amount_sum": 900}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="对账误差"):
        load_market(path)
