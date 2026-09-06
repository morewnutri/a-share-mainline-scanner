import math
from datetime import date

import pandas as pd

from mainline_scanner.valuation import (
    _geomean_ratio,
    annualization_factor,
    completed_report_candidates,
    report_period_triplet,
    select_report_periods,
    valuation_label,
    ValuationEngine,
)


def test_labels():
    assert valuation_label(-0.30) == "低估"
    assert valuation_label(-0.15) == "合理偏低"
    assert valuation_label(0.00) == "合理"
    assert valuation_label(0.20) == "偏高"
    assert valuation_label(0.50) == "高估"
    assert valuation_label(0.90) == "显著高估"


def test_geomean():
    v = _geomean_ratio([(1.2, 0.8), (0.8, 0.2)])
    assert math.isfinite(v)
    assert 0.8 < v < 1.2


def test_report_helpers():
    assert annualization_factor("20260331") == 4.0
    assert annualization_factor("20260630") == 2.0
    assert annualization_factor("20260930") == 4.0 / 3.0
    assert report_period_triplet("20260630") == ("20260630", "20250630", "20251231")
    assert completed_report_candidates(date(2026, 9, 6))[0] == "20260630"
    assert completed_report_candidates(date(2026, 11, 1))[0] == "20260930"


class FakeProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def spot(self):
        return pd.DataFrame({"code": [f"{i:06d}" for i in range(100)]})

    def performance(self, report_date):
        n = self.mapping[report_date]
        return pd.DataFrame({"code": [f"{i:06d}" for i in range(n)]})


def test_auto_report_selection_falls_back_on_incomplete_latest():
    provider = FakeProvider(
        {
            # H1 incomplete -> must fall back to Q1.
            "20260630": 70,
            "20250630": 70,
            "20251231": 90,
            "20260331": 90,
            "20250331": 90,
            # Q1 TTM base is still 2025 annual.
            # 90/90 coverage against current reporters.
        }
    )
    cfg = {
        "report_date": "auto",
        "prior_report_date": "auto",
        "annual_report_date": "auto",
        "report_policy": {"min_financial_coverage": 0.80},
    }
    sel = select_report_periods(provider, cfg, today=date(2026, 9, 6))
    assert sel.current == "20260331"
    assert sel.prior == "20250331"
    assert sel.annual == "20251231"
    assert sel.current_coverage == 0.90


def test_sustainable_growth_caps_profit_spike():
    m = {"revenue_growth": 0.2634, "profit_growth": 2.0455}
    c = {"profit_growth_premium_cap_pct": 20}
    g = ValuationEngine._normalized_growth_pct(m, c)
    assert round(g, 2) == 39.34
