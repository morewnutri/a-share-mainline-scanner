import math
from pathlib import Path

import numpy as np
import pandas as pd

from mainline_scanner.valuation import ValuationDataProvider, ValuationEngine, load_config
from mainline_scanner.valuation_policy import (
    HistoryStats,
    confidence_label,
    data_quality_score,
    prepare_history,
    position_policy,
    trade_bands,
)


def test_config_is_valid_and_referentially_complete():
    cfg = load_config(Path(__file__).parents[1] / "valuation_config.json")
    assert isinstance(cfg["sectors"], dict)
    for code, stock in cfg["stocks"].items():
        assert len(code) == 6
        assert stock["name"]
        assert stock["sector"]
        assert stock["sector"] in cfg["sectors"] or "model_override" in stock
    assert cfg["stocks"]["600863"]["name"] == "内蒙华电"


def _performance(code="000001", **values):
    defaults = {
        "name_fin": "测试股份", "revenue": 100.0, "revenue_yoy": 10.0,
        "net_profit": 10.0, "profit_yoy": 10.0, "roe_h1_pct": 5.0,
        "ocfps": 0.8, "eps": 0.5, "gross_margin_pct": 30.0,
    }
    defaults.update(values)
    return pd.DataFrame([{"code": code, **defaults}])


class TTMProvider:
    def __init__(self, frames):
        self.frames = frames

    def performance(self, report_date):
        return self.frames[report_date]


def test_exact_ttm_and_missing_prior_provenance():
    exact = TTMProvider({
        "20260630": _performance(revenue=60, net_profit=6, eps=.30, ocfps=.4),
        "20250630": _performance(revenue=50, net_profit=5, eps=.25, ocfps=.3),
        "20251231": _performance(revenue=100, net_profit=10, eps=.50, ocfps=.7),
    })
    row = ValuationDataProvider.fundamentals_ttm(exact, "20260630", "20250630", "20251231").iloc[0]
    assert row["ttm_profit"] == 11
    assert row["ttm_profit_method"] == "EXACT_TTM"
    assert row["ttm_profit_confidence"] == 1.0

    missing = TTMProvider({
        "20260630": _performance(revenue=60, net_profit=6, eps=.30, ocfps=.4),
        "20250630": _performance(revenue=np.nan, net_profit=np.nan, eps=np.nan, ocfps=np.nan),
        "20251231": _performance(revenue=100, net_profit=10, eps=.50, ocfps=.7),
    })
    row = ValuationDataProvider.fundamentals_ttm(missing, "20260630", "20250630", "20251231").iloc[0]
    assert row["ttm_profit"] == 12
    assert row["ttm_profit_method"] == "ANNUALIZED_H1"
    assert row["ttm_profit_confidence"] == 0.55


def test_q1_annualization_is_low_confidence():
    missing = TTMProvider({
        "20260331": _performance(net_profit=2),
        "20250331": _performance(net_profit=np.nan),
        "20251231": _performance(net_profit=np.nan),
    })
    row = ValuationDataProvider.fundamentals_ttm(missing, "20260331", "20250331", "20251231").iloc[0]
    assert row["ttm_profit"] == 8
    assert row["ttm_profit_method"] == "ANNUALIZED_Q1"
    assert row["ttm_profit_confidence"] == 0.35


def test_history_is_sorted_then_deduplicated_by_trade_date():
    history = pd.DataFrame({
        "trade_date": ["2026-01-03", "2026-01-01", "2026-01-03", "2026-01-02"],
        "pe": [30, 10, 31, 20],
    })
    out = prepare_history(history)
    assert out["trade_date"].is_monotonic_increasing
    assert out["pe"].tolist() == [10, 20, 31]


def test_loss_heavy_sector_disables_pe_and_reports_metric_coverages():
    engine = object.__new__(ValuationEngine)
    engine.cfg = {"valuation_policy": {"min_profitable_mcap_coverage": .70}}
    engine.master = pd.DataFrame({
        "code": ["1", "2"], "market_cap": [30.0, 70.0], "ttm_profit": [3.0, -1.0],
        "ttm_revenue": [10.0, 20.0], "pb": [2.0, np.nan],
        "revenue_cur": [12.0, 22.0], "revenue_pri": [10.0, 20.0],
        "net_profit_cur": [3.0, -1.0], "net_profit_pri": [2.0, -2.0],
        "gross_margin_pct_cur": [30.0, 20.0],
    })
    result = engine._aggregate(pd.DataFrame({"code": ["1", "2"]}))
    assert math.isnan(result["pe"])
    assert result["positive_profit_pe"] == 10
    assert result["aggregate_pe"] == 50
    assert result["profitable_mcap_coverage"] == .30
    assert result["loss_mcap_share"] == .70
    assert result["pb_mcap_coverage"] == .30
    assert result["ps_mcap_coverage"] == 1.0


def test_missing_data_never_improves_confidence():
    complete = data_quality_score({"ttm": 1, "growth": 1, "history": 1, "cashflow": 1, "sector": 1, "freshness": 1})
    missing = data_quality_score({"ttm": .35, "growth": 1, "history": 0, "cashflow": 0, "sector": 1, "freshness": 1})
    assert missing < complete
    assert confidence_label(missing) != "HIGH"


def test_trade_band_monotonicity():
    bands = trade_bands(100, 80, .25, HistoryStats(status="MISSING"), .20)
    assert bands["deep_buy_price"] < bands["buy_price"] < bands["trim_price"] < bands["exit_price"]
    assert bands["buy_price"] <= 75


def test_position_hysteresis_uses_separate_q35_exit():
    bands = trade_bands(100, 80, .20, HistoryStats(status="MISSING"), .20)
    price = (bands["buy_price"] + bands["buy_exit_price"]) / 2
    target, action, _ = position_policy(price, bands, "Mainline", True, "通过", previous_target=70)
    assert target == 70
    assert action == "HOLD_HYSTERESIS"


def test_akshare_performance_contract_normalizes_expected_fields():
    provider = object.__new__(ValuationDataProvider)
    raw = pd.DataFrame([{
        "股票代码": "000001", "股票简称": "平安银行", "营业总收入": "100",
        "营业总收入同比增长": "8.5%", "净利润": "10", "净利润同比增长": "6%",
        "净资产收益率": "5%", "每股经营现金流": "0.4", "每股收益": "0.3", "销售毛利率": "20%",
    }])
    provider._cached_frame = lambda *args, **kwargs: raw.copy()
    out = ValuationDataProvider.performance(provider, "20260630")
    assert set(["code", "revenue", "net_profit", "roe_h1_pct", "ocfps", "eps", "gross_margin_pct"]).issubset(out.columns)
    assert out.iloc[0]["code"] == "000001"
    assert out.iloc[0]["revenue_yoy"] == 8.5


class StockProvider:
    def __init__(self, price=10.0, history_failed=False):
        self.price = price
        self.history_failed = history_failed

    def spot(self):
        return pd.DataFrame([{
            "code": "000001", "name": "市场名称", "price": self.price,
            "market_cap": 1000.0, "pe_dynamic": 12.0, "pb": 2.0,
        }])

    def fundamentals_ttm(self, *args):
        return pd.DataFrame([{
            "code": "000001", "ttm_revenue": 200.0, "ttm_profit": 50.0,
            "ttm_eps": 1.0, "ttm_ocfps": 1.2,
            "ttm_revenue_method": "EXACT_TTM", "ttm_profit_method": "EXACT_TTM",
            "ttm_eps_method": "EXACT_TTM", "ttm_ocfps_method": "EXACT_TTM",
            "ttm_revenue_confidence": 1.0, "ttm_profit_confidence": 1.0,
            "ttm_eps_confidence": 1.0, "ttm_ocfps_confidence": 1.0,
            "revenue_cur": 110.0, "revenue_pri": 100.0,
            "net_profit_cur": 27.5, "net_profit_pri": 25.0,
            "revenue_yoy_cur": 10.0, "profit_yoy_cur": 10.0,
            "roe_h1_pct_cur": 10.0, "roe_h1_pct_pri": 9.0,
            "gross_margin_pct_cur": 30.0, "gross_margin_pct_pri": 29.0,
        }])

    def stock_history_indicator(self, code):
        if self.history_failed:
            out = pd.DataFrame()
            out.attrs.update(history_status="FAILED", history_source="AKShare/LeGu")
            return out
        out = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=40, freq="B"),
            "pe_ttm": np.linspace(18, 22, 40),
        })
        out.attrs.update(history_status="OK", history_source="AKShare/LeGu")
        return out


def _stock_config():
    return {
        "report_date": "20260630", "prior_report_date": "20250630", "annual_report_date": "20251231",
        "history_min_points": 30, "_report_min_coverage": 1.0,
        "valuation_policy": {"history_shrinkage_kappa": 60},
        "sectors": {"测试": {
            "model": "growth_pe", "target_peg": 1.2, "growth_floor_pct": 5, "growth_cap_pct": 30,
            "fair_pe_floor": 10, "fair_pe_cap": 30, "bootstrap_metric": "pe",
            "model_weights": {"pe": 1.0}, "base_margin": .2, "base_uncertainty": .12,
        }},
    }


def test_fair_price_is_independent_of_current_price_when_fundamentals_fixed(tmp_path):
    info = {"name": "配置名称", "sector": "测试"}
    first = ValuationEngine(StockProvider(price=10), _stock_config(), tmp_path / "a").evaluate_stock("000001", info)
    second = ValuationEngine(StockProvider(price=20), _stock_config(), tmp_path / "b").evaluate_stock("000001", info)
    assert first["fair_price_center"] == second["fair_price_center"]
    assert first["value_deviation"] < second["value_deviation"]
    assert first["name_match"] is False


def test_history_failure_is_visible(tmp_path):
    row = ValuationEngine(StockProvider(history_failed=True), _stock_config(), tmp_path).evaluate_stock(
        "000001", {"name": "市场名称", "sector": "测试"}
    )
    assert row["history_status"] == "FAILED"
    assert row["history_source"] == "AKShare/LeGu"


def test_unknown_model_is_no_trade_instead_of_silent_fallback(tmp_path):
    row = ValuationEngine(StockProvider(), _stock_config(), tmp_path).evaluate_stock(
        "000001", {"name": "市场名称", "sector": "未配置行业"}
    )
    assert row["valuation_status"] == "MODEL_UNRESOLVED"
    assert row["action"] == "NO_TRADE"
    assert row["data_quality"] == 0
