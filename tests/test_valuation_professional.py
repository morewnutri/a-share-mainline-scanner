import json
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
        "ttm_operating_cashflow": [4.0, -2.0],
    })
    result = engine._aggregate(pd.DataFrame({"code": ["1", "2"]}))
    assert math.isnan(result["pe"])
    assert result["positive_profit_pe"] == 10
    assert result["aggregate_pe"] == 50
    assert result["profitable_mcap_coverage"] == .30
    assert result["loss_mcap_share"] == .70
    assert result["pb_mcap_coverage"] == .30
    assert result["ps_mcap_coverage"] == 1.0
    assert result["cash_conversion"] == 1.0


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
            "market_cap": self.price * 100.0, "pe_dynamic": 12.0, "pb": 2.0,
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
    assert first["price_deviation"] < second["price_deviation"]
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


def test_curated_config_fixes_wrong_board_and_stock_classifications():
    cfg = load_config(Path(__file__).parents[1] / "valuation_config.json")
    grid_aliases = cfg["sectors"]["电网设备"]["valuation_boards"][0]["aliases"]
    assert "电池" not in grid_aliases
    assert "电网设备" in grid_aliases
    assert cfg["sectors"]["电网设备"]["valuation_boards"][0]["board_code"] == "BK0457"
    assert cfg["stocks"]["002050"]["sector"] == "热管理/汽车零部件"
    assert cfg["stocks"]["002645"]["sector"] == "资源循环/稀土资源"
    assert cfg["stocks"]["603601"]["sector"] == "过滤材料/功能材料"
    assert cfg["stocks"]["000823"]["sector"] == "PCB/电子元件"


def test_performance_prefers_attributable_profit_over_generic_profit():
    provider = object.__new__(ValuationDataProvider)
    raw = pd.DataFrame([{
        "股票代码": "000001", "股票简称": "测试", "营业总收入": 100,
        "归属于母公司所有者的净利润": 8, "净利润": 10,
        "归属于母公司所有者的净利润同比增长": 12,
    }])
    provider._cached_frame = lambda *args, **kwargs: raw.copy()
    row = ValuationDataProvider.performance(provider, "20260630").iloc[0]
    assert row["attributable_net_profit"] == 8
    assert row["net_profit"] == 8
    assert row["consolidated_net_profit"] == 10


def test_ttm_per_share_metrics_are_not_period_additions():
    provider = TTMProvider({
        "20260630": _performance(net_profit=6, eps=.30, ocfps=.4),
        "20250630": _performance(net_profit=5, eps=.50, ocfps=.8),
        "20251231": _performance(net_profit=10, eps=1.20, ocfps=1.4),
    })
    row = ValuationDataProvider.fundamentals_ttm(provider, "20260630", "20250630", "20251231").iloc[0]
    assert math.isnan(row["ttm_eps"])
    assert row["ttm_eps_method"].startswith("DERIVE_FROM_TTM_ATTRIBUTABLE")
    assert row["ttm_ocfps"] == .8  # annualised current-period fallback, not 1.4 + .4 - .8


def test_shares_and_multiples_close_from_market_cap_and_price(tmp_path):
    engine = ValuationEngine(StockProvider(price=10), _stock_config(), tmp_path)
    row = engine.master.iloc[0]
    assert row["shares_outstanding"] == 100
    assert row["ttm_eps"] == .5
    assert row["sps_ttm"] == 2
    assert row["pe_ttm_calc"] == row["price"] / row["ttm_eps"]
    assert row["ps_ttm_calc"] == row["price"] / row["sps_ttm"]


def test_ps_anchor_is_independent_and_changes_fair_value(tmp_path):
    cfg = _stock_config()
    cfg["sectors"]["测试"].update({
        "model_weights": {"pe": .5, "ps": .5}, "base_ps": 2,
        "fair_ps_floor": .5, "fair_ps_cap": 8, "revenue_growth_ps_slope": 0,
    })
    first = ValuationEngine(StockProvider(), cfg, tmp_path / "a").evaluate_stock("000001", {"name": "市场名称", "sector": "测试"})
    cfg["sectors"]["测试"]["base_ps"] = 6
    second = ValuationEngine(StockProvider(), cfg, tmp_path / "b").evaluate_stock("000001", {"name": "市场名称", "sector": "测试"})
    assert second["implied_price_ps"] > first["implied_price_ps"]
    assert second["fair_price_center"] > first["fair_price_center"]
    assert first["implied_price_pe"] != first["implied_price_ps"]


def test_cyclical_discount_is_clipped_after_discount():
    engine = object.__new__(ValuationEngine)
    c = {
        "model": "cyclical_growth", "target_peg": 1, "growth_floor_pct": 16, "growth_cap_pct": 30,
        "fair_pe_floor": 16, "fair_pe_cap": 32, "cycle_discount": .9,
    }
    fair, _ = engine._model_fair("测试", c, {"revenue_growth": 0, "profit_growth": 0, "net_margin": .1, "roe_ttm_pct": 10})
    assert fair["pe"] == 16


def test_turnaround_growth_is_flagged_and_not_capitalized_at_raw_rate():
    details = ValuationEngine._growth_diagnostics(
        {"revenue_growth": .20, "profit_growth": 13.28, "turnaround": True},
        {"growth_floor_pct": 5, "growth_cap_pct": 55, "profit_growth_premium_cap_pct": 20},
    )
    assert details["growth_regime"] == "TURNAROUND"
    assert details["growth_used_pct"] < 55
    assert details["growth_used_pct"] < details["raw_profit_growth_pct"]


def test_missing_large_model_weight_is_not_silently_renormalized(tmp_path):
    cfg = _stock_config()
    cfg["valuation_policy"]["required_model_weight_coverage"] = .7
    cfg["sectors"]["测试"].update({"model_weights": {"pe": .6, "ocf_yield": .4}, "target_ocf_yield": .06})
    provider = StockProvider()
    original = provider.fundamentals_ttm
    provider.fundamentals_ttm = lambda *args: original(*args).assign(ttm_ocfps=np.nan, ttm_ocfps_confidence=0)
    row = ValuationEngine(provider, cfg, tmp_path).evaluate_stock("000001", {"name": "市场名称", "sector": "测试"})
    assert row["model_weight_coverage"] == .6
    assert row["valuation_status"] == "LOW_MODEL_COVERAGE"
    assert row["effective_weight_pe"] == 1.0
    assert row["configured_weight_ocf_yield"] == .4
    assert row["model_drop_reason_ocf_yield"]


def test_snapshot_history_isolated_by_model_and_config_version(tmp_path):
    engine = ValuationEngine(StockProvider(), _stock_config(), tmp_path)
    stale = pd.DataFrame([{
        "trade_date": "2026-01-02", "code": "000001", "valuation_residual": .1,
        "valuation_model_version": "old", "config_hash": "old", "target_position_pct": 70,
    }])
    stale.to_csv(tmp_path / "stock_valuation_snapshots.csv", index=False)
    stats = engine._residual_history("000001", "current")
    assert stats.points == 0
    assert stats.status == "MISSING"


def test_saved_snapshot_uses_quote_date_and_version_fields(tmp_path):
    engine = ValuationEngine(StockProvider(), _stock_config(), tmp_path)
    engine.quote_trade_date = "2026-10-08"
    engine.save_snapshots([], [{
        "code": "000001", "entity": "测试", "price": 10, "fair_price_center": 9,
        "valuation_residual": .1, "data_quality": .8, "model_family": "growth_pe",
        "target_position_pct": 20, "action": "WAIT", "valuation_status": "OK",
        "config_hash": "stock-hash", "quote_trade_date": "2026-10-08",
    }])
    row = pd.read_csv(tmp_path / "stock_valuation_snapshots.csv", dtype={"code": str}).iloc[0]
    assert row["trade_date"] == "2026-10-08"
    assert row["valuation_model_version"] == engine.model_version
    assert row["config_hash"] == "stock-hash"


def test_failed_gate_reasonable_zone_is_zero_position():
    bands = {"deep_buy_price": 50, "buy_price": 70, "buy_exit_price": 75, "trim_price": 120, "exit_price": 140}
    target, action, _ = position_policy(100, bands, "Mainline", True, "不通过-业绩下滑")
    assert target == 0
    assert action == "NO_TRADE"


def test_prior_bands_watch_gate_and_decay_all_reduce_position():
    bands = {"deep_buy_price": 50, "buy_price": 70, "buy_exit_price": 75, "trim_price": 120, "exit_price": 140}
    target, _, _ = position_policy(60, bands, "Mainline", True, "观察-低于模型族增长阈值", watch_gate_scale=.5, prior_band_max_position=30, trade_band_source="MODEL_PRIOR_BANDS")
    assert target == 30
    prior, _, _ = position_policy(72, bands, "Mainline", True, "通过", previous_target=70, prior_band_max_position=30, trade_band_source="MODEL_PRIOR_BANDS")
    assert prior == 30
    decay, action, _ = position_policy(100, bands, "Decay", True, "通过")
    assert decay == 20
    assert action == "TRIM_OR_WAIT"


def test_config_rejects_mismatched_valuation_board_and_bad_weights(tmp_path):
    cfg = json.loads((Path(__file__).parents[1] / "valuation_config.json").read_text(encoding="utf-8"))
    cfg["sectors"]["电网设备"]["valuation_boards"] = [{"kind": "industry", "aliases": ["电池"]}]
    cfg["sectors"]["半导体"]["model_weights"] = {"pe": .8, "ps": .3}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    try:
        load_config(path)
    except ValueError as exc:
        assert "完全不相交" in str(exc)
        assert "权重和必须为1" in str(exc)
    else:
        raise AssertionError("bad config must fail fast")


def test_ambiguous_inferred_sector_requires_review():
    provider = object.__new__(ValuationDataProvider)
    provider.sector_universe = lambda sector_cfg, purpose="inference": (
        pd.DataFrame({"code": ["000001"], "name": ["测试"], "board_hits": [1]}), []
    )
    cfg = {
        "classification_policy": {"min_confidence": .6}, "sector_priority": ["甲", "乙"],
        "sectors": {"甲": {"boards": [{"kind": "concept", "aliases": ["A"]}]}, "乙": {"boards": [{"kind": "concept", "aliases": ["B"]}]}},
    }
    result = provider.infer_stock_sector("000001", cfg)
    assert result["classification_status"] == "MODEL_REVIEW_REQUIRED"
    assert result["sector"] == "MODEL_REVIEW_REQUIRED"
    assert len(result["sector_candidates"]) == 2
