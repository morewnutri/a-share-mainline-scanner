import json
import sys

import pandas as pd
import pytest

from mainline_scanner.valuation_cli import (
    _ask_new_stock,
    add_stock_to_universe,
    parse_new_stock_input,
    persist_config,
)


class FakeEngine:
    """Minimal stand-in for ValuationEngine, exposing just what
    add_stock_to_universe needs: the `master` lookup table and
    `evaluate_stock`."""

    def __init__(self, master: pd.DataFrame, row: dict):
        self.master = master
        self._row = row
        self.calls = []

    def evaluate_stock(self, code, info):
        self.calls.append((code, info))
        return dict(self._row)


def make_cfg():
    return {
        "sectors": {"半导体": {"model": "growth_pe"}},
        "stocks": {"600000": {"name": "浦发银行", "sector": "银行"}},
    }


def make_master():
    return pd.DataFrame({"code": ["688012"], "name": ["中微公司"]})


# ---- parse_new_stock_input -------------------------------------------------

def test_parse_full_comma_separated_input():
    assert parse_new_stock_input("688012,中微公司,半导体") == ("688012", "中微公司", "半导体")


def test_parse_fullwidth_comma_and_whitespace():
    assert parse_new_stock_input("688012，中微公司 半导体") == ("688012", "中微公司", "半导体")


def test_parse_code_only():
    assert parse_new_stock_input("688012") == ("688012", None, None)


def test_parse_none_is_cancelled():
    assert parse_new_stock_input(None) is None


@pytest.mark.parametrize("raw", ["", "   ", ",  ,"])
def test_parse_blank_input_returns_none(raw):
    assert parse_new_stock_input(raw) is None


# ---- add_stock_to_universe -------------------------------------------------

def test_add_new_stock_uses_configured_sector_model_and_updates_stock_list():
    cfg = make_cfg()
    row = {"entity": "中微公司", "code": "688012", "model_note": "PEG/正常化增长模型"}
    engine = FakeEngine(make_master(), row)

    result = add_stock_to_universe(engine, cfg, "688012", "中微公司", "半导体")

    assert result == row
    assert cfg["stocks"]["688012"] == {"name": "中微公司", "sector": "半导体"}
    assert engine.calls == [("688012", {"name": "中微公司", "sector": "半导体"})]
    # Pre-existing stock is untouched.
    assert cfg["stocks"]["600000"]["name"] == "浦发银行"


def test_add_new_stock_infers_name_from_master_when_omitted():
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {"entity": "中微公司", "code": "688012"})

    add_stock_to_universe(engine, cfg, "688012", None, None)

    assert cfg["stocks"]["688012"]["name"] == "中微公司"


def test_add_new_stock_warns_when_name_not_found_in_master(capsys):
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {"entity": "999999", "code": "999999"})

    add_stock_to_universe(engine, cfg, "999999", None, None)

    assert cfg["stocks"]["999999"]["name"] == "999999"
    assert "未在行情数据中找到" in capsys.readouterr().out


def test_add_new_stock_without_sector_falls_back_to_default_model(capsys):
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {"entity": "未知公司", "code": "999999"})

    result = add_stock_to_universe(engine, cfg, "999999", "未知公司", "冷门板块")

    assert "error" not in result
    captured = capsys.readouterr()
    assert "未配置板块的默认估值逻辑" in captured.out
    assert cfg["stocks"]["999999"]["sector"] == "冷门板块"
    assert any("未在配置中定义" in w for w in result["warnings"])


def test_add_new_stock_reports_evaluate_stock_exception_without_mutating_config():
    cfg = make_cfg()

    class RaisingEngine(FakeEngine):
        def evaluate_stock(self, code, info):
            raise RuntimeError("网络请求失败")

    engine = RaisingEngine(make_master(), {})

    result = add_stock_to_universe(engine, cfg, "688012", "中微公司", "半导体")

    assert "error" in result
    assert "网络请求失败" in result["error"]
    assert "688012" not in cfg["stocks"]


def test_add_new_stock_rejects_invalid_code():
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {})

    result = add_stock_to_universe(engine, cfg, "ABC123", "假股票", None)

    assert "error" in result
    assert "688012" not in cfg["stocks"]
    assert "ABC123" not in cfg["stocks"]
    assert engine.calls == []


def test_add_new_stock_rejects_duplicate():
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {})

    result = add_stock_to_universe(engine, cfg, "600000", "浦发银行", "银行")

    assert "error" in result
    assert "已存在" in result["error"]
    assert engine.calls == []
    assert cfg["stocks"]["600000"]["name"] == "浦发银行"


def test_add_new_stock_reports_missing_market_data_without_mutating_config():
    cfg = make_cfg()
    engine = FakeEngine(make_master(), {"error": "股票不存在或无行情"})

    result = add_stock_to_universe(engine, cfg, "000001", "无效股票", None)

    assert "error" in result
    assert "000001" not in cfg["stocks"]


# ---- persist_config ---------------------------------------------------------

def test_persist_config_writes_json(tmp_path):
    cfg = make_cfg()
    path = tmp_path / "valuation_config.json"

    assert persist_config(cfg, path) is True
    assert json.loads(path.read_text(encoding="utf-8")) == cfg


def test_persist_config_reports_failure_gracefully(tmp_path, capsys):
    cfg = make_cfg()
    missing_dir_path = tmp_path / "no_such_dir" / "valuation_config.json"

    assert persist_config(cfg, missing_dir_path) is False
    assert "无法写入配置文件" in capsys.readouterr().out


# ---- _ask_new_stock GUI/CLI fallback ---------------------------------------

def test_ask_new_stock_falls_back_to_cli_input_when_tkinter_unavailable(monkeypatch):
    # Force the "no GUI toolkit" branch deterministically, regardless of
    # whether the environment running this test happens to have a display.
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "688012,中微公司,半导体")

    assert _ask_new_stock() == "688012,中微公司,半导体"


def test_ask_new_stock_returns_none_when_cli_input_cancelled(monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert _ask_new_stock() is None
