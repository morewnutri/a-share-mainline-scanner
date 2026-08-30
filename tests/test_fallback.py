import os
from datetime import datetime, timedelta

import pandas as pd

from mainline_scanner.data import EastmoneyAkshareProvider, _parse_market_dates


def test_history_falls_back_to_ths_when_eastmoney_fails(tmp_path):
    provider = object.__new__(EastmoneyAkshareProvider)
    provider.cache_dir = tmp_path
    provider.refresh = True
    provider.ttl = timedelta(hours=1)
    provider._eastmoney_history_available = True
    provider._direct_history = lambda *args: (_ for _ in ()).throw(ConnectionError("blocked"))
    provider._ths_history = lambda *args: pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=20, freq="B").strftime("%Y%m%d"),
        "开盘": range(100, 120), "最高": range(101, 121), "最低": range(99, 119),
        "收盘": range(100, 120), "成交量": range(1000, 1020), "成交额": range(2000, 2020),
        "数据源": ["同花顺"] * 20,
    })

    result = provider.get_history("industry", "BK0001", "测试行业", "20250101", "20250228")
    assert len(result) == 20
    assert result["data_source"].eq("同花顺").all()
    assert result["close"].iloc[-1] == 119


def test_board_name_normalization_matches_cross_source_names():
    normalize = EastmoneyAkshareProvider._normalize_board_name
    assert normalize("AI 芯片概念") == normalize("AI芯片")
    assert normalize("焦炭Ⅲ") == normalize("焦炭")


def test_numeric_yyyymmdd_cache_dates_do_not_become_1970():
    parsed = _parse_market_dates(pd.Series([20260821, "20260822", "2026-08-23", "20260824.0"]))
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24",
    ]


def test_industry_history_falls_back_to_sw_after_ths_fails(tmp_path):
    provider = object.__new__(EastmoneyAkshareProvider)
    provider.cache_dir = tmp_path
    provider.refresh = True
    provider.ttl = timedelta(hours=1)
    provider._eastmoney_history_available = False
    provider._ths_history = lambda *args: (_ for _ in ()).throw(LookupError("no ths"))
    provider._sw_history = lambda *args: pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=20, freq="B").strftime("%Y-%m-%d"),
        "开盘": range(100, 120), "最高": range(101, 121), "最低": range(99, 119),
        "收盘": range(100, 120), "成交量": range(1000, 1020), "成交额": range(2000, 2020),
        "数据源": ["申万研究"] * 20,
    })
    result = provider.get_history("industry", "BK0002", "测试行业", "20250101", "20250228")
    assert len(result) == 20
    assert result["data_source"].eq("申万研究").all()


def test_snapshot_and_history_use_separate_cache_ttl(tmp_path):
    path = tmp_path / "cache.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    ten_minutes_ago = (datetime.now() - timedelta(minutes=10)).timestamp()
    os.utime(path, (ten_minutes_ago, ten_minutes_ago))
    provider = object.__new__(EastmoneyAkshareProvider)
    provider.refresh = False
    provider.ttl = timedelta(hours=24)
    provider.snapshot_ttl = timedelta(minutes=5)
    assert provider._fresh(path)
    assert not provider._fresh_snapshot(path)
