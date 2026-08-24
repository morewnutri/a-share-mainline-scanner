from datetime import timedelta

import pandas as pd

from mainline_scanner.data import EastmoneyAkshareProvider


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
