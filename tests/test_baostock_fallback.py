import numpy as np
import pandas as pd

from mainline_scanner.baostock_fallback import BaoStockSyntheticProvider, to_baostock_code


def stock_history(multiplier=1.0):
    close = np.linspace(10, 12, 20) * multiplier
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20, freq="B"),
        "open": close * .995, "high": close * 1.01, "low": close * .99, "close": close,
        "volume": 1000, "amount": 10000, "turn": 1.2,
    })


def test_baostock_code_normalization():
    assert to_baostock_code("600000") == "sh.600000"
    assert to_baostock_code("000001.SZ") == "sz.000001"
    assert to_baostock_code("830001") == "bj.830001"


def test_baostock_constituents_are_synthesized_and_labeled():
    provider = object.__new__(BaoStockSyntheticProvider)
    provider.min_constituents = 3
    result = provider._synthesize([stock_history(1), stock_history(2), stock_history(3)])
    assert len(result) >= 18
    assert result["data_source"].eq("BaoStock成分股等权合成").all()
    assert result["constituent_count"].min() == 3
    assert result["close"].iloc[-1] > result["close"].iloc[0]
