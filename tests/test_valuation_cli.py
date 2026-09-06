from mainline_scanner.valuation_cli import (
    load_custom_stocks,
    merge_stock_pools,
    parse_stock_codes,
    save_custom_stocks,
)


def test_parse_stock_codes():
    assert parse_stock_codes("688256, 601138 300274") == ["688256", "601138", "300274"]
    assert parse_stock_codes(["sh.600519", "000001.SZ", "600519"]) == ["600519", "000001"]


def test_custom_stock_round_trip(tmp_path):
    stocks = {
        "688256": {
            "name": "寒武纪-U",
            "sector": "AI算力",
            "source": "user",
        }
    }
    save_custom_stocks(tmp_path, stocks)
    assert load_custom_stocks(tmp_path)["688256"]["sector"] == "AI算力"


def test_default_pool_wins_duplicate():
    default = {"600519": {"name": "贵州茅台", "sector": "通用成长"}}
    custom = {
        "600519": {"name": "错误名字", "sector": "AI算力"},
        "688256": {"name": "寒武纪-U", "sector": "AI算力"},
    }
    merged = merge_stock_pools(default, custom)
    assert merged["600519"]["name"] == "贵州茅台"
    assert "688256" in merged
