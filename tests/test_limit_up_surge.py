from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from mainline_scanner.limit_up_surge import enrich_limit_up_surge, surge_rule


class FakeAk:
    def stock_zt_pool_em(self, date):
        assert date in {"20250107", "20250108"}
        codes = (["000001", "000002", "000003", "000004", "000005", "000006", "000007"]
                 if date == "20250108" else ["000001", "000005"])
        return pd.DataFrame({"代码": codes})

    def stock_board_industry_cons_em(self, symbol):
        if symbol == "缺失":
            raise RuntimeError("接口失败")
        if symbol == "零基数":
            return pd.DataFrame({"代码": ["000002", "000003", "000004"]})
        return pd.DataFrame({"代码": ["000001", "000002", "000003", "000004"]})

    def stock_board_concept_cons_em(self, symbol):
        return pd.DataFrame({"代码": ["000005", "000006", "000007"]})


def test_limit_up_surge_uses_adjacent_sessions_and_handles_missing_boards(tmp_path):
    scored = pd.DataFrame([
        {"kind": "industry", "code": "B1", "name": "行业", "as_of": "2025-01-08"},
        {"kind": "concept", "code": "C1", "name": "概念", "as_of": "2025-01-08"},
        {"kind": "industry", "code": "B2", "name": "缺失", "as_of": "2025-01-08"},
        {"kind": "industry", "code": "B3", "name": "过期", "as_of": "2025-01-07"},
        {"kind": "industry", "code": "B4", "name": "零基数", "as_of": "2025-01-08"},
    ])
    provider = SimpleNamespace(ak=FakeAk(), cache_dir=tmp_path, refresh=False)
    result = enrich_limit_up_surge(scored, provider, datetime(2025, 1, 8, 16), workers=2)
    assert result.loc[0, "limit_up_previous_date"] == "2025-01-07"
    assert result.loc[0, "limit_up_today"] == 4
    assert result.loc[0, "limit_up_previous"] == 1
    assert result.loc[0, "limit_up_increase"] == 3
    assert result.loc[0, "limit_up_ratio"] == 4
    assert bool(result.loc[0, "limit_up_surge"])
    assert result.loc[1, "limit_up_increase"] == 2
    assert not bool(result.loc[1, "limit_up_surge"])
    assert result.loc[2, "limit_up_status"].startswith("成分股不可用")
    assert result.loc[3, "limit_up_status"] == "板块行情非本次收盘日"
    assert result.loc[4, "limit_up_previous"] == 0
    assert pd.isna(result.loc[4, "limit_up_ratio"])
    assert bool(result.loc[4, "limit_up_surge"])
    assert "增加 ≥ 3" in surge_rule(3, 3, 2)


def test_limit_up_surge_never_treats_unavailable_pool_as_zero(tmp_path):
    class EmptyPool(FakeAk):
        def stock_zt_pool_em(self, date):
            return pd.DataFrame()

    scored = pd.DataFrame([{"kind": "industry", "code": "B1", "name": "行业", "as_of": "2025-01-08"}])
    result = enrich_limit_up_surge(scored, SimpleNamespace(ak=EmptyPool(), cache_dir=tmp_path),
                                   datetime(2025, 1, 8, 16))
    assert result.loc[0, "limit_up_status"].startswith("涨停池不可用")
    assert pd.isna(result.loc[0, "limit_up_today"])
    assert not bool(result.loc[0, "limit_up_surge"])
