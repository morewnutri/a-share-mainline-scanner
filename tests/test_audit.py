import pandas as pd

from mainline_scanner.audit import build_completeness_audit


def history(end="2025-02-28"):
    dates = pd.date_range(end=end, periods=20, freq="B")
    return pd.DataFrame({"date": dates, "close": range(100, 120)})


def test_audit_distinguishes_filter_failure_and_quality_warning():
    source = pd.DataFrame([
        {"kind": "industry", "code": "A", "name": "完整板块"},
        {"kind": "industry", "code": "B", "name": "失败板块"},
        {"kind": "industry", "code": "C", "name": "滞后板块"},
        {"kind": "industry", "code": "D", "name": "主动过滤板块"},
    ])
    target = source.iloc[:3].copy()
    histories = {
        ("industry", "A"): history(),
        ("industry", "C"): history("2025-02-27"),
    }
    failures = [{"kind": "industry", "code": "B", "name": "失败板块", "error": "网络超时"}]
    flows = pd.DataFrame({
        "kind": ["industry"], "name": ["完整板块"],
        "flow_1d_pct": [1.0], "flow_5d_pct": [2.0], "flow_10d_pct": [3.0],
    })
    scored = source.iloc[[0, 2]].copy()
    audit, summary = build_completeness_audit(source, target, histories, failures, flows, scored)
    indexed = audit.set_index("code")
    assert indexed.loc["A", "audit_status"] == "完整"
    assert indexed.loc["B", "audit_status"] == "日线抓取失败"
    assert indexed.loc["B", "is_omitted"]
    assert indexed.loc["C", "audit_status"] == "行情未到最新日"
    assert indexed.loc["D", "audit_status"] == "主动过滤"
    total = summary.set_index("kind").loc["all"]
    assert total["source_universe"] == 4
    assert total["scan_target"] == 3
    assert total["final_scored"] == 2
    assert total["omitted_from_scoring"] == 1
