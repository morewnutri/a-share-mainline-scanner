import numpy as np
import pandas as pd

from mainline_scanner.analysis import build_metric_table, score_boards
from mainline_scanner.audit import build_completeness_audit
from mainline_scanner.report import write_outputs
from mainline_scanner.research import add_research_scores, add_switch_signals, enrich_market_history


def test_report_writes_all_artifacts(tmp_path):
    boards, histories, flow_rows = [], {}, []
    for i in range(6):
        code, name = f"B{i}", f"板块{i}"
        boards.append({"kind": "industry", "code": code, "name": name, "breadth": .4 + i * .08})
        x = np.arange(45, dtype=float)
        close = np.exp(5 + (-.003 + i * .002) * x + (i > 3) * .0002 * np.maximum(x - 38, 0) ** 2)
        histories[("industry", code)] = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=45, freq="B"),
            "close": close, "amount": np.linspace(1e9, (1.2 + i * .2) * 1e9, 45),
            "turnover": np.linspace(1, 1 + i * .1, 45),
        })
        flow_rows.append({"kind": "industry", "name": name, "flow_1d_pct": i - 2, "flow_5d_pct": i - 3, "flow_10d_pct": i - 4})
    board_frame, flow_frame = pd.DataFrame(boards), pd.DataFrame(flow_rows)
    metrics = enrich_market_history(build_metric_table(board_frame, histories, flow_frame), histories, pd.DataFrame())
    scored = add_switch_signals(add_research_scores(score_boards(metrics)))
    audit, summary = build_completeness_audit(board_frame, board_frame, histories, [], flow_frame, scored)
    paths = write_outputs(scored, histories, pd.DataFrame(), tmp_path, audit, summary)
    assert set(paths) == {
        "csv", "sideways_csv", "xlsx", "dashboard", "sideways_chart", "trends",
        "markdown", "html", "audit_xlsx", "omitted_csv",
        "研究潜在主线", "市场确认主线", "主线切换", "退潮风险",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert "研究潜在主线" in paths["markdown"].read_text(encoding="utf-8")
