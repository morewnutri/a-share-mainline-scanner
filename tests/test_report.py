import numpy as np
import pandas as pd

from mainline_scanner.analysis import build_metric_table, score_boards
from mainline_scanner.audit import build_completeness_audit
from mainline_scanner.report import write_outputs


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
    scored = score_boards(build_metric_table(board_frame, histories, flow_frame))
    audit, summary = build_completeness_audit(board_frame, board_frame, histories, [], flow_frame, scored)
    paths = write_outputs(scored, histories, pd.DataFrame(), tmp_path, audit, summary)
    assert set(paths) == {"csv", "xlsx", "dashboard", "trends", "markdown", "html", "audit_xlsx", "omitted_csv"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
