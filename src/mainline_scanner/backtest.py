from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .snapshot_store import SnapshotStore


def evaluate_snapshots(
    snapshot_dir: Path,
    *,
    top_k: int = 10,
    horizon: int = 10,
    ignition_threshold: float = 70,
    mainline_threshold: float = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """评价扫描器本身的提前量；不把它包装成交易策略回测。"""
    store = SnapshotStore(snapshot_dir)
    refs = store.list()
    if not refs:
        return pd.DataFrame(), pd.DataFrame()
    latest_by_day = {}
    for ref in refs:
        latest_by_day[ref.captured_at.date()] = ref
    days = sorted(latest_by_day)
    frames = [store._read(latest_by_day[day]).assign(snapshot_date=pd.Timestamp(day)) for day in days]
    rows: list[dict[str, object]] = []
    for index, current in enumerate(frames):
        if "ignition_score" not in current:
            continue
        candidates = current[current["ignition_score"] >= ignition_threshold].nlargest(top_k, "ignition_score")
        future = frames[index + 1:index + 1 + horizon]
        if not future:
            continue
        for signal in candidates.to_dict("records"):
            key = (str(signal["kind"]), str(signal["code"]))
            future_rows = []
            for offset, frame in enumerate(future, start=1):
                match = frame[(frame["kind"].astype(str) == key[0]) & (frame["code"].astype(str) == key[1])]
                if not match.empty:
                    future_rows.append((offset, match.iloc[0]))
            hits = [(offset, row) for offset, row in future_rows if float(row.get("mainline_score", 0)) >= mainline_threshold]
            forward_rs = []
            for offset, row in future_rows[:10]:
                ret = pd.to_numeric(row.get("ret_1d"), errors="coerce")
                median = pd.to_numeric(
                    frames[index + offset].loc[frames[index + offset]["kind"] == signal["kind"], "ret_1d"], errors="coerce"
                ).median() if index + offset < len(frames) else np.nan
                if pd.notna(ret) and pd.notna(median):
                    forward_rs.append(float(ret - median))
            rows.append({
                "signal_date": current["snapshot_date"].iloc[0],
                "kind": signal["kind"], "code": signal["code"], "name": signal.get("name", ""),
                "ignition_score": signal["ignition_score"],
                "alert_ret_5d": signal.get("ret_5d", np.nan),
                "became_mainline": bool(hits),
                "lead_time_sessions": hits[0][0] if hits else np.nan,
                "false_start": not bool(hits),
                "forward_rs_3d": float(np.nansum(forward_rs[:3])) if forward_rs else np.nan,
                "forward_rs_5d": float(np.nansum(forward_rs[:5])) if forward_rs else np.nan,
                "forward_rs_10d": float(np.nansum(forward_rs[:10])) if forward_rs else np.nan,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = pd.DataFrame([{
        "signals": len(detail),
        f"precision_at_{top_k}": float(detail["became_mainline"].mean()),
        "false_start_rate": float(detail["false_start"].mean()),
        "median_lead_time_sessions": float(detail.loc[detail["became_mainline"], "lead_time_sessions"].median()),
        "median_alert_ret_5d": float(detail["alert_ret_5d"].median()),
        "mean_forward_rs_3d": float(detail["forward_rs_3d"].mean()),
        "mean_forward_rs_5d": float(detail["forward_rs_5d"].mean()),
        "mean_forward_rs_10d": float(detail["forward_rs_10d"].mean()),
    }])
    return detail, summary


def write_backtest(snapshot_dir: Path, output_dir: Path, **kwargs: object) -> dict[str, Path]:
    detail, summary = evaluate_snapshots(snapshot_dir, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "火种信号回放明细.csv"
    summary_path = output_dir / "火种信号回放汇总.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"backtest_detail": detail_path, "backtest_summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="回放评估主线火种发现能力")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/backtest"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    paths = write_backtest(args.snapshot_dir, args.output_dir, top_k=args.top_k, horizon=args.horizon)
    print("\n".join(f"{name}: {path.resolve()}" for name, path in paths.items()))


if __name__ == "__main__":
    main()
