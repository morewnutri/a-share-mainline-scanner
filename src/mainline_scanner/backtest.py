from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .snapshot_store import SnapshotStore


def evaluate_future_dominance(snapshot_dir: Path, horizon: int = 20, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-only dominance label; requires a full future window of saved daily snapshots."""
    store = SnapshotStore(snapshot_dir)
    latest = {}
    for ref in store.list():
        latest[ref.captured_at.date()] = ref
    frames = [store._read(latest[day]) for day in sorted(latest)]
    rows = []
    for start in range(max(0, len(frames) - horizon)):
        current = frames[start]
        future = frames[start + 1:start + 1 + horizon]
        if len(future) < horizon or "potential_score" not in current:
            continue
        for kind, group in current.groupby("kind"):
            candidate_cols = ["kind", "code", "name", "potential_score"]
            if "potential_rank_score" in group:
                candidate_cols.append("potential_rank_score")
            if "last_close" in group:
                candidate_cols.append("last_close")
            candidates = group[candidate_cols].copy()
            values = []
            for _, candidate in candidates.iterrows():
                key = str(candidate["code"])
                daily = []
                for frame in future:
                    peers = frame[frame["kind"].astype(str) == str(kind)].copy()
                    peers["code"] = peers["code"].astype(str)
                    match = peers[peers["code"] == key]
                    if match.empty:
                        continue
                    r = pd.to_numeric(peers.get("ret_1d", pd.Series(np.nan, index=peers.index)), errors="coerce")
                    turnover = pd.to_numeric(peers.get("turnover_share", pd.Series(np.nan, index=peers.index)), errors="coerce")
                    row = match.iloc[0]
                    daily.append({
                        "excess": pd.to_numeric(row.get("ret_1d"), errors="coerce") - r.median(),
                        "top_day": float(r.rank(pct=True).loc[match.index[0]] >= .8) if r.notna().sum() >= 3 else np.nan,
                        "share_rank": turnover.rank(pct=True).loc[match.index[0]] if turnover.notna().sum() >= 3 else np.nan,
                        "breadth": pd.to_numeric(row.get("breadth"), errors="coerce"),
                        "close": pd.to_numeric(row.get("last_close"), errors="coerce"),
                    })
                if len(daily) < horizon:
                    continue
                series = pd.DataFrame(daily)
                close = series["close"]
                starting_close = pd.to_numeric(candidate.get("last_close"), errors="coerce")
                if pd.notna(starting_close):
                    close = pd.concat([pd.Series([starting_close]), close], ignore_index=True)
                max_drawdown = (close / close.cummax() - 1).min() if close.notna().all() else np.nan
                values.append({
                    "kind": kind, "code": key, "name": candidate["name"],
                    "signal_date": pd.Timestamp(sorted(latest)[start]),
                    "potential_score_at_signal": candidate["potential_score"],
                    "potential_rank_score_at_signal": candidate.get("potential_rank_score", candidate["potential_score"]),
                    "future_turnover_rank": series["share_rank"].mean(),
                    "future_excess_return": series["excess"].sum(min_count=1),
                    "future_top_strength_days": series["top_day"].mean(),
                    "future_breadth": series["breadth"].mean(),
                    "future_max_drawdown": max_drawdown,
                })
            if not values:
                continue
            block = pd.DataFrame(values)
            components = ["future_turnover_rank", "future_excess_return", "future_top_strength_days",
                          "future_breadth", "future_max_drawdown"]
            ranked = block[components].rank(pct=True)
            block["future_dominance_score"] = ranked.mean(axis=1) * 100
            block["future_dominance_coverage"] = ranked.notna().mean(axis=1)
            block["future_mainline_label"] = block["future_dominance_score"].rank(pct=True) >= .8
            rows.append(block)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if detail.empty:
        return detail, pd.DataFrame()
    eligible = detail[detail["potential_rank_score_at_signal"].notna()]
    picks = eligible.sort_values("potential_rank_score_at_signal", ascending=False).groupby(
        ["signal_date", "kind"], group_keys=False
    ).head(top_k)
    summary = pd.DataFrame([{
        "evaluated_signals": len(picks), "horizon_sessions": horizon,
        f"potential_precision_at_{top_k}": picks["future_mainline_label"].mean() if len(picks) else np.nan,
        "label_is_relative": True,
    }])
    return detail, summary


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
    dominance, dominance_summary = evaluate_future_dominance(snapshot_dir)
    dominance_path = output_dir / "未来主线标签明细.csv"
    dominance_summary_path = output_dir / "未来主线标签汇总.csv"
    dominance.to_csv(dominance_path, index=False, encoding="utf-8-sig")
    dominance_summary.to_csv(dominance_summary_path, index=False, encoding="utf-8-sig")
    return {"backtest_detail": detail_path, "backtest_summary": summary_path,
            "dominance_detail": dominance_path, "dominance_summary": dominance_summary_path}


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
