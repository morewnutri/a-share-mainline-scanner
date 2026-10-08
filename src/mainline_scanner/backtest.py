from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .snapshot_store import SnapshotStore


def replay_daily_histories(boards: pd.DataFrame, histories: dict[tuple[str, str], pd.DataFrame],
                           market: pd.DataFrame, output_dir: Path, *, min_history: int = 20,
                           horizon: int = 10) -> dict[str, Path]:
    """Rebuild each close from data known then; never reuse today's flow or constituents."""
    from .analysis import build_metric_table, score_boards
    from .market_metrics import enrich_market_history
    from .quant_scores import add_quant_radar

    if not histories:
        raise ValueError("历史回放需要板块日线")
    safe_boards = boards.drop(columns=["breadth", "up_count", "down_count", "snapshot_return",
                                       "snapshot_amount", "snapshot_turnover", "leader_stock"], errors="ignore")
    calendars = [set(pd.to_datetime(h["date"]).dt.normalize()) for h in histories.values() if len(h) >= min_history]
    days = sorted(set.intersection(*calendars)) if calendars else []
    replay_dir = output_dir / "replay_snapshots"
    store = SnapshotStore(replay_dir)
    for day in days[min_history - 1:]:
        sliced = {key: h[pd.to_datetime(h["date"]).dt.normalize() <= day].copy()
                  for key, h in histories.items()}
        sliced = {key: h for key, h in sliced.items() if len(h) >= 12}
        if not sliced:
            continue
        metrics = build_metric_table(safe_boards, sliced, pd.DataFrame())
        known_market = market[market["date"] <= day] if not market.empty else market
        metrics = enrich_market_history(metrics, sliced, known_market)
        preliminary = add_quant_radar(score_boards(metrics))
        captured = pd.Timestamp(day) + pd.Timedelta(hours=16)
        enriched = store.enrich(preliminary, captured)
        scored = add_quant_radar(score_boards(enriched))
        store.save(scored, captured)
    paths = write_backtest(replay_dir, output_dir, horizon=horizon)
    paths["replay_snapshots"] = replay_dir
    return paths


def evaluate_future_dominance(snapshot_dir: Path, horizon: int = 20, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-only dominance label; requires a full future window of saved daily snapshots."""
    store = SnapshotStore(snapshot_dir)
    frames = store.daily_closes()
    rows = []
    for start in range(max(0, len(frames) - horizon)):
        current = frames[start]
        future = frames[start + 1:start + 1 + horizon]
        signal_col = "ignition_score"
        if len(future) < horizon or signal_col not in current:
            continue
        for kind, group in current.groupby("kind"):
            candidate_cols = ["kind", "code", "name", signal_col]
            if "last_close" in group:
                candidate_cols.append("last_close")
            candidates = group[candidate_cols].copy()
            values = []
            for _, candidate in candidates.iterrows():
                key = str(candidate["code"])
                daily = []
                future_mainline = False
                for frame in future:
                    peers = frame[frame["kind"].astype(str) == str(kind)].copy()
                    peers["code"] = peers["code"].astype(str)
                    match = peers[peers["code"] == key]
                    if match.empty:
                        continue
                    r = pd.to_numeric(peers.get("ret_1d", pd.Series(np.nan, index=peers.index)), errors="coerce")
                    turnover = pd.to_numeric(peers.get("turnover_share", pd.Series(np.nan, index=peers.index)), errors="coerce")
                    row = match.iloc[0]
                    future_mainline = future_mainline or str(row.get("lifecycle", "")) == "Mainline"
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
                    "signal_date": pd.Timestamp(current["market_as_of"].iloc[0]),
                    "ignition_score_at_signal": candidate[signal_col],
                    "future_mainline_label": future_mainline,
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
            rows.append(block)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if detail.empty:
        return detail, pd.DataFrame()
    eligible = detail[detail["ignition_score_at_signal"].notna()]
    picks = eligible.sort_values("ignition_score_at_signal", ascending=False).groupby(
        ["signal_date", "kind"], group_keys=False
    ).head(top_k)
    summary = pd.DataFrame([{
        "evaluated_signals": len(picks), "horizon_sessions": horizon,
        f"ignition_precision_at_{top_k}": picks["future_mainline_label"].mean() if len(picks) else np.nan,
        "label_is_relative": False,
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
    frames = [frame.assign(snapshot_date=pd.Timestamp(frame["market_as_of"].iloc[0]))
              for frame in store.daily_closes()]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    for index, current in enumerate(frames):
        if "ignition_score" not in current:
            continue
        candidates = current[current["ignition_score"] >= ignition_threshold].nlargest(top_k, "ignition_score")
        future = frames[index + 1:index + 1 + horizon]
        for signal in candidates.to_dict("records"):
            key = (str(signal["kind"]), str(signal["code"]))
            future_rows = []
            for offset, frame in enumerate(future, start=1):
                match = frame[(frame["kind"].astype(str) == key[0]) & (frame["code"].astype(str) == key[1])]
                if not match.empty:
                    future_rows.append((offset, match.iloc[0]))
            hits = [(offset, row) for offset, row in future_rows
                    if str(row.get("lifecycle", "")) == "Mainline"
                    and pd.to_numeric(row.get("mainline_score"), errors="coerce") >= mainline_threshold]
            complete = len(future) == horizon and len(future_rows) == horizon
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
                "outcome": "success" if complete and hits else "failure" if complete else "censored",
                "became_mainline": bool(hits) if complete else np.nan,
                "lead_time_sessions": hits[0][0] if hits and complete else np.nan,
                "false_start": not bool(hits) if complete else np.nan,
                "forward_rs_3d": float(np.nansum(forward_rs[:3])) if forward_rs else np.nan,
                "forward_rs_5d": float(np.nansum(forward_rs[:5])) if forward_rs else np.nan,
                "forward_rs_10d": float(np.nansum(forward_rs[:10])) if forward_rs else np.nan,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    evaluated = detail[detail["outcome"] != "censored"]
    mainline_events = 0
    missed_events = 0
    for index in range(horizon, len(frames)):
        current = frames[index]
        previous = frames[index - 1]
        for row in current[current.get("lifecycle", pd.Series("", index=current.index)).eq("Mainline")].to_dict("records"):
            key = (str(row["kind"]), str(row["code"]))
            prior_state = previous[(previous["kind"].astype(str) == key[0]) & (previous["code"].astype(str) == key[1])]
            if not prior_state.empty and str(prior_state.iloc[0].get("lifecycle", "")) == "Mainline":
                continue
            mainline_events += 1
            alerted = any(
                not (match := frame[(frame["kind"].astype(str) == key[0]) & (frame["code"].astype(str) == key[1])]).empty
                and pd.to_numeric(match.iloc[0].get("ignition_score"), errors="coerce") >= ignition_threshold
                for frame in frames[index - horizon:index]
            )
            missed_events += not alerted
    summary = pd.DataFrame([{
        "signals": len(detail), "evaluated_signals": len(evaluated),
        "censored_signals": int((detail["outcome"] == "censored").sum()),
        "new_mainline_events": mainline_events,
        "missed_mainline_events": missed_events,
        "miss_rate": missed_events / mainline_events if mainline_events else np.nan,
        f"precision_at_{top_k}": float(evaluated["became_mainline"].mean()) if len(evaluated) else np.nan,
        "false_start_rate": float(evaluated["false_start"].mean()) if len(evaluated) else np.nan,
        "median_lead_time_sessions": float(detail.loc[detail["outcome"] == "success", "lead_time_sessions"].median()),
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
