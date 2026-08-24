from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _key(kind: object, code: object) -> tuple[str, str]:
    return str(kind), str(code)


def build_completeness_audit(
    source_boards: pd.DataFrame,
    target_boards: pd.DataFrame,
    histories: Mapping[tuple[str, str], pd.DataFrame],
    failures: list[dict[str, str]],
    flows: pd.DataFrame,
    scored: pd.DataFrame,
    calendar_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对账源全集、过滤后目标、历史行情、资金流与最终评分五个阶段。"""
    source = source_boards[["kind", "code", "name"]].copy()
    source["kind"] = source["kind"].astype(str)
    source["code"] = source["code"].astype(str)
    source["source_duplicate_code"] = source.duplicated(["kind", "code"], keep=False)
    source["source_duplicate_name"] = source.duplicated(["kind", "name"], keep=False)

    target_keys = {_key(r.kind, r.code) for r in target_boards[["kind", "code"]].itertuples(index=False)}
    scored_keys = {_key(r.kind, r.code) for r in scored[["kind", "code"]].itertuples(index=False)} if not scored.empty else set()
    failure_map = {
        _key(row.get("kind", ""), row.get("code", "")): str(row.get("error", ""))
        for row in failures
    }

    flow_sets: dict[int, set[tuple[str, str]]] = {}
    for window in (1, 5, 10):
        col = f"flow_{window}d_pct"
        if not flows.empty and col in flows:
            matched = flows.loc[flows[col].notna(), ["kind", "name"]]
            flow_sets[window] = {(str(r.kind), str(r.name)) for r in matched.itertuples(index=False)}
        else:
            flow_sets[window] = set()

    reference_date: dict[str, pd.Timestamp] = {}
    reference_calendar: dict[str, pd.DatetimeIndex] = {}
    for kind in source["kind"].unique():
        kind_frames = [h for (k, _), h in histories.items() if k == kind and not h.empty]
        if not kind_frames:
            continue
        reference_date[kind] = max(pd.to_datetime(h["date"]).max() for h in kind_frames)
        best = max(kind_frames, key=lambda h: len(h))
        reference_calendar[kind] = pd.DatetimeIndex(
            pd.to_datetime(best["date"]).dropna().sort_values().unique()[-calendar_days:]
        )

    records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        key = _key(row.kind, row.code)
        intentional_filter = key not in target_keys
        history = histories.get(key)
        error = failure_map.get(key, "")
        record: dict[str, object] = {
            "kind": row.kind, "code": row.code, "name": row.name,
            "source_duplicate_code": bool(row.source_duplicate_code),
            "source_duplicate_name": bool(row.source_duplicate_name),
            "intentional_filter": intentional_filter,
            "fetch_error": error,
            "history_source": "",
            "history_rows": 0, "history_start": pd.NaT, "history_end": pd.NaT,
            "stale_trading_days": np.nan, "missing_trading_days_count": 0,
            "missing_trading_dates": "", "duplicate_dates": 0,
            "invalid_close_rows": 0, "invalid_ohlc_rows": 0,
            "flow_1d_matched": (str(row.kind), str(row.name)) in flow_sets[1],
            "flow_5d_matched": (str(row.kind), str(row.name)) in flow_sets[5],
            "flow_10d_matched": (str(row.kind), str(row.name)) in flow_sets[10],
            "in_final_scoring": key in scored_keys,
        }
        if history is not None and not history.empty:
            h = history.copy()
            dates = pd.to_datetime(h["date"], errors="coerce").dropna().sort_values()
            record["history_rows"] = len(h)
            if "data_source" in h and h["data_source"].notna().any():
                record["history_source"] = str(h["data_source"].dropna().iloc[-1])
            record["history_start"] = dates.min() if not dates.empty else pd.NaT
            record["history_end"] = dates.max() if not dates.empty else pd.NaT
            record["duplicate_dates"] = int(dates.duplicated().sum())
            close = pd.to_numeric(h.get("close"), errors="coerce")
            record["invalid_close_rows"] = int((close.isna() | (close <= 0)).sum())
            if {"high", "low"}.issubset(h.columns):
                high = pd.to_numeric(h["high"], errors="coerce")
                low = pd.to_numeric(h["low"], errors="coerce")
                record["invalid_ohlc_rows"] = int(
                    ((high < low) | (close > high) | (close < low)).fillna(False).sum()
                )
            ref = reference_calendar.get(str(row.kind), pd.DatetimeIndex([]))
            if len(ref) and not dates.empty:
                eligible = ref[ref >= dates.min()]
                missing = eligible.difference(pd.DatetimeIndex(dates))
                record["missing_trading_days_count"] = len(missing)
                record["missing_trading_dates"] = ",".join(d.strftime("%Y-%m-%d") for d in missing)
                record["stale_trading_days"] = int((ref > dates.max()).sum())

        if intentional_filter:
            status = "主动过滤"
        elif history is None or history.empty:
            status = "日线不足" if "不足" in error else "日线抓取失败"
        elif record["duplicate_dates"] or record["invalid_close_rows"] or record["invalid_ohlc_rows"]:
            status = "行情质量异常"
        elif record["stale_trading_days"] and record["stale_trading_days"] > 0:
            status = "行情未到最新日"
        elif record["missing_trading_days_count"]:
            status = "行情日期缺口"
        elif key not in scored_keys:
            status = "未进入评分"
        elif not all(record[f"flow_{w}d_matched"] for w in (1, 5, 10)):
            status = "资金流部分缺失"
        else:
            status = "完整"
        record["audit_status"] = status
        record["is_omitted"] = key not in scored_keys
        records.append(record)

    audit = pd.DataFrame(records).sort_values(
        ["is_omitted", "audit_status", "kind", "name"],
        ascending=[False, True, True, True],
    )
    summary_rows: list[dict[str, object]] = []
    for kind in [*source["kind"].unique(), "all"]:
        part = audit if kind == "all" else audit[audit["kind"] == kind]
        target = part[~part["intentional_filter"]]
        summary_rows.append({
            "kind": kind,
            "source_universe": len(part),
            "intentional_filtered": int(part["intentional_filter"].sum()),
            "scan_target": len(target),
            "history_success": int((target["history_rows"] > 0).sum()),
            "final_scored": int(target["in_final_scoring"].sum()),
            "omitted_from_scoring": int(target["is_omitted"].sum()),
            "target_coverage_pct": round(float(target["in_final_scoring"].mean() * 100), 2) if len(target) else np.nan,
            "flow_1d_match_pct": round(float(target["flow_1d_matched"].mean() * 100), 2) if len(target) else np.nan,
            "flow_5d_match_pct": round(float(target["flow_5d_matched"].mean() * 100), 2) if len(target) else np.nan,
            "flow_10d_match_pct": round(float(target["flow_10d_matched"].mean() * 100), 2) if len(target) else np.nan,
            "quality_warning_count": int(target["audit_status"].isin([
                "行情质量异常", "行情未到最新日", "行情日期缺口", "资金流部分缺失",
            ]).sum()),
        })
    return audit, pd.DataFrame(summary_rows)
