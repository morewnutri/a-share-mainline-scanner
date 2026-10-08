from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from .trading_calendar import is_market_session


KEYS = ["kind", "code", "history_source"]
MODEL_VERSION = "quant-v2"


@dataclass(frozen=True)
class SnapshotRef:
    path: Path
    captured_at: pd.Timestamp


class SnapshotStore:
    """持久化评分时间序列，并给当前结果补充排名、广度和成交额份额变化。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _timestamp_from_path(path: Path) -> pd.Timestamp | None:
        try:
            stem = path.name.removesuffix(".csv.gz")
            return pd.Timestamp(datetime.strptime(stem, "%Y-%m-%d_%H%M%S"))
        except ValueError:
            return None

    def list(self) -> list[SnapshotRef]:
        refs = []
        for path in self.root.glob("*.csv.gz") if self.root.exists() else []:
            captured_at = self._timestamp_from_path(path)
            if captured_at is not None:
                refs.append(SnapshotRef(path, captured_at))
        return sorted(refs, key=lambda item: item.captured_at)

    def daily_closes(self) -> list[pd.DataFrame]:
        """One completed close per actual market session and model version."""
        selected: dict[tuple[pd.Timestamp, str, str, str], tuple[SnapshotRef, pd.DataFrame]] = {}
        for ref in self.list():
            frame = self._read(ref)
            if frame.empty or str(frame["run_mode"].iloc[0]) != "close":
                continue
            day = pd.Timestamp(frame["market_as_of"].max()).normalize()
            key = (day, str(frame["model_version"].iloc[0]), str(frame["config_hash"].iloc[0]),
                   str(frame["universe_hash"].iloc[0]))
            selected[key] = (ref, frame)
        if not selected:
            return []
        latest_model = max(selected.items(), key=lambda item: item[1][0].captured_at)[0][1:]
        return [selected[key][1] for key in sorted(selected) if key[1:] == latest_model]

    def save(self, scored: pd.DataFrame, captured_at: datetime | pd.Timestamp | None = None) -> Path:
        captured = pd.Timestamp(captured_at or datetime.now()).floor("s")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{captured:%Y-%m-%d_%H%M%S}.csv.gz"
        out = scored.copy()
        out["captured_at"] = captured
        out["market_as_of"] = pd.to_datetime(out.get("as_of", captured)).dt.normalize() if "as_of" in out else captured.normalize()
        if out["market_as_of"].nunique() != 1 or not is_market_session(pd.Timestamp(out["market_as_of"].iloc[0])):
            raise ValueError("快照必须对应同一个真实交易日")
        out["history_as_of"] = out["market_as_of"]
        out["run_mode"] = out.get("run_mode", "close")
        out["model_version"] = out.get("model_version", MODEL_VERSION)
        out["config_hash"] = out.get("config_hash", MODEL_VERSION)
        source = out["history_source"].astype(str) if "history_source" in out else pd.Series("来源未标记", index=out.index)
        universe = "|".join(sorted(out["kind"].astype(str) + ":" + out["code"].astype(str) + ":" + source))
        out["universe_hash"] = hashlib.sha256(universe.encode()).hexdigest()[:16]
        fingerprint_cols = [c for c in ("kind", "code", "market_as_of", "last_close", "last_amount", "history_source") if c in out]
        payload = pd.util.hash_pandas_object(out[fingerprint_cols], index=False).values.tobytes()
        out["data_fingerprint"] = hashlib.sha256(payload).hexdigest()[:16]
        out.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        return path

    @staticmethod
    def _read(ref: SnapshotRef) -> pd.DataFrame:
        # Numeric-looking board codes must remain text across sessions. Otherwise
        # pandas infers int64 here and the next scan's string keys cannot merge.
        frame = pd.read_csv(ref.path, encoding="utf-8-sig", dtype={"kind": str, "code": str})
        frame["captured_at"] = pd.to_datetime(frame.get("captured_at", ref.captured_at), errors="coerce")
        frame["market_as_of"] = pd.to_datetime(frame.get("market_as_of", frame.get("as_of", ref.captured_at)), errors="coerce").dt.normalize() if "market_as_of" in frame or "as_of" in frame else pd.Timestamp(ref.captured_at).normalize()
        frame["model_version"] = frame.get("model_version", "legacy")
        frame["config_hash"] = frame.get("config_hash", frame["model_version"])
        frame["universe_hash"] = frame.get("universe_hash", "legacy")
        frame["history_source"] = frame.get("history_source", "来源未标记")
        frame["run_mode"] = frame.get("run_mode", "close")
        return frame

    def _reference_frames(self, now: pd.Timestamp, market_as_of: pd.Timestamp, model_version: str, config_hash: str, universe_hash: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        refs = [ref for ref in self.list() if ref.captured_at < now]
        if not refs:
            return None, None, None
        intraday = None
        close_by_date: dict[object, SnapshotRef] = {}
        for ref in refs:
            frame = self._read(ref)
            if (frame.empty or str(frame["model_version"].iloc[0]) != model_version
                    or str(frame["config_hash"].iloc[0]) != config_hash
                    or str(frame["universe_hash"].iloc[0]) != universe_hash):
                continue
            trade_date = pd.Timestamp(frame["market_as_of"].max()).normalize()
            if trade_date == market_as_of and str(frame["run_mode"].iloc[0]) == "intraday":
                intraday = ref
            elif trade_date < market_as_of and str(frame["run_mode"].iloc[0]) == "close":
                close_by_date[trade_date] = ref
        closes = [close_by_date[day] for day in sorted(close_by_date)]
        prior_1d = closes[-1] if closes else None
        prior_3d = closes[-3] if len(closes) >= 3 else None
        return (
            self._read(intraday) if intraday else None,
            self._read(prior_1d) if prior_1d else None,
            self._read(prior_3d) if prior_3d else None,
        )

    @staticmethod
    def _add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "kind" in out:
            out["kind"] = out["kind"].astype(str)
        if "code" in out:
            out["code"] = out["code"].astype(str)
        if "history_source" not in out:
            out["history_source"] = "来源未标记"
        for score in ("mainline_score", "confirmation_score", "candidate_score", "ignition_score",
                      "market_confirmation_score", "exhaustion_score"):
            if score in out:
                out[f"{score}_rank"] = out.groupby(["kind", "history_source"])[score].rank(method="min", ascending=False)
        return out

    @staticmethod
    def _merge_delta(current: pd.DataFrame, previous: pd.DataFrame | None, suffix: str) -> pd.DataFrame:
        if previous is None or previous.empty:
            return current
        previous = SnapshotStore._add_ranks(previous)
        value_cols = [
            col for col in (
                "mainline_score", "confirmation_score", "candidate_score", "ignition_score",
                "market_confirmation_score", "exhaustion_score",
                "breadth", "amount_share", "turnover_share", "rs_market_5d",
                "mainline_score_rank", "confirmation_score_rank", "candidate_score_rank", "ignition_score_rank",
                "market_confirmation_score_rank", "exhaustion_score_rank",
            ) if col in previous and col in current
        ]
        prior = previous[KEYS + value_cols].drop_duplicates(KEYS).rename(
            columns={col: f"{col}_prev_{suffix}" for col in value_cols}
        )
        out = current.merge(prior, on=KEYS, how="left")
        for col in value_cols:
            old = f"{col}_prev_{suffix}"
            if col.endswith("_rank"):
                # 排名数值越小越强，因此 previous-current 为正代表排名跃迁。
                out[f"{col}_velocity_{suffix}"] = out[old] - out[col]
            else:
                out[f"{col}_delta_{suffix}"] = pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(out[old], errors="coerce")
        return out

    def enrich(self, scored: pd.DataFrame, captured_at: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
        now = pd.Timestamp(captured_at or datetime.now())
        out = self._add_ranks(scored)
        dates = pd.to_datetime(out.get("as_of", pd.Series(now, index=out.index)), errors="coerce").dt.normalize()
        if dates.nunique() != 1:
            raise ValueError("快照板块行情截止交易日不一致")
        market_as_of = dates.iloc[0]
        if not is_market_session(market_as_of):
            raise ValueError("行情截止日不是交易日")
        model_version = str(out.get("model_version", pd.Series(MODEL_VERSION, index=out.index)).iloc[0])
        config_hash = str(out.get("config_hash", pd.Series(MODEL_VERSION, index=out.index)).iloc[0])
        universe = "|".join(sorted(out["kind"].astype(str) + ":" + out["code"].astype(str)
                                   + ":" + out["history_source"].astype(str)))
        universe_hash = hashlib.sha256(universe.encode()).hexdigest()[:16]
        intraday, prior_1d, prior_3d = self._reference_frames(now, market_as_of, model_version, config_hash, universe_hash)
        out["market_as_of"] = market_as_of
        out["model_version"] = model_version
        out["config_hash"] = config_hash
        out["universe_hash"] = universe_hash
        out = self._merge_delta(out, intraday, "intraday")
        out = self._merge_delta(out, prior_1d, "1d")
        out = self._merge_delta(out, prior_3d, "3d")
        history_cols = [col for col in out if col.endswith(("_delta_1d", "_velocity_1d"))]
        out["snapshot_history_coverage"] = (
            out[history_cols].notna().mean(axis=1) if history_cols else np.zeros(len(out), dtype=float)
        )
        return out


def add_amount_share(metrics: pd.DataFrame) -> pd.DataFrame:
    """计算同类别内板块成交额份额；概念有重叠，适合看自身时间变化而非绝对占比。"""
    out = metrics.copy()
    if "snapshot_amount" in out:
        amount = pd.to_numeric(out["snapshot_amount"], errors="coerce")
        if "last_amount" in out:
            amount = amount.fillna(pd.to_numeric(out["last_amount"], errors="coerce"))
    elif "last_amount" in out:
        amount = pd.to_numeric(out["last_amount"], errors="coerce")
    elif "amount" in out:
        amount = pd.to_numeric(out["amount"], errors="coerce")
    else:
        out["amount_share"] = np.nan
        return out
    amount = amount.clip(lower=0)
    sources = out["history_source"] if "history_source" in out else pd.Series("来源未标记", index=out.index)
    denominator = amount.groupby([out["kind"], sources]).transform("sum").replace(0, np.nan)
    out["amount_share"] = amount / denominator
    return out
