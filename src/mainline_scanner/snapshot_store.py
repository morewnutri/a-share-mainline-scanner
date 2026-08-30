from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["kind", "code"]


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

    def save(self, scored: pd.DataFrame, captured_at: datetime | pd.Timestamp | None = None) -> Path:
        captured = pd.Timestamp(captured_at or datetime.now()).floor("s")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{captured:%Y-%m-%d_%H%M%S}.csv.gz"
        out = scored.copy()
        out["captured_at"] = captured
        out.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        return path

    @staticmethod
    def _read(ref: SnapshotRef) -> pd.DataFrame:
        frame = pd.read_csv(ref.path, encoding="utf-8-sig")
        frame["captured_at"] = pd.to_datetime(frame.get("captured_at", ref.captured_at), errors="coerce")
        return frame

    def _reference_frames(self, now: pd.Timestamp) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        refs = [ref for ref in self.list() if ref.captured_at < now]
        if not refs:
            return None, None, None
        intraday = next((ref for ref in reversed(refs) if ref.captured_at.date() == now.date()), None)
        close_by_date: dict[object, SnapshotRef] = {}
        for ref in refs:
            if ref.captured_at.date() < now.date():
                close_by_date[ref.captured_at.date()] = ref
        closes = list(close_by_date.values())
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
        for score in ("mainline_score", "confirmation_score", "candidate_score", "ignition_score"):
            if score in out:
                out[f"{score}_rank"] = out.groupby("kind")[score].rank(method="min", ascending=False)
        return out

    @staticmethod
    def _merge_delta(current: pd.DataFrame, previous: pd.DataFrame | None, suffix: str) -> pd.DataFrame:
        if previous is None or previous.empty:
            return current
        previous = SnapshotStore._add_ranks(previous)
        value_cols = [
            col for col in (
                "mainline_score", "confirmation_score", "candidate_score", "ignition_score",
                "breadth", "amount_share",
                "mainline_score_rank", "confirmation_score_rank", "candidate_score_rank", "ignition_score_rank",
            ) if col in previous
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
        intraday, prior_1d, prior_3d = self._reference_frames(now)
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
    denominator = amount.groupby(out["kind"]).transform("sum").replace(0, np.nan)
    out["amount_share"] = amount / denominator
    return out
