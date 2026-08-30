from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

LOG = logging.getLogger(__name__)


def normalize_theme_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^[a-z]\d{2}", "", text)
    text = re.sub(r"[\s_（）()\-—·]", "", text)
    text = re.sub(r"(概念|板块)$", "", text)
    text = re.sub(r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", text)
    return text


def to_baostock_code(value: object) -> str | None:
    text = str(value).strip().lower()
    if re.fullmatch(r"(?:sh|sz|bj)\.\d{6}", text):
        return text
    digits = re.sub(r"\D", "", text)[-6:]
    if len(digits) != 6:
        return None
    if digits.startswith(("6", "68")):
        return f"sh.{digits}"
    if digits.startswith(("0", "3")):
        return f"sz.{digits}"
    if digits.startswith(("4", "8", "92")):
        return f"bj.{digits}"
    return None


class BaoStockSyntheticProvider:
    """用 BaoStock 个股日线合成等权板块指数。

    BaoStock 没有东方财富概念指数，因此该结果只作为末级回退，并在 data_source 中
    明确标记为合成数据。默认优先用 BaoStock 行业分类；概念需要外部成分股解析器。
    """

    def __init__(
        self,
        cache_dir: Path,
        constituent_resolver: Callable[[str, str], Iterable[str]] | None = None,
        *,
        max_constituents: int = 24,
        min_constituents: int = 3,
    ):
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("启用 BaoStock 回退需要安装 baostock>=0.9.3,<1") from exc
        self.bs = bs
        self.cache_dir = Path(cache_dir) / "baostock"
        self.constituent_resolver = constituent_resolver
        self.max_constituents = max(1, int(max_constituents))
        self.min_constituents = max(1, int(min_constituents))
        self._lock = threading.Lock()
        self._industry_members: dict[str, list[str]] | None = None

    @staticmethod
    def _rows(result: object) -> pd.DataFrame:
        if getattr(result, "error_code", "1") != "0":
            raise RuntimeError(getattr(result, "error_msg", "BaoStock 返回错误"))
        fields = list(getattr(result, "fields", []))
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=fields)

    def _login(self) -> None:
        result = self.bs.login()
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {result.error_msg}")

    def _load_industry_members(self) -> dict[str, list[str]]:
        if self._industry_members is not None:
            return self._industry_members
        frame = self._rows(self.bs.query_stock_industry())
        mapping: dict[str, list[str]] = {}
        if {"industry", "code"}.issubset(frame.columns):
            for name, group in frame.dropna(subset=["industry", "code"]).groupby("industry"):
                mapping[normalize_theme_name(name)] = group["code"].astype(str).tolist()
        self._industry_members = mapping
        return mapping

    def _members(self, kind: str, name: str) -> list[str]:
        members: list[str] = []
        if kind == "industry":
            industry_members = self._load_industry_members()
            target = normalize_theme_name(name)
            members = industry_members.get(target, [])
            if not members and len(target) >= 2:
                candidates = [key for key in industry_members if target in key]
                if candidates:
                    members = industry_members[min(candidates, key=len)]
        if not members and self.constituent_resolver is not None:
            members = list(self.constituent_resolver(kind, name))
        normalized = []
        for value in members:
            code = to_baostock_code(value)
            if code and code not in normalized:
                normalized.append(code)
        # 固定抽样保证跨次扫描可复现；共享个股缓存会显著降低重叠概念的请求量。
        return sorted(normalized)[: self.max_constituents]

    def _stock_path(self, code: str) -> Path:
        return self.cache_dir / "stocks" / f"{code.replace('.', '_')}.csv"

    def _stock_history(self, code: str, start: str, end: str) -> pd.DataFrame:
        path = self._stock_path(code)
        if path.exists():
            cached = pd.read_csv(path, encoding="utf-8-sig")
            dates = pd.to_datetime(cached.get("date"), errors="coerce")
            written_today = datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()
            if not dates.dropna().empty and (dates.max().date() >= pd.Timestamp(end).date() or written_today):
                return cached
        result = self.bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn,tradestatus",
            start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        frame = self._rows(result)
        if frame.empty:
            return frame
        frame = frame[frame.get("tradestatus", "1").astype(str) == "1"].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume", "amount", "turn"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame

    def _synthesize(self, histories: list[pd.DataFrame]) -> pd.DataFrame:
        pieces = []
        for index, history in enumerate(histories):
            h = history.set_index("date").sort_index()
            previous_close = h["close"].shift(1)
            piece = pd.DataFrame(index=h.index)
            for col in ("open", "high", "low", "close"):
                piece[f"{col}_{index}"] = h[col] / previous_close - 1
            piece[f"amount_{index}"] = h["amount"]
            piece[f"volume_{index}"] = h["volume"]
            piece[f"turn_{index}"] = h["turn"]
            pieces.append(piece)
        merged = pd.concat(pieces, axis=1).sort_index()
        close_returns = merged.filter(regex=r"^close_\d+$")
        available = close_returns.notna().sum(axis=1)
        valid = available >= min(self.min_constituents, len(histories))
        mean_close_return = close_returns.mean(axis=1).where(valid)
        close = (1 + mean_close_return.fillna(0)).cumprod() * 100
        previous = close.shift(1).fillna(100)
        out = pd.DataFrame({"date": merged.index, "close": close.values})
        for col in ("open", "high", "low"):
            relative = merged.filter(regex=rf"^{col}_\d+$").mean(axis=1).where(valid)
            out[col] = (previous * (1 + relative.fillna(mean_close_return))).values
        out["high"] = out[["open", "high", "close"]].max(axis=1)
        out["low"] = out[["open", "low", "close"]].min(axis=1)
        out["amount"] = merged.filter(regex=r"^amount_\d+$").sum(axis=1, min_count=1).values
        out["volume"] = merged.filter(regex=r"^volume_\d+$").sum(axis=1, min_count=1).values
        out["turnover"] = merged.filter(regex=r"^turn_\d+$").mean(axis=1).values
        out["constituent_count"] = available.values
        out["synthetic_coverage"] = (available / len(histories)).values
        out["data_source"] = "BaoStock成分股等权合成"
        return out.loc[valid.values].reset_index(drop=True)

    def get_history(self, kind: str, name: str, start: str, end: str) -> pd.DataFrame:
        with self._lock:
            self._login()
            try:
                members = self._members(kind, name)
                if len(members) < self.min_constituents:
                    raise LookupError(f"BaoStock/成分股映射不足: {name} 仅 {len(members)} 只")
                histories = []
                for code in members:
                    try:
                        frame = self._stock_history(code, start, end)
                        if len(frame) >= 12:
                            histories.append(frame)
                    except Exception as exc:
                        LOG.debug("BaoStock 个股 %s 获取失败: %s", code, exc)
                if len(histories) < self.min_constituents:
                    raise RuntimeError(f"BaoStock 有效成分股不足: {name} {len(histories)}/{len(members)}")
                return self._synthesize(histories)
            finally:
                self.bs.logout()
