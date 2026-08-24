from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(value))


def _find_col(columns: Iterable[str], *needles: str) -> str | None:
    cols = [str(c) for c in columns]
    for needle in needles:
        if needle in cols:
            return needle
    for needle in needles:
        for col in cols:
            if needle in col:
                return col
    return None


@dataclass
class FetchResult:
    histories: dict[tuple[str, str], pd.DataFrame]
    failures: list[dict[str, str]]


class EastmoneyAkshareProvider:
    """通过 AKShare 获取东方财富行业/概念数据，并做本地缓存。"""

    def __init__(self, cache_dir: Path, refresh: bool = False, ttl_hours: float = 8):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.ttl = timedelta(hours=ttl_hours)
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("缺少 akshare，请先执行 pip install -e .") from exc
        self.ak = ak
        self._eastmoney_history_available: bool | None = None
        self._fallback_lock = threading.Lock()
        self._ths_maps: dict[str, dict[str, str]] = {}

    def _fresh(self, path: Path) -> bool:
        if self.refresh or not path.exists():
            return False
        return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) <= self.ttl

    def _read_cache(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig")

    def _write_cache(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def _eastmoney_json(self, endpoint: str, params: dict, timeout: int = 15) -> dict:
        """优先使用延迟行情域名；部分网络无法访问数字分片域名。"""
        hosts = ["https://push2delay.eastmoney.com", "https://push2.eastmoney.com"]
        last_error: Exception | None = None
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        for host in hosts:
            try:
                response = requests.get(host + endpoint, params=params, headers=headers, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("data") is not None:
                    return payload
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"东方财富域名均不可用: {last_error}")

    def _direct_universe(self, kind: str) -> pd.DataFrame:
        fs = "m:90 t:2 f:!50" if kind == "industry" else "m:90 t:3 f:!50"
        fields = "f2,f3,f4,f8,f12,f14,f20,f104,f105,f128,f136"
        rows: list[dict] = []
        page = 1
        while True:
            params = {
                "pn": page, "pz": 100, "po": 1, "np": 1,
                "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": 2, "invt": 2,
                "fid": "f3", "fs": fs, "fields": fields,
            }
            payload = self._eastmoney_json("/api/qt/clist/get", params)
            data = payload["data"]
            rows.extend(data.get("diff") or [])
            if len(rows) >= int(data.get("total") or 0) or not data.get("diff"):
                break
            page += 1
        raw = pd.DataFrame(rows)
        raw.rename(columns={
            "f14": "板块名称", "f12": "板块代码", "f2": "最新价", "f4": "涨跌额",
            "f3": "涨跌幅", "f20": "总市值", "f8": "换手率", "f104": "上涨家数",
            "f105": "下跌家数", "f128": "领涨股票", "f136": "领涨股票-涨跌幅",
        }, inplace=True)
        raw.insert(0, "排名", range(1, len(raw) + 1))
        return raw

    def _direct_history(self, code: str, start: str, end: str) -> pd.DataFrame:
        params = {
            "secid": f"90.{code}", "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101", "fqt": "0", "beg": start, "end": end,
            "smplmt": "10000", "lmt": "1000000",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        last_error: Exception | None = None
        for host in ["https://push2his.eastmoney.com", "https://7.push2his.eastmoney.com", "https://91.push2his.eastmoney.com"]:
            try:
                response = requests.get(host + "/api/qt/stock/kline/get", params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json().get("data")
                if data is None:
                    continue
                lines = data.get("klines") or []
                columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
                result = pd.DataFrame([line.split(",") for line in lines], columns=columns)
                result["数据源"] = "东方财富"
                return result
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"板块 {code} 日线域名均不可用: {last_error}")

    @staticmethod
    def _normalize_board_name(name: str) -> str:
        text = str(name).strip().lower()
        text = re.sub(r"[\s_（）()\-—·]", "", text)
        text = re.sub(r"(概念|板块)$", "", text)
        text = re.sub(r"[ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ]+$", "", text)
        return text

    def _load_ths_map(self, kind: str) -> dict[str, str]:
        """只加载一次同花顺名称-代码映射，供东方财富日线失败时使用。"""
        if kind in self._ths_maps:
            return self._ths_maps[kind]
        with self._fallback_lock:
            if kind in self._ths_maps:
                return self._ths_maps[kind]
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://q.10jqka.com.cn/"}
            url = "https://q.10jqka.com.cn/thshy/" if kind == "industry" else "https://q.10jqka.com.cn/gn/detail/code/307822/"
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, features="lxml")
            container = soup.find("div", attrs={"class": "cate_inner"})
            mapping: dict[str, str] = {}
            if container is not None:
                for item in container.find_all("a"):
                    href = item.get("href", "")
                    match = re.search(r"/code/(\d+)/", href)
                    if match and item.get_text(strip=True):
                        mapping[item.get_text(strip=True)] = match.group(1)
            if kind == "concept":
                try:
                    extra = self.ak.stock_board_concept_name_ths()
                    if {"name", "code"}.issubset(extra.columns):
                        mapping.update(dict(zip(extra["name"].astype(str), extra["code"].astype(str))))
                except Exception as exc:
                    LOG.warning("同花顺新增概念目录补充失败，使用静态目录: %s", exc)
            normalized = {self._normalize_board_name(name): code for name, code in mapping.items()}
            self._ths_maps[kind] = normalized
            LOG.info("同花顺备用目录已加载: %s %d 个", kind, len(normalized))
            return normalized

    def _ths_history(self, kind: str, name: str, start: str, end: str) -> pd.DataFrame:
        """同花顺板块指数日线备用源。"""
        from bs4 import BeautifulSoup

        mapping = self._load_ths_map(kind)
        normalized_name = self._normalize_board_name(name)
        if normalized_name not in mapping:
            raise LookupError(f"同花顺无同名板块: {name}")
        outer_code = mapping[normalized_name]
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://q.10jqka.com.cn/"}
        inner_code = outer_code
        if kind == "concept":
            detail_url = f"https://q.10jqka.com.cn/gn/detail/code/{outer_code}/"
            detail = requests.get(detail_url, headers=headers, timeout=20)
            detail.raise_for_status()
            node = BeautifulSoup(detail.text, features="lxml").find("input", attrs={"id": "clid"})
            if node is not None and node.get("value"):
                inner_code = str(node["value"])

        rows: list[list[str]] = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            url = f"https://d.10jqka.com.cn/v4/line/bk_{inner_code}/01/{year}.js"
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            text = response.text
            left, right = text.find("{"), text.rfind("}")
            if left < 0 or right <= left:
                continue
            payload = json.loads(text[left : right + 1])
            for line in str(payload.get("data", "")).split(";"):
                values = line.split(",")
                if len(values) >= 7:
                    rows.append(values[:7])
        if not rows:
            raise RuntimeError(f"同花顺未返回日线: {name}")
        result = pd.DataFrame(rows, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"])
        dates = pd.to_datetime(result["日期"], format="%Y%m%d", errors="coerce")
        begin, finish = pd.to_datetime(start), pd.to_datetime(end)
        result = result.loc[dates.between(begin, finish)].copy()
        result["数据源"] = "同花顺"
        return result.reset_index(drop=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=1, max=6), reraise=True)
    def _call(self, func, **kwargs) -> pd.DataFrame:
        df = func(**kwargs)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"接口返回类型异常: {type(df)}")
        return df

    def get_universe(self, kind: str) -> pd.DataFrame:
        path = self.cache_dir / f"universe_{kind}.csv"
        if self._fresh(path):
            raw = self._read_cache(path)
        else:
            try:
                raw = self._direct_universe(kind)
            except Exception as direct_error:
                LOG.warning("直连板块列表失败，回退 AKShare: %s", direct_error)
                func = self.ak.stock_board_industry_name_em if kind == "industry" else self.ak.stock_board_concept_name_em
                raw = self._call(func)
            self._write_cache(raw, path)
        mapping = {
            _find_col(raw.columns, "板块名称", "名称"): "name",
            _find_col(raw.columns, "板块代码", "代码"): "code",
            _find_col(raw.columns, "涨跌幅"): "snapshot_return",
            _find_col(raw.columns, "换手率"): "snapshot_turnover",
            _find_col(raw.columns, "上涨家数"): "up_count",
            _find_col(raw.columns, "下跌家数"): "down_count",
            _find_col(raw.columns, "领涨股票"): "leader_stock",
        }
        mapping = {k: v for k, v in mapping.items() if k is not None}
        out = raw.rename(columns=mapping).copy()
        if "name" not in out or "code" not in out:
            raise ValueError(f"板块列表字段不符合预期: {list(raw.columns)}")
        out["kind"] = kind
        for col in ["snapshot_return", "snapshot_turnover", "up_count", "down_count"]:
            if col in out:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if {"up_count", "down_count"}.issubset(out.columns):
            denom = out["up_count"] + out["down_count"]
            out["breadth"] = out["up_count"] / denom.where(denom > 0)
        return out

    def _history_path(self, kind: str, code: str, name: str) -> Path:
        return self.cache_dir / "histories" / kind / f"{code}_{_safe_name(name)}.csv"

    def get_history(self, kind: str, code: str, name: str, start: str, end: str) -> pd.DataFrame:
        path = self._history_path(kind, code, name)
        if self._fresh(path):
            raw = self._read_cache(path)
        else:
            eastmoney_error: Exception | None = None
            raw = pd.DataFrame()
            if self._eastmoney_history_available is not False:
                try:
                    raw = self._direct_history(code, start, end)
                except Exception as exc:
                    eastmoney_error = exc
            if raw.empty:
                try:
                    raw = self._ths_history(kind, name, start, end)
                except Exception as ths_error:
                    raise RuntimeError(
                        f"东方财富失败: {eastmoney_error or '端点探测已判定不可用'}; 同花顺失败: {ths_error}"
                    ) from ths_error
            self._write_cache(raw, path)
        if raw.empty:
            return raw
        rename = {}
        aliases = {
            "date": ("日期", "date"), "open": ("开盘", "开盘价", "open"),
            "close": ("收盘", "收盘价", "close"), "high": ("最高", "最高价", "high"),
            "low": ("最低", "最低价", "low"), "pct_change": ("涨跌幅",),
            "volume": ("成交量",), "amount": ("成交额",), "turnover": ("换手率",),
            "data_source": ("数据源", "data_source"),
        }
        for target, candidates in aliases.items():
            col = _find_col(raw.columns, *candidates)
            if col:
                rename[col] = target
        out = raw.rename(columns=rename).copy()
        if not {"date", "close"}.issubset(out.columns):
            raise ValueError(f"{kind}/{name} 历史字段异常: {list(raw.columns)}")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ["open", "close", "high", "low", "pct_change", "volume", "amount", "turnover"]:
            if col in out:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    def get_histories(self, boards: pd.DataFrame, start: str, end: str, max_workers: int = 8) -> FetchResult:
        histories: dict[tuple[str, str], pd.DataFrame] = {}
        failures: list[dict[str, str]] = []
        rows = boards[["kind", "code", "name"]].to_dict("records")
        uncached = [
            r for r in rows
            if not self._fresh(self._history_path(str(r["kind"]), str(r["code"]), str(r["name"])))
        ]
        if uncached and self._eastmoney_history_available is None:
            probe = uncached[0]
            try:
                raw = self._direct_history(str(probe["code"]), start, end)
                self._eastmoney_history_available = True
                self._write_cache(raw, self._history_path(str(probe["kind"]), str(probe["code"]), str(probe["name"])))
                LOG.info("东方财富板块日线端点可用")
            except Exception as exc:
                self._eastmoney_history_available = False
                LOG.warning("东方财富板块日线整体不可用，切换同花顺备用源: %s", exc)
                for kind in boards["kind"].astype(str).unique():
                    try:
                        self._load_ths_map(kind)
                    except Exception as map_error:
                        LOG.warning("同花顺 %s 备用目录加载失败: %s", kind, map_error)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.get_history, str(r["kind"]), str(r["code"]), str(r["name"]), start, end): r
                for r in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="下载板块日线", unit="板块"):
                row = futures[future]
                key = (str(row["kind"]), str(row["code"]))
                try:
                    frame = future.result()
                    if len(frame) >= 12:
                        histories[key] = frame
                    else:
                        failures.append({**row, "error": f"有效日线不足: {len(frame)}"})
                except Exception as exc:  # 单板块失败不终止全局扫描
                    failures.append({**row, "error": str(exc)[:300]})
        return FetchResult(histories=histories, failures=failures)

    def get_fund_flows(self, kinds: Iterable[str]) -> pd.DataFrame:
        pieces = []
        for kind in kinds:
            sector_type = "行业资金流" if kind == "industry" else "概念资金流"
            for indicator, window in [("今日", 1), ("5日", 5), ("10日", 10)]:
                path = self.cache_dir / f"flow_{kind}_{window}.csv"
                try:
                    if self._fresh(path):
                        raw = self._read_cache(path)
                    else:
                        raw = self._direct_fund_flow(kind, window)
                        self._write_cache(raw, path)
                    name_col = _find_col(raw.columns, "名称")
                    amount_col = _find_col(raw.columns, "主力净流入-净额")
                    pct_col = _find_col(raw.columns, "主力净流入-净占比")
                    if not all([name_col, amount_col, pct_col]):
                        raise ValueError(f"资金流字段异常: {list(raw.columns)}")
                    x = raw[[name_col, amount_col, pct_col]].copy()
                    x.columns = ["name", f"flow_{window}d_amount", f"flow_{window}d_pct"]
                    x["kind"] = kind
                    x[f"flow_{window}d_amount"] = pd.to_numeric(x[f"flow_{window}d_amount"], errors="coerce")
                    x[f"flow_{window}d_pct"] = pd.to_numeric(x[f"flow_{window}d_pct"], errors="coerce")
                    pieces.append(x)
                except Exception as exc:
                    LOG.warning("%s %s 获取失败，将以缺失值继续: %s", sector_type, indicator, exc)
                time.sleep(0.15)
        if not pieces:
            return pd.DataFrame(columns=["kind", "name"])
        result = pieces[0]
        for piece in pieces[1:]:
            result = result.merge(piece, on=["kind", "name"], how="outer")
        return result

    def _direct_fund_flow(self, kind: str, window: int) -> pd.DataFrame:
        field_map = {
            1: ("f62", "f184", "f62", "1"),
            5: ("f164", "f165", "f164", "5"),
            10: ("f174", "f175", "f174", "10"),
        }
        amount_field, pct_field, sort_field, stat = field_map[window]
        rows: list[dict] = []
        page = 1
        while True:
            params = {
                "pn": page, "pz": 100, "po": 1, "np": 1,
                "ut": "b2884a393a59ad64002292a3e90d46a5", "fltt": 2, "invt": 2,
                "fid0": sort_field, "fid": sort_field,
                "fs": f"m:90 t:{'2' if kind == 'industry' else '3'}", "stat": stat,
                "fields": f"f12,f14,{amount_field},{pct_field}", "_": int(time.time() * 1000),
            }
            payload = self._eastmoney_json("/api/qt/clist/get", params)
            data = payload["data"]
            rows.extend(data.get("diff") or [])
            if len(rows) >= int(data.get("total") or 0) or not data.get("diff"):
                break
            page += 1
        raw = pd.DataFrame(rows).rename(columns={
            "f14": "名称", amount_field: f"{window}日主力净流入-净额",
            pct_field: f"{window}日主力净流入-净占比",
        })
        return raw
