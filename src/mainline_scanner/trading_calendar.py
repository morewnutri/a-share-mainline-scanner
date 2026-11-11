"""Completed Shanghai market sessions for close-only scans."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import exchange_calendars as xcals


def is_market_session(day: pd.Timestamp) -> bool:
    return bool(xcals.get_calendar("XSHG").is_session(pd.Timestamp(day).normalize()))


def expected_close_session(end: str | pd.Timestamp, now: datetime | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(end).normalize()
    china_now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if china_now.tzinfo is not None:
        china_now = china_now.astimezone(ZoneInfo("Asia/Shanghai"))
    if stamp.date() >= china_now.date() and china_now.time() < time(15, 15):
        stamp = pd.Timestamp(china_now.date() - timedelta(days=1))
    calendar = xcals.get_calendar("XSHG")
    sessions = calendar.sessions_in_range(stamp - pd.Timedelta(days=35), stamp)
    if sessions.empty:
        raise ValueError(f"无法确定 {stamp.date()} 前的交易日")
    return pd.Timestamp(sessions[-1]).normalize()
