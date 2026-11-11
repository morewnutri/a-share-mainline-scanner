"""Static research prompts; never used by numerical scoring."""
from __future__ import annotations

import pandas as pd

ITEMS = ("产业政策及实施细则", "供需、价格或库存", "订单、产能或技术进展",
         "财报、业绩预告及预期", "事件兑现、估值拥挤或监管风险")


def attach_manual_checklist(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["人工核查提醒"] = "；".join(f"{item}：未检查" for item in ITEMS)
    return out
