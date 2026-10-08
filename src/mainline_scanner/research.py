"""Compatibility imports for objective market metrics.

Research observations are intentionally excluded from automatic scoring.
"""
from .market_metrics import load_market, enrich_market_history

__all__ = ["load_market", "enrich_market_history"]
