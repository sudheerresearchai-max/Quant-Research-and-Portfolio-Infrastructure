"""Data pipeline module for loading and cleaning financial data."""

from quant_research.data.loader import DataLoader
from quant_research.data.corporate_actions import CorporateActionsHandler, adjust_for_corporate_actions
from quant_research.data.universe import UniverseManager

__all__ = [
    "DataLoader",
    "CorporateActionsHandler",
    "adjust_for_corporate_actions",
    "UniverseManager",
]
