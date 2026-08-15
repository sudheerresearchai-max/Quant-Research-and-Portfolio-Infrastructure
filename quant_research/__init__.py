"""
Quant Research Pipeline

A reproducible quantitative research pipeline for systematic equity/multi-asset 
strategy development, from raw data to out-of-sample performance reporting.
"""

__version__ = "1.0.0"
__author__ = "Quant Research Team"

from quant_research.data.loader import DataLoader
from quant_research.signals.momentum import MomentumSignal
from quant_research.portfolio.constructor import PortfolioConstructor
from quant_research.backtest.engine import BacktestEngine
from quant_research.analytics.metrics import PerformanceMetrics

__all__ = [
    "DataLoader",
    "MomentumSignal",
    "PortfolioConstructor",
    "BacktestEngine",
    "PerformanceMetrics",
]
