"""Signals module for computing and testing trading signals."""

from quant_research.signals.base import SignalBase
from quant_research.signals.momentum import MomentumSignal, CrossSectionalMomentum
from quant_research.signals.mean_reversion import MeanReversionSignal
from quant_research.signals.statistical_tests import (
    test_signal_significance,
    newey_west_tstat,
    compute_ic,
)

__all__ = [
    "SignalBase",
    "MomentumSignal",
    "CrossSectionalMomentum",
    "MeanReversionSignal",
    "test_signal_significance",
    "newey_west_tstat",
    "compute_ic",
]
