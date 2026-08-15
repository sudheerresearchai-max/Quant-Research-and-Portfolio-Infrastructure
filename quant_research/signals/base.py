"""
Base Signal Class

Abstract base class for all trading signals.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class SignalBase(ABC):
    """
    Abstract base class for trading signals.
    
    All signal classes should inherit from this base class and implement
    the compute() method.
    
    Attributes:
        name: Name of the signal
        params: Dictionary of signal parameters
    """
    
    def __init__(self, name: str = "base_signal", **params) -> None:
        """
        Initialize the signal.
        
        Args:
            name: Descriptive name for the signal
            **params: Signal-specific parameters
        """
        self.name = name
        self.params = params
    
    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute the signal values.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame of signal values (dates x tickers)
        """
        pass
    
    def standardize(self, signals: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Standardize signal values.
        
        Args:
            signals: Raw signal DataFrame
            method: Standardization method ('zscore', 'rank', 'winsorize')
            
        Returns:
            Standardized signal DataFrame
        """
        if method == 'zscore':
            # Cross-sectional z-score at each date
            mean = signals.mean(axis=1)
            std = signals.std(axis=1)
            standardized = signals.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
            
        elif method == 'rank':
            # Cross-sectional rank at each date
            standardized = signals.rank(axis=1, pct=True)
            standardized = (standardized - 0.5) * 2  # Scale to [-1, 1]
            
        elif method == 'winsorize':
            # Winsorize extreme values
            lower = signals.quantile(0.01, axis=1)
            upper = signals.quantile(0.99, axis=1)
            standardized = signals.clip(lower, upper, axis=0)
            standardized = self.standardize(standardized, method='zscore')
            
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        return standardized
    
    def lag(self, signals: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Lag signals to prevent look-ahead bias.
        
        Args:
            signals: Signal DataFrame
            periods: Number of periods to lag
            
        Returns:
            Lagged signal DataFrame
        """
        return signals.shift(periods)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get signal parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def set_params(self, **params) -> 'SignalBase':
        """
        Set signal parameters.
        
        Args:
            **params: Parameters to update
            
        Returns:
            Self for method chaining
        """
        self.params.update(params)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"
