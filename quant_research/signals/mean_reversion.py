"""
Mean Reversion Signals

Implements mean-reversion based trading signals.
"""

from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

from quant_research.signals.base import SignalBase


class MeanReversionSignal(SignalBase):
    """
    Mean reversion signal based on deviation from moving average.
    
    Hypothesis: Assets that have deviated significantly from their 
    recent average will revert back toward the mean.
    
    Attributes:
        window: Lookback window for computing the moving average
        method: Method for computing deviation ('zscore', 'percentile', 'distance')
    """
    
    def __init__(
        self,
        window: int = 21,
        method: str = 'zscore',
        name: str = "mean_reversion",
    ) -> None:
        """
        Initialize the mean reversion signal.
        
        Args:
            window: Window for moving average calculation
            method: Method for computing deviation
            name: Signal name
        """
        super().__init__(name=name, window=window, method=method)
        self.window = window
        self.method = method
    
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute mean reversion signal.
        
        Negative values indicate price is above average (short signal).
        Positive values indicate price is below average (long signal).
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            
        Returns:
            DataFrame of signal values
        """
        if self.method == 'zscore':
            # Z-score of price relative to rolling mean
            rolling_mean = prices.rolling(window=self.window).mean()
            rolling_std = prices.rolling(window=self.window).std()
            
            signal = -(prices - rolling_mean) / rolling_std.replace(0, np.nan)
            
        elif self.method == 'percentile':
            # Percentile rank of price in rolling window
            signal = prices.rolling(window=self.window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) 
                if x.max() > x.min() else 0.5,
                raw=False,
            )
            # Invert so high percentile = short signal
            signal = 0.5 - signal
            
        elif self.method == 'distance':
            # Simple distance from moving average
            rolling_mean = prices.rolling(window=self.window).mean()
            signal = -(prices - rolling_mean) / rolling_mean
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return signal


class RSI_Signal(SignalBase):
    """
    Relative Strength Index (RSI) based mean reversion signal.
    
    Hypothesis: Overbought (RSI > 70) and oversold (RSI < 30) conditions
    tend to reverse.
    
    Attributes:
        window: Window for RSI calculation (typically 14)
        upper_threshold: Overbought threshold (typically 70)
        lower_threshold: Oversold threshold (typically 30)
    """
    
    def __init__(
        self,
        window: int = 14,
        upper_threshold: float = 70,
        lower_threshold: float = 30,
        name: str = "rsi",
    ) -> None:
        """
        Initialize RSI signal.
        
        Args:
            window: RSI lookback window
            upper_threshold: Overbought threshold
            lower_threshold: Oversold threshold
            name: Signal name
        """
        super().__init__(
            name=name,
            window=window,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        self.window = window
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
    
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute RSI-based signal.
        
        Returns negative values when RSI is high (overbought),
        positive when RSI is low (oversold).
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            
        Returns:
            DataFrame of signal values
        """
        rsi = self._compute_rsi(prices)
        
        # Center around 50 and invert
        # RSI > 50 -> negative signal (potential short)
        # RSI < 50 -> positive signal (potential long)
        signal = (50 - rsi) / 50
        
        return signal
    
    def _compute_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI for each asset.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            DataFrame of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Rolling average of gains and losses
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class BollingerBandsSignal(SignalBase):
    """
    Bollinger Bands based mean reversion signal.
    
    Hypothesis: Prices near the upper band tend to revert down,
    prices near the lower band tend to revert up.
    
    Attributes:
        window: Moving average window
        std_dev: Number of standard deviations for bands
    """
    
    def __init__(
        self,
        window: int = 20,
        std_dev: float = 2.0,
        name: str = "bollinger_bands",
    ) -> None:
        """
        Initialize Bollinger Bands signal.
        
        Args:
            window: Moving average window
            std_dev: Standard deviations for band width
            name: Signal name
        """
        super().__init__(name=name, window=window, std_dev=std_dev)
        self.window = window
        self.std_dev = std_dev
    
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute Bollinger Bands signal.
        
        Signal is normalized position within the bands:
        -1 = at upper band (short signal)
         0 = at middle band
        +1 = at lower band (long signal)
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            
        Returns:
            DataFrame of signal values
        """
        # Calculate Bollinger Bands
        middle = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        # Normalize position within bands
        band_width = upper - lower
        signal = (upper - prices) / band_width.replace(0, np.nan)
        
        # Scale to [-1, 1]
        signal = 2 * signal - 1
        
        return signal
