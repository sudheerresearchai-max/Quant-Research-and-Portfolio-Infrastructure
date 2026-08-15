"""
Momentum Signals

Implements various momentum-based trading signals including:
- Time-series momentum (trend following)
- Cross-sectional momentum (relative strength)
"""

from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

from quant_research.signals.base import SignalBase


class MomentumSignal(SignalBase):
    """
    Time-series momentum signal.
    
    Computes the return over a lookback period, optionally with a skip period
    to avoid short-term reversal effects.
    
    Hypothesis: Assets that have performed well (poorly) in the past will
    continue to perform well (poorly) in the near future.
    
    Attributes:
        lookback: Lookback period in days (e.g., 252 for 1-year momentum)
        skip: Skip period to exclude recent returns (e.g., 21 for 1 month)
    """
    
    def __init__(
        self,
        lookback: int = 252,
        skip: int = 0,
        name: str = "momentum",
    ) -> None:
        """
        Initialize the momentum signal.
        
        Args:
            lookback: Lookback period in days
            skip: Number of recent days to exclude
            name: Signal name
        """
        super().__init__(name=name, lookback=lookback, skip=skip)
        self.lookback = lookback
        self.skip = skip
    
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute momentum signal as past returns.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            
        Returns:
            DataFrame of momentum values
        """
        # Calculate total return over lookback period
        if self.skip == 0:
            momentum = prices.pct_change(periods=self.lookback)
        else:
            # Price 'lookback + skip' days ago
            price_past = prices.shift(self.skip).pct_change(periods=self.lookback)
            momentum = price_past
        
        return momentum
    
    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum using log returns (alternative formulation).
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            DataFrame of log momentum values
        """
        log_prices = np.log(prices)
        
        if self.skip == 0:
            momentum = log_prices.diff(periods=self.lookback)
        else:
            momentum = log_prices.shift(self.skip).diff(periods=self.lookback)
        
        return momentum


class CrossSectionalMomentum(SignalBase):
    """
    Cross-sectional momentum signal.
    
    Ranks assets by their past performance and goes long top performers
    and short bottom performers.
    
    Hypothesis: Relative winners will continue to outperform relative losers.
    
    Attributes:
        lookback: Lookback period in days
        skip: Skip period
        method: Ranking method ('percentile', 'zscore', 'rank')
    """
    
    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        method: str = 'percentile',
        name: str = "cross_sectional_momentum",
    ) -> None:
        """
        Initialize cross-sectional momentum signal.
        
        Args:
            lookback: Lookback period in days
            skip: Skip period to avoid short-term reversal
            method: Method for computing cross-sectional signal
            name: Signal name
        """
        super().__init__(
            name=name,
            lookback=lookback,
            skip=skip,
            method=method,
        )
        self.lookback = lookback
        self.skip = skip
        self.method = method
        
        # Internal time-series momentum calculator
        self._ts_momentum = MomentumSignal(lookback=lookback, skip=skip)
    
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute cross-sectional momentum signal.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            
        Returns:
            DataFrame of standardized signal values
        """
        # First compute time-series momentum for each asset
        ts_mom = self._ts_momentum.compute(prices)
        
        # Then standardize cross-sectionally
        if self.method == 'percentile':
            # Rank to percentile [0, 1], then center to [-0.5, 0.5]
            signal = ts_mom.rank(axis=1, pct=True) - 0.5
        elif self.method == 'zscore':
            # Z-score across assets at each date
            mean = ts_mom.mean(axis=1)
            std = ts_mom.std(axis=1)
            signal = ts_mom.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
        elif self.method == 'rank':
            # Simple rank
            n_assets = ts_mom.shape[1]
            signal = ts_mom.rank(axis=1) - (n_assets + 1) / 2
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return signal


class ResidualMomentum(SignalBase):
    """
    Residual momentum signal (momentum orthogonalized to market).
    
    Computes momentum of residuals from a market model regression,
    capturing idiosyncratic momentum.
    
    Hypothesis: Idiosyncratic trends persist more than market-driven moves.
    
    Attributes:
        lookback: Lookback period for momentum
        window: Rolling window for beta estimation
        market_col: Column name or series for market returns
    """
    
    def __init__(
        self,
        lookback: int = 252,
        window: int = 63,
        market_col: Optional[str] = None,
        name: str = "residual_momentum",
    ) -> None:
        """
        Initialize residual momentum signal.
        
        Args:
            lookback: Lookback period in days
            window: Rolling window for beta estimation
            market_col: Market returns column name or None for equal-weighted market
            name: Signal name
        """
        super().__init__(
            name=name,
            lookback=lookback,
            window=window,
            market_col=market_col,
        )
        self.lookback = lookback
        self.window = window
        self.market_col = market_col
    
    def compute(self, prices: pd.DataFrame, market_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compute residual momentum signal.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            market_returns: Optional market returns series
            
        Returns:
            DataFrame of residual momentum values
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Get market returns
        if market_returns is None:
            if self.market_col and self.market_col in returns.columns:
                market_returns = returns[self.market_col]
            else:
                # Use equal-weighted market return
                market_returns = returns.mean(axis=1)
        
        # Compute rolling betas and residuals
        residual_momentum = pd.DataFrame(
            index=returns.index,
            columns=returns.columns,
            dtype=float,
        )
        
        for ticker in returns.columns:
            asset_returns = returns[ticker]
            
            # Rolling regression to get beta
            betas = pd.Series(index=returns.index, dtype=float)
            
            for t in range(self.window, len(returns)):
                y = asset_returns.iloc[t-self.window:t]
                x = market_returns.iloc[t-self.window:t]
                
                # Simple OLS: beta = cov(x,y) / var(x)
                cov_xy = np.cov(x, y)[0, 1]
                var_x = np.var(x)
                
                if var_x > 0:
                    beta = cov_xy / var_x
                else:
                    beta = 1.0
                
                betas.iloc[t] = beta
            
            # Compute residuals
            residuals = asset_returns - betas * market_returns
            
            # Cumulative residual return over lookback
            residual_momentum[ticker] = residuals.rolling(self.lookback).sum()
        
        return residual_momentum
