"""
Risk Controls

Implements portfolio-level risk controls including:
- Volatility targeting
- Drawdown controls
- Position limits
"""

from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class VolatilityTarget:
    """
    Scale positions to achieve target portfolio volatility.
    
    Uses rolling realized volatility to dynamically adjust exposure.
    
    Attributes:
        target_vol: Target annualized volatility
        vol_window: Window for realized vol calculation
        max_leverage: Maximum allowed leverage
        min_leverage: Minimum allowed leverage
    """
    
    def __init__(
        self,
        target_vol: float = 0.10,
        vol_window: int = 63,
        max_leverage: float = 2.0,
        min_leverage: float = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        """
        Initialize volatility target.
        
        Args:
            target_vol: Target annualized volatility (e.g., 0.10 for 10%)
            vol_window: Rolling window for realized vol (in days)
            max_leverage: Maximum gross exposure
            min_leverage: Minimum gross exposure
            annualization_factor: Trading days per year
        """
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.annualization_factor = annualization_factor
    
    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply volatility targeting to weights.
        
        Args:
            weights: Raw target weights (dates x tickers)
            returns: Asset returns (dates x tickers)
            
        Returns:
            Scaled weights
        """
        # Calculate portfolio returns with current weights
        # Use lagged weights to avoid look-ahead
        lagged_weights = weights.shift(1)
        portfolio_returns = (lagged_weights * returns).sum(axis=1)
        
        # Calculate rolling realized volatility
        realized_vol = portfolio_returns.rolling(window=self.vol_window).std()
        realized_vol_annual = realized_vol * np.sqrt(self.annualization_factor)
        
        # Calculate scaling factor
        scale = self.target_vol / realized_vol_annual.replace(0, np.nan)
        
        # Clip scaling to leverage bounds
        scale = scale.clip(self.min_leverage, self.max_leverage)
        
        # Apply scaling
        scaled_weights = weights.mul(scale, axis=0)
        
        return scaled_weights
    
    def get_scaling_factor(self, returns: pd.Series) -> pd.Series:
        """
        Get the volatility scaling factor.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Scaling factor series
        """
        realized_vol = returns.rolling(window=self.vol_window).std()
        realized_vol_annual = realized_vol * np.sqrt(self.annualization_factor)
        
        scale = self.target_vol / realized_vol_annual.replace(0, np.nan)
        return scale.clip(self.min_leverage, self.max_leverage)


class DrawdownControl:
    """
    Implement drawdown-based position reduction.
    
    Reduces exposure when portfolio experiences drawdowns beyond thresholds.
    
    Attributes:
        max_drawdown: Maximum allowable drawdown before full exit
        trigger_dd: Drawdown level at which to start reducing
        reduction_speed: Speed of position reduction
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        trigger_dd: float = 0.10,
        reduction_speed: float = 1.0,
    ) -> None:
        """
        Initialize drawdown control.
        
        Args:
            max_drawdown: Maximum drawdown before full exit (e.g., 0.15 for 15%)
            trigger_dd: Drawdown level to start reducing positions
            reduction_speed: How aggressively to reduce (1 = linear)
        """
        self.max_drawdown = max_drawdown
        self.trigger_dd = trigger_dd
        self.reduction_speed = reduction_speed
    
    def apply(
        self,
        weights: pd.DataFrame,
        cumulative_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Apply drawdown control to weights.
        
        Args:
            weights: Target weights
            cumulative_returns: Cumulative returns series
            
        Returns:
            Adjusted weights
        """
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max.replace(0, np.nan)
        
        # Calculate reduction factor
        reduction_factor = self._compute_reduction(drawdown)
        
        # Apply reduction
        adjusted_weights = weights.mul(reduction_factor, axis=0)
        
        return adjusted_weights
    
    def _compute_reduction(self, drawdown: pd.Series) -> pd.Series:
        """
        Compute position reduction factor based on drawdown.
        
        Args:
            drawdown: Drawdown series (negative values)
            
        Returns:
            Reduction factor (0 to 1)
        """
        # Convert to positive drawdown magnitude
        dd_mag = -drawdown
        
        # No reduction until trigger
        factor = pd.Series(1.0, index=dd_mag.index)
        
        # Between trigger and max, reduce linearly (or by power)
        mask = (dd_mag > self.trigger_dd) & (dd_mag <= self.max_drawdown)
        excess_dd = dd_mag[mask] - self.trigger_dd
        dd_range = self.max_drawdown - self.trigger_dd
        
        if self.reduction_speed == 1:
            # Linear reduction
            factor[mask] = 1 - (excess_dd / dd_range)
        else:
            # Power reduction (faster or slower)
            factor[mask] = 1 - (excess_dd / dd_range) ** self.reduction_speed
        
        # Full exit beyond max
        factor[dd_mag > self.max_drawdown] = 0
        
        return factor


class PositionLimits:
    """
    Enforce position limits and constraints.
    
    Attributes:
        max_long: Maximum long position per asset
        max_short: Maximum short position per asset
        max_gross: Maximum gross exposure
        max_net: Maximum net exposure
    """
    
    def __init__(
        self,
        max_long: Optional[float] = None,
        max_short: Optional[float] = None,
        max_gross: Optional[float] = None,
        max_net: Optional[float] = None,
    ) -> None:
        """
        Initialize position limits.
        
        Args:
            max_long: Maximum long weight per asset
            max_short: Maximum short weight per asset (as positive number)
            max_gross: Maximum sum of absolute weights
            max_net: Maximum net exposure (long - short)
        """
        self.max_long = max_long
        self.max_short = max_short
        self.max_gross = max_gross
        self.max_net = max_net
    
    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position limits to weights.
        
        Args:
            weights: Raw weights
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        # Apply per-asset limits
        if self.max_long is not None:
            constrained = constrained.clip(upper=self.max_long, axis=1)
        
        if self.max_short is not None:
            constrained = constrained.clip(lower=-self.max_short, axis=1)
        
        # Apply portfolio-level limits
        if self.max_gross is not None:
            gross = constrained.abs().sum(axis=1)
            excess_gross = (gross - self.max_gross).clip(lower=0)
            
            # Scale down proportionally
            scale = self.max_gross / gross.replace(0, np.nan)
            scale = scale.clip(upper=1.0)  # Don't scale up
            constrained = constrained.mul(scale, axis=0)
        
        if self.max_net is not None:
            net = constrained.sum(axis=1).abs()
            excess_net = (net - self.max_net).clip(lower=0)
            
            if excess_net.any():
                # Scale down to meet net constraint
                scale = self.max_net / net.replace(0, np.nan)
                scale = scale.clip(upper=1.0)
                constrained = constrained.mul(scale, axis=0)
        
        return constrained


def apply_all_risk_controls(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cumulative_returns: pd.Series,
    vol_target: Optional[VolatilityTarget] = None,
    dd_control: Optional[DrawdownControl] = None,
    position_limits: Optional[PositionLimits] = None,
) -> pd.DataFrame:
    """
    Apply all risk controls in sequence.
    
    Order of operations:
    1. Position limits (hard constraints)
    2. Volatility targeting (scaling)
    3. Drawdown control (final adjustment)
    
    Args:
        weights: Raw target weights
        returns: Asset returns
        cumulative_returns: Cumulative returns
        vol_target: Volatility target object
        dd_control: Drawdown control object
        position_limits: Position limits object
        
    Returns:
        Final adjusted weights
    """
    adjusted = weights.copy()
    
    # 1. Position limits
    if position_limits is not None:
        adjusted = position_limits.apply(adjusted)
    
    # 2. Volatility targeting
    if vol_target is not None:
        adjusted = vol_target.apply(adjusted, returns)
    
    # 3. Drawdown control
    if dd_control is not None:
        adjusted = dd_control.apply(adjusted, cumulative_returns)
    
    return adjusted
