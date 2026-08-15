"""
Risk Analysis

Advanced risk analysis including:
- Drawdown analysis
- VaR/CVaR calculations
- Tail risk metrics
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class RiskAnalysis:
    """
    Comprehensive risk analysis.
    
    Attributes:
        returns: Return series
    """
    
    def __init__(self, returns: pd.Series) -> None:
        """
        Initialize risk analysis.
        
        Args:
            returns: Return series
        """
        self.returns = returns.dropna()
    
    def var(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        Calculate Value at Risk.
        
        Args:
            confidence: Confidence level
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR as a positive number (loss)
        """
        if method == 'historical':
            return -np.percentile(self.returns, (1 - confidence) * 100)
        
        elif method == 'parametric':
            mu = self.returns.mean()
            sigma = self.returns.std()
            z = stats.norm.ppf(1 - confidence)
            return -(mu + z * sigma)
        
        elif method == 'monte_carlo':
            # Simulate future returns
            n_sims = 10000
            mu = self.returns.mean()
            sigma = self.returns.std()
            simulated = np.random.normal(mu, sigma, n_sims)
            return -np.percentile(simulated, (1 - confidence) * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            confidence: Confidence level
            
        Returns:
            CVaR as a positive number
        """
        var_threshold = -self.var(confidence)
        tail_returns = self.returns[self.returns <= var_threshold]
        return -tail_returns.mean() if len(tail_returns) > 0 else 0
    
    def tail_ratio(self) -> float:
        """
        Calculate tail ratio (right tail / left tail).
        
        Returns:
            Ratio of 95th percentile to absolute 5th percentile
        """
        right_tail = np.percentile(self.returns, 95)
        left_tail = abs(np.percentile(self.returns, 5))
        return right_tail / left_tail if left_tail > 0 else 0
    
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            threshold: Return threshold
            
        Returns:
            Omega ratio (gains/losses relative to threshold)
        """
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = abs((self.returns - threshold).clip(upper=0)).sum()
        return gains / losses if losses > 0 else np.inf
    
    def downside_deviation(self, target_return: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Args:
            target_return: Target return
            
        Returns:
            Annualized downside deviation
        """
        downside = self.returns[self.returns < target_return]
        return np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    
    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (measure of drawdown severity).
        
        Returns:
            Ulcer Index
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return np.sqrt((drawdown ** 2).mean())
    
    def risk_metrics(self) -> Dict[str, float]:
        """
        Get all risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'var_95': self.var(0.95),
            'var_99': self.var(0.99),
            'cvar_95': self.cvar(0.95),
            'cvar_99': self.cvar(0.99),
            'tail_ratio': self.tail_ratio(),
            'omega_ratio': self.omega_ratio(),
            'downside_deviation': self.downside_deviation(),
            'ulcer_index': self.ulcer_index(),
        }


class DrawdownAnalysis:
    """
    Detailed drawdown analysis.
    
    Attributes:
        returns: Return series
    """
    
    def __init__(self, returns: pd.Series) -> None:
        """
        Initialize drawdown analysis.
        
        Args:
            returns: Return series
        """
        self.returns = returns.dropna()
        self.drawdowns = self._calculate_drawdowns()
    
    def _calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        return self.drawdowns.min()
    
    def avg_drawdown(self) -> float:
        """Get average drawdown during drawdown periods."""
        dd_periods = self.drawdowns[self.drawdowns < 0]
        return dd_periods.mean() if len(dd_periods) > 0 else 0
    
    def max_drawdown_duration(self) -> int:
        """Get maximum drawdown duration in periods."""
        # Find drawdown periods
        in_drawdown = self.drawdowns < 0
        
        # Count consecutive periods
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return max(durations) if durations else 0
    
    def max_recovery_time(self) -> int:
        """Get maximum time to recover from drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        
        # Find when we're below running max
        underwater = cumulative < running_max
        
        durations = []
        current_duration = 0
        
        for is_under in underwater:
            if is_under:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        return max(durations) if durations else 0
    
    def drawdown_events(self, threshold: float = -0.05) -> pd.DataFrame:
        """
        Identify significant drawdown events.
        
        Args:
            threshold: Drawdown threshold to consider
            
        Returns:
            DataFrame of drawdown events with start, end, depth, duration
        """
        events = []
        in_event = False
        event_start = None
        event_min = 0
        event_min_date = None
        
        for date, dd in self.drawdowns.items():
            if dd < threshold and not in_event:
                in_event = True
                event_start = date
                event_min = dd
                event_min_date = date
            elif in_event:
                if dd < event_min:
                    event_min = dd
                    event_min_date = date
                if dd >= 0:
                    events.append({
                        'start': event_start,
                        'end': date,
                        'trough': event_min_date,
                        'depth': event_min,
                        'duration': (date - event_start).days,
                    })
                    in_event = False
        
        # Handle ongoing drawdown
        if in_event:
            events.append({
                'start': event_start,
                'end': self.drawdowns.index[-1],
                'trough': event_min_date,
                'depth': event_min,
                'duration': (self.drawdowns.index[-1] - event_start).days,
            })
        
        return pd.DataFrame(events)
    
    def summary(self) -> Dict[str, Any]:
        """
        Get drawdown summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        return {
            'max_drawdown': self.max_drawdown(),
            'avg_drawdown': self.avg_drawdown(),
            'max_duration_days': self.max_drawdown_duration(),
            'max_recovery_days': self.max_recovery_time(),
            'current_drawdown': self.drawdowns.iloc[-1],
        }
