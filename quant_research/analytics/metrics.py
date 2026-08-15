"""
Performance Metrics

Comprehensive performance metrics calculation including:
- Return metrics (annualized, cumulative)
- Risk metrics (volatility, drawdown)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Other statistics (skew, kurtosis, hit rate)
"""

from typing import Optional, Dict, Any, Union, List
import pandas as pd
import numpy as np
from scipy import stats


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Attributes:
        returns: Return series
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        periods_per_year: Number of periods per year (252 for daily)
    """
    
    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize performance metrics calculator.
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization
        """
        self.returns = returns.dropna()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def total_return(self) -> float:
        """Calculate total cumulative return."""
        return (1 + self.returns).prod() - 1
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        total_ret = self.total_return()
        n_years = len(self.returns) / self.periods_per_year
        return (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.returns.std() * np.sqrt(self.periods_per_year)
    
    def sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        ann_ret = self.annualized_return()
        ann_vol = self.annualized_volatility()
        return (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            target_return: Target return for downside calculation
        """
        ann_ret = self.annualized_return()
        
        # Downside deviation
        downside_returns = self.returns[self.returns < target_return]
        downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(self.periods_per_year)
        
        return (ann_ret - self.risk_free_rate) / downside_std if downside_std > 0 else 0
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        ann_ret = self.annualized_return()
        max_dd = abs(self.max_drawdown())
        return ann_ret / max_dd if max_dd > 0 else 0
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def avg_drawdown(self) -> float:
        """Calculate average drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.mean()
    
    def hit_rate(self) -> float:
        """Calculate hit rate (percentage of positive periods)."""
        return (self.returns > 0).mean()
    
    def best_period(self) -> float:
        """Return the best period return."""
        return self.returns.max()
    
    def worst_period(self) -> float:
        """Return the worst period return."""
        return self.returns.min()
    
    def skewness(self) -> float:
        """Calculate return distribution skewness."""
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """Calculate return distribution excess kurtosis."""
        return stats.kurtosis(self.returns)
    
    def var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
        """
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            confidence: Confidence level
        """
        var_threshold = self.var(confidence)
        return self.returns[self.returns <= var_threshold].mean()
    
    def turnover(self, weights: pd.DataFrame) -> float:
        """
        Calculate average turnover.
        
        Args:
            weights: DataFrame of portfolio weights
        """
        turnover = weights.diff().abs().sum(axis=1) / 2
        return turnover.mean() * self.periods_per_year
    
    def all_metrics(self, weights: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            weights: Optional weights for turnover calculation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'annualized_volatility': self.annualized_volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'max_drawdown': self.max_drawdown(),
            'avg_drawdown': self.avg_drawdown(),
            'hit_rate': self.hit_rate(),
            'best_period': self.best_period(),
            'worst_period': self.worst_period(),
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
            'var_95': self.var(0.95),
            'cvar_95': self.cvar(0.95),
            'n_periods': len(self.returns),
            'n_years': len(self.returns) / self.periods_per_year,
        }
        
        if weights is not None:
            metrics['annualized_turnover'] = self.turnover(weights)
        
        return metrics


def compare_strategies(
    returns_dict: Dict[str, pd.Series],
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Args:
        returns_dict: Dictionary mapping strategy name to returns
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with metrics for each strategy
    """
    results = []
    
    for name, returns in returns_dict.items():
        metrics = PerformanceMetrics(returns, risk_free_rate)
        metric_dict = metrics.all_metrics()
        metric_dict['strategy'] = name
        results.append(metric_dict)
    
    df = pd.DataFrame(results)
    df = df.set_index('strategy')
    
    return df
