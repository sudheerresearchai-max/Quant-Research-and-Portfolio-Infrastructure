"""
Backtest Engine

Vectorized backtesting engine that applies portfolio weights period-by-period
and tracks returns net of transaction costs.
"""

from typing import Optional, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np

from quant_research.backtest.costs import TransactionCostModel


class BacktestEngine:
    """
    Vectorized backtesting engine.
    
    Runs a backtest by applying target weights period-by-period,
    calculating returns, and deducting transaction costs.
    
    Attributes:
        cost_model: Transaction cost model
        rebalance_frequency: How often to rebalance ('D', 'W', 'M')
        initial_capital: Starting capital
    """
    
    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        rebalance_frequency: str = 'D',
        initial_capital: float = 1_000_000,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize the backtest engine.
        
        Args:
            cost_model: Transaction cost model (default: simple model)
            rebalance_frequency: Rebalancing frequency
            initial_capital: Initial capital
            benchmark_returns: Benchmark returns for comparison
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns
    
    def run(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            weights: Target weights (dates x tickers)
            prices: Asset prices (dates x tickers)
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary containing:
                - portfolio_returns: Daily portfolio returns
                - cumulative_returns: Cumulative returns
                - weights_history: Actual weights over time
                - turnover: Turnover series
                - costs: Transaction costs series
                - benchmark_returns: Benchmark returns (if provided)
        """
        # Align data
        weights, prices = weights.align(prices, join='inner')
        
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(weights.index)
        elif self.benchmark_returns is not None:
            benchmark_returns = self.benchmark_returns.reindex(weights.index)
        
        # Calculate asset returns
        asset_returns = prices.pct_change()
        
        # Initialize tracking arrays
        dates = weights.index
        n_dates = len(dates)
        
        portfolio_returns = []
        turnover_series = []
        costs_series = []
        weights_history = []
        
        # Track previous weights
        prev_weights = pd.Series(0, index=weights.columns)
        
        for i, date in enumerate(dates):
            # Get target weights
            target_weights = weights.loc[date].fillna(0)
            
            # Determine if we rebalance today
            should_rebalance = self._should_rebalance(i, dates)
            
            if should_rebalance:
                # Calculate turnover and costs
                turnover = self.cost_model.calculate_turnover(prev_weights, target_weights)
                cost = self.cost_model.calculate_costs(prev_weights, target_weights)
                
                # Update weights (after costs)
                current_weights = target_weights
                prev_weights = target_weights
            else:
                # No rebalancing - keep previous weights
                # But weights change due to price movements
                turnover = 0
                cost = 0
                current_weights = prev_weights
            
            # Calculate portfolio return for this period
            # Use lagged weights to avoid look-ahead
            if i > 0:
                period_return = (prev_weights * asset_returns.loc[date]).sum()
                # Deduct transaction costs
                period_return -= cost
                
                portfolio_returns.append(period_return)
            else:
                portfolio_returns.append(0)  # First day has no return
            
            turnover_series.append(turnover)
            costs_series.append(cost)
            weights_history.append(current_weights.copy())
        
        # Create result Series
        returns_series = pd.Series(portfolio_returns, index=dates)
        cumulative_returns = (1 + returns_series).cumprod()
        
        results = {
            'portfolio_returns': returns_series,
            'cumulative_returns': cumulative_returns,
            'weights_history': pd.DataFrame(weights_history, index=dates),
            'turnover': pd.Series(turnover_series, index=dates),
            'costs': pd.Series(costs_series, index=dates),
            'total_costs': sum(costs_series),
            'average_turnover': np.mean(turnover_series),
        }
        
        if benchmark_returns is not None:
            results['benchmark_returns'] = benchmark_returns
            results['benchmark_cumulative'] = (1 + benchmark_returns).cumprod()
        
        return results
    
    def _should_rebalance(self, index: int, dates: pd.Index) -> bool:
        """
        Determine if we should rebalance on this date.
        
        Args:
            index: Current index position
            dates: All dates
            
        Returns:
            True if should rebalance
        """
        if self.rebalance_frequency == 'D':
            return True
        
        current_date = dates[index]
        
        if self.rebalance_frequency == 'W':
            # Rebalance on week ends
            return current_date.dayofweek == 4
        
        elif self.rebalance_frequency == 'M':
            # Rebalance on month ends
            return current_date.month != dates[index + 1].month if index < len(dates) - 1 else True
        
        return True
    
    def run_with_risk_controls(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        risk_controls: Dict[str, Any],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest with dynamic risk controls.
        
        Args:
            weights: Target weights
            prices: Asset prices
            risk_controls: Dictionary of risk control objects
            benchmark_returns: Benchmark returns
            
        Returns:
            Backtest results dictionary
        """
        from quant_research.portfolio.risk_controls import (
            VolatilityTarget,
            DrawdownControl,
        )
        
        # Calculate asset returns
        asset_returns = prices.pct_change()
        
        # Track cumulative returns for drawdown control
        cum_returns = pd.Series(1.0, index=prices.index)
        
        # Adjust weights dynamically
        adjusted_weights = weights.copy()
        
        vol_target = risk_controls.get('volatility_target')
        dd_control = risk_controls.get('drawdown_control')
        
        for i, date in enumerate(weights.index):
            if i == 0:
                continue
            
            # Update cumulative returns
            prev_weights = adjusted_weights.iloc[i - 1]
            period_ret = (prev_weights * asset_returns.loc[date]).sum()
            cum_returns.loc[date] = cum_returns.iloc[i - 1] * (1 + period_ret)
            
            # Apply volatility targeting
            if vol_target is not None:
                rolling_vol = asset_returns.loc[:date].rolling(63).std() * np.sqrt(252)
                scale = vol_target.target_vol / rolling_vol.mean(axis=1).iloc[-1]
                scale = np.clip(scale, vol_target.min_leverage, vol_target.max_leverage)
                adjusted_weights.loc[date] *= scale
            
            # Apply drawdown control
            if dd_control is not None:
                running_max = cum_returns.loc[:date].cummax()
                dd = (cum_returns.loc[date] - running_max) / running_max
                
                if dd < -dd_control.trigger_dd:
                    reduction = min(1, (dd + dd_control.max_drawdown) / 
                                   (dd_control.trigger_dd - dd_control.max_drawdown))
                    adjusted_weights.loc[date] *= max(0, reduction)
        
        # Run standard backtest with adjusted weights
        return self.run(adjusted_weights, prices, benchmark_returns)


def calculate_strategy_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate basic strategy metrics from backtest results.
    
    Args:
        results: Backtest results dictionary
        
    Returns:
        Dictionary of metrics
    """
    returns = results['portfolio_returns']
    
    # Basic statistics
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Turnover
    avg_turnover = results.get('average_turnover', 0) * 252  # Annualized
    
    return {
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'annualized_turnover': avg_turnover,
        'total_transaction_costs': results.get('total_costs', 0),
    }
