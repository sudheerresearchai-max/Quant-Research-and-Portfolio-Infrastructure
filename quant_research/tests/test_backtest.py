"""
Unit Tests for Backtest Engine

Tests the backtest engine and transaction cost model.
"""

import pytest
import pandas as pd
import numpy as np

from quant_research.backtest.engine import BacktestEngine, calculate_strategy_metrics
from quant_research.backtest.costs import TransactionCostModel


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    tickers = ['A', 'B', 'C']
    
    returns = np.random.normal(0.0005, 0.02, (252, 3))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def sample_weights():
    """Create sample weights."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    tickers = ['A', 'B', 'C']
    
    weights = np.random.randn(252, 3)
    weights = weights / np.abs(weights).sum(axis=1, keepdims=True)
    
    return pd.DataFrame(weights, index=dates, columns=tickers)


class TestTransactionCostModel:
    """Tests for TransactionCostModel class."""
    
    def test_cost_calculation(self):
        """Test basic cost calculation."""
        model = TransactionCostModel(spread=0.001, slippage=0.0005, commission=0.001)
        
        w_prev = pd.Series({'A': 0.5, 'B': 0.5})
        w_target = pd.Series({'A': 0.3, 'B': 0.7})
        
        cost = model.calculate_costs(w_prev, w_target)
        
        # Turnover is 0.2, total rate is 0.0025
        expected_turnover = 0.2
        assert model.calculate_turnover(w_prev, w_target) == expected_turnover
        assert cost > 0
    
    def test_zero_turnover(self):
        """Test that zero turnover gives zero costs."""
        model = TransactionCostModel()
        
        w_prev = pd.Series({'A': 0.5, 'B': 0.5})
        w_target = pd.Series({'A': 0.5, 'B': 0.5})
        
        cost = model.calculate_costs(w_prev, w_target)
        assert cost == 0
    
    def test_cost_breakdown(self):
        """Test cost breakdown calculation."""
        model = TransactionCostModel(spread=0.001, slippage=0.0005, commission=0.001)
        
        w_prev = pd.Series({'A': 0.5, 'B': 0.5})
        w_target = pd.Series({'A': 0.0, 'B': 1.0})
        
        breakdown = model.get_cost_breakdown(w_prev, w_target)
        
        assert 'turnover' in breakdown
        assert 'spread_cost' in breakdown
        assert 'slippage_cost' in breakdown
        assert 'commission_cost' in breakdown
        assert 'total_cost' in breakdown
        
        # Verify components sum to total
        component_sum = (
            breakdown['spread_cost'] + 
            breakdown['slippage_cost'] + 
            breakdown['commission_cost']
        )
        assert abs(breakdown['total_cost'] - component_sum) < 1e-10


class TestBacktestEngine:
    """Tests for BacktestEngine class."""
    
    def test_backtest_run(self, sample_prices, sample_weights):
        """Test basic backtest execution."""
        engine = BacktestEngine()
        results = engine.run(sample_weights, sample_prices)
        
        assert 'portfolio_returns' in results
        assert 'cumulative_returns' in results
        assert 'turnover' in results
        assert 'costs' in results
        
        # Check lengths match
        assert len(results['portfolio_returns']) == len(sample_prices)
    
    def test_backtest_with_benchmark(self, sample_prices, sample_weights):
        """Test backtest with benchmark."""
        benchmark = sample_prices.mean(axis=1).pct_change()
        
        engine = BacktestEngine(benchmark_returns=benchmark)
        results = engine.run(sample_weights, sample_prices)
        
        assert 'benchmark_returns' in results
        assert 'benchmark_cumulative' in results
    
    def test_no_lookahead(self, sample_prices, sample_weights):
        """Test that backtest doesn't use future data."""
        engine = BacktestEngine()
        results = engine.run(sample_weights, sample_prices)
        
        # First return should be 0 (no look-ahead)
        assert results['portfolio_returns'].iloc[0] == 0


class TestStrategyMetrics:
    """Tests for strategy metrics calculation."""
    
    def test_metrics_calculation(self, sample_prices, sample_weights):
        """Test strategy metrics calculation."""
        engine = BacktestEngine()
        results = engine.run(sample_weights, sample_prices)
        
        metrics = calculate_strategy_metrics(results)
        
        assert 'annualized_return' in metrics
        assert 'annualized_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
