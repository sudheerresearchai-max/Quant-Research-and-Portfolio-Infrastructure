"""
Unit Tests for Signal Calculations

Tests the signal computation modules for correctness.
"""

import pytest
import pandas as pd
import numpy as np

from quant_research.signals.momentum import MomentumSignal, CrossSectionalMomentum
from quant_research.signals.mean_reversion import MeanReversionSignal, RSI_Signal
from quant_research.signals.base import SignalBase


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    # Generate correlated prices
    returns = np.random.normal(0.0005, 0.02, (500, 5))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestMomentumSignal:
    """Tests for MomentumSignal class."""
    
    def test_momentum_computation(self, sample_prices):
        """Test that momentum signal is computed correctly."""
        signal = MomentumSignal(lookback=21, skip=0)
        result = signal.compute(sample_prices)
        
        assert result.shape == sample_prices.shape
        assert isinstance(result, pd.DataFrame)
        
        # Check that first 'lookback' rows are NaN
        assert result.iloc[:21].isna().all().all()
    
    def test_momentum_with_skip(self, sample_prices):
        """Test momentum with skip period."""
        signal = MomentumSignal(lookback=21, skip=5)
        result = signal.compute(sample_prices)
        
        # First lookback + skip rows should be NaN
        assert result.iloc[:26].isna().all().all()
    
    def test_momentum_sign(self, sample_prices):
        """Test that positive returns give positive momentum."""
        # Create prices with clear upward trend
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['UP']
        prices = pd.DataFrame(
            100 * np.exp(np.arange(100) * 0.01),
            index=dates,
            columns=tickers
        )
        
        signal = MomentumSignal(lookback=10, skip=0)
        result = signal.compute(prices)
        
        # All non-NaN values should be positive
        assert (result.dropna() > 0).all().all()


class TestCrossSectionalMomentum:
    """Tests for CrossSectionalMomentum class."""
    
    def test_cs_momentum_computation(self, sample_prices):
        """Test cross-sectional momentum computation."""
        signal = CrossSectionalMomentum(lookback=21, skip=5)
        result = signal.compute(sample_prices)
        
        assert result.shape == sample_prices.shape
        
        # Values should be centered around 0 (for percentile method)
        non_nan = result.dropna()
        assert abs(non_nan.mean(axis=1).mean()) < 0.1
    
    def test_cs_momentum_ranking(self, sample_prices):
        """Test that ranking works correctly."""
        signal = CrossSectionalMomentum(lookback=21, skip=5, method='zscore')
        result = signal.compute(sample_prices)
        
        non_nan = result.dropna()
        
        # Z-scores should have mean close to 0 across assets
        means = non_nan.mean(axis=1)
        assert abs(means.mean()) < 0.1


class TestMeanReversionSignal:
    """Tests for MeanReversionSignal class."""
    
    def test_mr_computation(self, sample_prices):
        """Test mean reversion signal computation."""
        signal = MeanReversionSignal(window=21)
        result = signal.compute(sample_prices)
        
        assert result.shape == sample_prices.shape
        assert result.iloc[:21].isna().all().all()
    
    def test_mr_zscore_method(self, sample_prices):
        """Test z-score method produces standardized values."""
        signal = MeanReversionSignal(window=21, method='zscore')
        result = signal.compute(sample_prices)
        
        non_nan = result.dropna()
        
        # Check reasonable bounds for z-scores
        assert non_nan.abs().max().max() < 10  # Sanity check


class TestRSISignal:
    """Tests for RSI_Signal class."""
    
    def test_rsi_computation(self, sample_prices):
        """Test RSI signal computation."""
        signal = RSI_Signal(window=14)
        result = signal.compute(sample_prices)
        
        assert result.shape == sample_prices.shape
        
        # RSI-based signal should be bounded
        non_nan = result.dropna()
        assert non_nan.min().min() >= -1
        assert non_nan.max().max() <= 1


class TestSignalBase:
    """Tests for SignalBase class methods."""
    
    def test_standardize_zscore(self, sample_prices):
        """Test z-score standardization."""
        # Create raw signals
        raw_signals = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        signal_obj = SignalBase(name='test')
        standardized = signal_obj.standardize(raw_signals, method='zscore')
        
        # Each row should have mean ~0 and std ~1
        row_means = standardized.mean(axis=1)
        row_stds = standardized.std(axis=1)
        
        assert np.allclose(row_means, 0, atol=1e-10)
        assert np.allclose(row_stds[~np.isnan(row_stds)], 1, rtol=1e-5)
    
    def test_lag(self, sample_prices):
        """Test signal lagging."""
        raw_signals = pd.DataFrame(
            np.arange(100).reshape(-1, 1),
            columns=['A']
        )
        
        signal_obj = SignalBase(name='test')
        lagged = signal_obj.lag(raw_signals, periods=1)
        
        assert lagged.iloc[0].isna().all()
        assert lagged.iloc[1:].equals(raw_signals.iloc[:-1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
