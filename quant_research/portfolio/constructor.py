"""
Portfolio Constructor

Converts trading signals into target portfolio weights using various methods.
"""

from typing import Optional, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class PortfolioConstructor:
    """
    Convert signals to portfolio weights.
    
    Supports multiple construction methods:
    - rank_zscore: Rank-based weighting with z-score normalization
    - equal_weight: Equal weight for all active positions
    - signal_proportional: Weights proportional to signal strength
    - optimization: Mean-variance optimization (if expected returns provided)
    
    Attributes:
        method: Construction method
        max_weight: Maximum absolute weight per asset
        target_gross: Target gross exposure (sum of absolute weights)
    """
    
    def __init__(
        self,
        method: str = 'rank_zscore',
        max_weight: Optional[float] = None,
        target_gross: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize the portfolio constructor.
        
        Args:
            method: Construction method
            max_weight: Maximum absolute weight per asset (None for no limit)
            target_gross: Target gross exposure
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.max_weight = max_weight
        self.target_gross = target_gross
        self.kwargs = kwargs
    
    def build(
        self,
        signals: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build portfolio weights from signals.
        
        Args:
            signals: DataFrame of signal values (dates x tickers)
            expected_returns: Optional expected returns for optimization
            cov_matrix: Optional covariance matrix for optimization
            
        Returns:
            DataFrame of target weights (dates x tickers)
        """
        if self.method == 'rank_zscore':
            weights = self._rank_zscore(signals)
        elif self.method == 'equal_weight':
            weights = self._equal_weight(signals)
        elif self.method == 'signal_proportional':
            weights = self._signal_proportional(signals)
        elif self.method == 'optimization':
            if expected_returns is None or cov_matrix is None:
                raise ValueError("Optimization method requires expected_returns and cov_matrix")
            weights = self._optimize(signals, expected_returns, cov_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply constraints
        if self.max_weight is not None:
            weights = self._apply_weight_limits(weights)
        
        # Scale to target gross exposure
        weights = self._scale_to_target_gross(weights)
        
        return weights
    
    def _rank_zscore(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Rank-based weighting with z-score normalization.
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Weight DataFrame
        """
        # Cross-sectional rank at each date
        ranks = signals.rank(axis=1, pct=True)
        
        # Transform to z-score (-1 to 1 range)
        weights = (ranks - 0.5) * 2
        
        return weights
    
    def _equal_weight(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Equal weight for assets with non-zero signals.
        
        Long positions for positive signals, short for negative.
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Weight DataFrame
        """
        # Determine direction
        direction = np.sign(signals)
        
        # Count active positions
        n_active = direction.abs().sum(axis=1)
        
        # Equal weight for each active position
        weights = direction / n_active.replace(0, np.nan)
        
        return weights
    
    def _signal_proportional(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Weights proportional to signal strength.
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Weight DataFrame
        """
        # Take absolute value for scaling
        abs_signals = signals.abs()
        
        # Sum of absolute signals
        total_signal = abs_signals.sum(axis=1)
        
        # Proportional weights with sign
        weights = signals / total_signal.replace(0, np.nan)
        
        return weights
    
    def _optimize(
        self,
        signals: pd.DataFrame,
        expected_returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Mean-variance optimization.
        
        Uses signals as expected returns if not provided.
        
        Args:
            signals: Signal DataFrame
            expected_returns: Expected returns DataFrame
            cov_matrix: Covariance matrix
            
        Returns:
            Weight DataFrame
        """
        weights_list = []
        dates = []
        
        for date in signals.index:
            signal_row = signals.loc[date]
            valid_assets = signal_row.notna()
            
            if valid_assets.sum() < 2:
                weights_list.append(pd.Series(index=signals.columns, dtype=float))
                dates.append(date)
                continue
            
            # Get valid assets
            tickers = signals.columns[valid_assets]
            n_assets = len(tickers)
            
            # Expected returns (use signal as proxy)
            mu = expected_returns.loc[date, tickers].values if expected_returns is not None else signal_row[tickers].values
            
            # Covariance (use diagonal if not available)
            if cov_matrix is not None and tickers[0] in cov_matrix.columns:
                Sigma = cov_matrix.loc[tickers, tickers].values
            else:
                # Use identity matrix scaled by average variance
                Sigma = np.eye(n_assets) * 0.04  # 20% vol assumption
            
            # Optimize
            result = self._mean_variance_opt(mu, Sigma)
            
            if result is not None:
                w = pd.Series(result, index=tickers)
                weights_list.append(w.reindex(signals.columns))
            else:
                weights_list.append(pd.Series(index=signals.columns, dtype=float))
            
            dates.append(date)
        
        return pd.DataFrame(weights_list, index=dates)
    
    def _mean_variance_opt(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Solve mean-variance optimization problem.
        
        Maximize: w'mu - (risk_aversion/2) * w'Sigma*w
        
        Args:
            mu: Expected returns vector
            Sigma: Covariance matrix
            risk_aversion: Risk aversion parameter
            
        Returns:
            Optimal weights or None if optimization fails
        """
        n = len(mu)
        
        def objective(w):
            return -(mu @ w - (risk_aversion / 2) * w @ Sigma @ w)
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Constraints: sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: long-only (can be modified)
        bounds = [(0, 1) for _ in range(n)]
        
        try:
            result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                return result.x
        except Exception:
            pass
        
        return None
    
    def _apply_weight_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply maximum weight constraints.
        
        Args:
            weights: Weight DataFrame
            
        Returns:
            Clipped weight DataFrame
        """
        return weights.clip(-self.max_weight, self.max_weight)
    
    def _scale_to_target_gross(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Scale weights to achieve target gross exposure.
        
        Args:
            weights: Weight DataFrame
            
        Returns:
            Scaled weight DataFrame
        """
        gross = weights.abs().sum(axis=1)
        scale_factor = self.target_gross / gross.replace(0, np.nan)
        
        return weights.mul(scale_factor, axis=0)
