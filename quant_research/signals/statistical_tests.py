"""
Statistical Tests for Signals

Provides statistical significance testing for trading signals including:
- OLS regression with Newey-West adjusted standard errors
- Information Coefficient (IC) analysis
- T-statistics and p-values
"""

from typing import Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac


def test_signal_significance(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'newey_west',
    lags: int = 5,
) -> Dict[str, Any]:
    """
    Test the statistical significance of a trading signal.
    
    Regresses forward returns on signal values and computes t-statistics
    with appropriate standard error adjustments.
    
    Args:
        signals: DataFrame of signal values (dates x tickers)
        forward_returns: DataFrame of forward returns (dates x tickers)
        method: Standard error method ('ols', 'newey_west', 'clustered')
        lags: Number of lags for Newey-West adjustment
        
    Returns:
        Dictionary containing:
            - alpha: Intercept (abnormal return)
            - beta: Signal coefficient
            - t_stat: T-statistic for beta
            - p_value: P-value for beta
            - r_squared: R-squared of regression
            - n_obs: Number of observations
            - method: Method used for standard errors
    """
    # Align signals and returns
    signals_aligned, returns_aligned = signals.align(forward_returns, join='inner')
    
    # Stack to long format for pooled regression
    y = returns_aligned.stack()
    X = signals_aligned.stack()
    
    # Remove NaN values
    mask = y.notna() & X.notna()
    y = y[mask]
    X = X[mask]
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Run OLS regression
    model = OLS(y, X_with_const)
    results = model.fit()
    
    # Get base results
    beta = results.params.iloc[1]  # Signal coefficient
    alpha = results.params.iloc[0]  # Intercept
    
    if method == 'ols':
        # Standard OLS standard errors
        t_stat = results.tvalues.iloc[1]
        p_value = results.pvalues.iloc[1]
        
    elif method == 'newey_west':
        # Newey-West HAC standard errors (robust to autocorrelation)
        nw_cov = cov_hac(results, nlags=lags)
        nw_se = np.sqrt(np.diag(nw_cov))
        t_stat = beta / nw_se[1]
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
    elif method == 'clustered':
        # Clustered standard errors by date
        # Requires date information in index
        dates = y.index.get_level_values(0)
        unique_dates = dates.unique()
        
        # Compute clustered SE manually
        residuals_by_date = []
        X_by_date = []
        
        for date in unique_dates:
            date_mask = dates == date
            if date_mask.sum() > 0:
                residuals_by_date.append(y[date_mask] - results.predict(X_with_const[date_mask]))
                X_by_date.append(X_with_const[date_mask].values)
        
        # Sandwich estimator
        n_clusters = len(unique_dates)
        k = X_with_const.shape[1]
        
        meat = np.zeros((k, k))
        for i, (resid, x) in enumerate(zip(residuals_by_date, X_by_date)):
            score = x.T @ resid.values
            meat += np.outer(score, score)
        
        bread = np.linalg.inv(X_with_const.T @ X_with_const)
        cluster_cov = (n_clusters / (n_clusters - 1)) * bread @ meat @ bread * (n_clusters - 1) / n_clusters
        
        cluster_se = np.sqrt(np.diag(cluster_cov))
        t_stat = beta / cluster_se[1]
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'alpha': alpha,
        'beta': beta,
        't_stat': t_stat,
        'p_value': p_value,
        'r_squared': results.rsquared,
        'n_obs': len(y),
        'method': method,
        'lags': lags if method == 'newey_west' else None,
    }


def newey_west_tstat(
    y: np.ndarray,
    X: np.ndarray,
    lags: int = 5,
) -> Tuple[float, float, float]:
    """
    Compute Newey-West adjusted t-statistic for regression coefficients.
    
    Args:
        y: Dependent variable array
        X: Independent variable matrix (should include constant if desired)
        lags: Number of lags for HAC adjustment
        
    Returns:
        Tuple of (coefficient, t_statistic, p_value) for the last column of X
    """
    # Run OLS
    model = OLS(y, X)
    results = model.fit()
    
    # Get coefficient
    beta = results.params.iloc[-1]
    
    # Compute Newey-West covariance
    nw_cov = cov_hac(results, nlags=lags)
    nw_se = np.sqrt(np.diag(nw_cov))
    
    # T-statistic and p-value
    t_stat = beta / nw_se[-1]
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    return beta, t_stat, p_value


def compute_ic(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'rank',
) -> pd.Series:
    """
    Compute Information Coefficient (IC) time series.
    
    IC is the correlation between signal and subsequent returns.
    
    Args:
        signals: DataFrame of signal values
        forward_returns: DataFrame of forward returns
        method: Correlation method ('pearson', 'rank', 'kendall')
        
    Returns:
        Series of IC values over time
    """
    # Align data
    signals_aligned, returns_aligned = signals.align(forward_returns, join='inner')
    
    ics = []
    dates = []
    
    for date in signals_aligned.index:
        signal_row = signals_aligned.loc[date]
        return_row = returns_aligned.loc[date]
        
        # Remove NaN pairs
        mask = signal_row.notna() & return_row.notna()
        s = signal_row[mask]
        r = return_row[mask]
        
        if len(s) > 2:  # Need at least 3 points for correlation
            if method == 'pearson':
                ic = s.corr(r)
            elif method == 'rank':
                ic = s.rank().corr(r.rank())
            elif method == 'kendall':
                ic = s.corr(r, method='kendall')
            else:
                raise ValueError(f"Unknown method: {method}")
            
            ics.append(ic)
            dates.append(date)
    
    return pd.Series(ics, index=dates)


def compute_ic_statistics(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'rank',
) -> Dict[str, Any]:
    """
    Compute summary statistics for Information Coefficient.
    
    Args:
        signals: DataFrame of signal values
        forward_returns: DataFrame of forward returns
        method: Correlation method
        
    Returns:
        Dictionary with IC statistics:
            - mean_ic: Average IC
            - std_ic: Standard deviation of IC
            - ic_tstat: T-statistic of mean IC
            - ic_pvalue: P-value of mean IC
            - hit_rate: Percentage of positive IC periods
            - min_ic: Minimum IC
            - max_ic: Maximum IC
    """
    ic_series = compute_ic(signals, forward_returns, method=method)
    
    if len(ic_series) == 0:
        return {
            'mean_ic': np.nan,
            'std_ic': np.nan,
            'ic_tstat': np.nan,
            'ic_pvalue': np.nan,
            'hit_rate': np.nan,
            'min_ic': np.nan,
            'max_ic': np.nan,
            'n_periods': 0,
        }
    
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    
    # T-statistic for mean IC
    ic_tstat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0
    ic_pvalue = 2 * (1 - stats.t.cdf(abs(ic_tstat), df=n-1))
    
    # Hit rate
    hit_rate = (ic_series > 0).mean()
    
    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ic_tstat': ic_tstat,
        'ic_pvalue': ic_pvalue,
        'hit_rate': hit_rate,
        'min_ic': ic_series.min(),
        'max_ic': ic_series.max(),
        'n_periods': n,
    }


def check_lookahead_bias(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    max_lag: int = 5,
) -> Dict[int, float]:
    """
    Check for potential look-ahead bias in signals.
    
    Computes correlation between signal at time t and price changes
    at times t+k for various k. A significant correlation for k > 0
    suggests look-ahead bias.
    
    Args:
        signals: DataFrame of signal values
        prices: DataFrame of prices
        max_lag: Maximum lag to check
        
    Returns:
        Dictionary mapping lag to correlation
    """
    returns = prices.pct_change()
    
    correlations = {}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Signal vs future returns (potential look-ahead)
            shifted_returns = returns.shift(-lag)
        elif lag > 0:
            # Signal vs past returns (expected relationship)
            shifted_returns = returns.shift(lag)
        else:
            shifted_returns = returns
        
        # Compute average correlation across assets
        corr_matrix = signals.corrwith(shifted_returns)
        avg_corr = corr_corr_matrix.mean()
        correlations[lag] = avg_corr
    
    return correlations


def detect_data_snooping(
    backtest_results: Dict[str, Any],
    n_strategies_tested: int,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Apply Deflated Sharpe Ratio adjustment for data snooping.
    
    Based on Bailey & López de Prado (2014), adjusts performance
    metrics for multiple testing.
    
    Args:
        backtest_results: Dictionary containing 'sharpe_ratio' and other metrics
        n_strategies_tested: Number of strategies tested before finding this one
        confidence: Confidence level for adjustment
        
    Returns:
        Dictionary with adjusted metrics
    """
    sharpe = backtest_results.get('sharpe_ratio', 0)
    n_obs = backtest_results.get('n_obs', 252)
    
    # Adjusted p-value for multiple testing (Bonferroni correction)
    raw_pvalue = 1 - stats.norm.cdf(sharpe * np.sqrt(n_obs))
    adjusted_pvalue = min(1, raw_pvalue * n_strategies_tested)
    
    # Deflated Sharpe Ratio
    # The expected maximum Sharpe from n random strategies
    expected_max_sharpe = stats.norm.ppf(1 - 1/n_strategies_tested) / np.sqrt(n_obs)
    deflated_sharpe = sharpe - expected_max_sharpe
    
    # Probability that Sharpe ratio is due to chance
    prob_random = stats.norm.cdf(-(sharpe * np.sqrt(n_obs) - expected_max_sharpe * np.sqrt(n_obs)))
    
    return {
        'original_sharpe': sharpe,
        'deflated_sharpe': deflated_sharpe,
        'adjusted_pvalue': adjusted_pvalue,
        'prob_random': prob_random,
        'n_strategies_tested': n_strategies_tested,
        'confidence_level': confidence,
    }
