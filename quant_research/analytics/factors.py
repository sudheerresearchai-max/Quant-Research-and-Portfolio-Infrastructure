"""
Factor Analysis

Factor exposure analysis using regression-based methods.
Supports Fama-French style factor models and custom factors.
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


class FactorAnalysis:
    """
    Analyze portfolio factor exposures.
    
    Supports:
    - Single factor model (CAPM)
    - Multi-factor models (Fama-French)
    - Custom factor specification
    
    Attributes:
        returns: Portfolio returns
        factors: Factor returns DataFrame
    """
    
    def __init__(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize factor analysis.
        
        Args:
            returns: Portfolio returns series
            factors: DataFrame of factor returns (columns are factors)
            risk_free_rate: Optional risk-free rate series
        """
        self.returns = returns.dropna()
        self.factors = factors.dropna()
        self.risk_free_rate = risk_free_rate
        
        # Align data
        self.returns, self.factors = self.returns.align(self.factors, join='inner')
        
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate.reindex(self.returns.index)
    
    def excess_returns(self) -> pd.Series:
        """Calculate excess returns over risk-free rate."""
        if self.risk_free_rate is not None:
            return self.returns - self.risk_free_rate
        else:
            # Assume 2% annual risk-free rate
            daily_rf = 0.02 / 252
            return self.returns - daily_rf
    
    def fit_model(self, factor_names: Optional[List[str]] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit factor model via OLS.
        
        Args:
            factor_names: List of factor column names to include (None for all)
            
        Returns:
            Statsmodels regression results
        """
        y = self.excess_returns()
        
        if factor_names is None:
            X = self.factors
        else:
            X = self.factors[factor_names]
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        model = OLS(y, X)
        results = model.fit()
        
        return results
    
    def alpha(self, factor_names: Optional[List[str]] = None) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            factor_names: Factors to include
            
        Returns:
            Annualized alpha
        """
        results = self.fit_model(factor_names)
        return results.params['const'] * 252  # Annualize
    
    def beta(self, factor_names: Optional[List[str]] = None) -> pd.Series:
        """
        Get factor betas.
        
        Args:
            factor_names: Factors to include
            
        Returns:
            Series of factor betas
        """
        results = self.fit_model(factor_names)
        betas = results.params.drop('const', errors='ignore')
        return betas
    
    def r_squared(self, factor_names: Optional[List[str]] = None) -> float:
        """Get R-squared of factor model."""
        results = self.fit_model(factor_names)
        return results.rsquared
    
    def t_stats(self, factor_names: Optional[List[str]] = None) -> pd.Series:
        """Get t-statistics for factor loadings."""
        results = self.fit_model(factor_names)
        return results.tvalues.drop('const', errors='ignore')
    
    def p_values(self, factor_names: Optional[List[str]] = None) -> pd.Series:
        """Get p-values for factor loadings."""
        results = self.fit_model(factor_names)
        return results.pvalues.drop('const', errors='ignore')
    
    def factor_summary(self, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comprehensive factor analysis summary.
        
        Args:
            factor_names: Factors to include
            
        Returns:
            DataFrame with factor statistics
        """
        results = self.fit_model(factor_names)
        
        summary = []
        for factor in self.factors.columns:
            if factor in results.params:
                summary.append({
                    'factor': factor,
                    'beta': results.params[factor],
                    't_stat': results.tvalues[factor],
                    'p_value': results.pvalues[factor],
                    'significant_5pct': results.pvalues[factor] < 0.05,
                })
        
        df = pd.DataFrame(summary)
        df = df.set_index('factor')
        
        return df
    
    def attribution(self, factor_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Decompose returns into factor contributions.
        
        Args:
            factor_names: Factors to include
            
        Returns:
            Dictionary with return attribution
        """
        results = self.fit_model(factor_names)
        betas = results.params.drop('const')
        
        # Factor contributions
        contributions = {}
        for factor in betas.index:
            factor_mean = self.factors[factor].mean() * 252  # Annualized
            contributions[factor] = betas[factor] * factor_mean
        
        # Alpha contribution
        contributions['alpha'] = results.params['const'] * 252
        
        # Total explained
        contributions['total_explained'] = sum(contributions.values())
        
        return contributions
    
    def rolling_factor_analysis(
        self,
        window: int = 63,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform rolling factor analysis.
        
        Args:
            window: Rolling window size
            factor_names: Factors to include
            
        Returns:
            DataFrame of rolling alphas and betas
        """
        dates = self.returns.index
        results_list = []
        
        for i in range(window, len(dates)):
            start_idx = i - window
            end_idx = i
            
            y_window = self.excess_returns().iloc[start_idx:end_idx]
            X_window = self.factors.iloc[start_idx:end_idx]
            
            if factor_names is not None:
                X_window = X_window[factor_names]
            
            X_window = sm.add_constant(X_window)
            
            try:
                model = OLS(y_window, X_window)
                res = model.fit()
                
                row = {'date': dates[i]}
                row['alpha'] = res.params['const']
                
                for factor in X_window.columns:
                    if factor != 'const':
                        row[f'beta_{factor}'] = res.params[factor]
                
                results_list.append(row)
            except Exception:
                continue
        
        return pd.DataFrame(results_list).set_index('date')


def fama_french_3factor(
    returns: pd.Series,
    market_return: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
) -> FactorAnalysis:
    """
    Create FactorAnalysis for Fama-French 3-factor model.
    
    Args:
        returns: Portfolio returns
        market_return: Market factor return
        smb: Small minus Book (size factor)
        hml: High minus Low (value factor)
        risk_free_rate: Risk-free rate
        
    Returns:
        FactorAnalysis object
    """
    factors = pd.DataFrame({
        'MKT': market_return,
        'SMB': smb,
        'HML': hml,
    })
    
    return FactorAnalysis(returns, factors, risk_free_rate)


def fama_french_5factor(
    returns: pd.Series,
    market_return: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    rmw: pd.Series,  # Robust minus Weak (profitability)
    cma: pd.Series,  # Conservative minus Aggressive (investment)
    risk_free_rate: Optional[pd.Series] = None,
) -> FactorAnalysis:
    """
    Create FactorAnalysis for Fama-French 5-factor model.
    
    Args:
        returns: Portfolio returns
        market_return: Market factor
        smb: Size factor
        hml: Value factor
        rmw: Profitability factor
        cma: Investment factor
        risk_free_rate: Risk-free rate
        
    Returns:
        FactorAnalysis object
    """
    factors = pd.DataFrame({
        'MKT': market_return,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma,
    })
    
    return FactorAnalysis(returns, factors, risk_free_rate)
