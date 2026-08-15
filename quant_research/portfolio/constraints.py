"""
Portfolio Constraints

Additional constraint classes for portfolio construction.
"""

from typing import Optional, Dict, Any, Union, List
import pandas as pd
import numpy as np


class SectorConstraints:
    """
    Enforce sector-level position limits.
    
    Attributes:
        sector_map: Dictionary mapping tickers to sectors
        max_sector_weight: Maximum weight per sector
    """
    
    def __init__(
        self,
        sector_map: Dict[str, str],
        max_sector_weight: float = 0.30,
    ) -> None:
        """
        Initialize sector constraints.
        
        Args:
            sector_map: Mapping of ticker to sector name
            max_sector_weight: Maximum total weight per sector
        """
        self.sector_map = sector_map
        self.max_sector_weight = max_sector_weight
    
    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sector constraints to weights.
        
        Args:
            weights: Raw weights
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        # Group by sector
        for date in weights.index:
            date_weights = weights.loc[date].dropna()
            
            # Calculate sector weights
            sector_weights = {}
            for ticker, weight in date_weights.items():
                sector = self.sector_map.get(ticker, 'Other')
                if sector not in sector_weights:
                    sector_weights[sector] = []
                sector_weights[sector].append((ticker, weight))
            
            # Check each sector
            for sector, positions in sector_weights.items():
                total_weight = sum(abs(w) for _, w in positions)
                
                if total_weight > self.max_sector_weight:
                    # Scale down proportionally
                    scale = self.max_sector_weight / total_weight
                    for ticker, _ in positions:
                        constrained.loc[date, ticker] = weights.loc[date, ticker] * scale
        
        return constrained


class LeverageConstraint:
    """
    Enforce leverage constraints on portfolio.
    
    Attributes:
        max_gross_leverage: Maximum gross exposure (sum of absolute weights)
        max_net_exposure: Maximum net exposure (long - short)
    """
    
    def __init__(
        self,
        max_gross_leverage: float = 1.5,
        max_net_exposure: Optional[float] = None,
    ) -> None:
        """
        Initialize leverage constraint.
        
        Args:
            max_gross_leverage: Maximum gross exposure
            max_net_exposure: Maximum net exposure (None for no limit)
        """
        self.max_gross_leverage = max_gross_leverage
        self.max_net_exposure = max_net_exposure
    
    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply leverage constraints.
        
        Args:
            weights: Raw weights
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        # Gross leverage constraint
        gross = weights.abs().sum(axis=1)
        excess_gross = gross > self.max_gross_leverage
        
        if excess_gross.any():
            scale = self.max_gross_leverage / gross[excess_gross]
            constrained.loc[excess_gross] = weights.loc[excess_gross].mul(scale, axis=0)
        
        # Net exposure constraint
        if self.max_net_exposure is not None:
            net = weights.sum(axis=1).abs()
            excess_net = net > self.max_net_exposure
            
            if excess_net.any():
                scale = self.max_net_exposure / net[excess_net]
                constrained.loc[excess_net] = constrained.loc[excess_net].mul(scale, axis=0)
        
        return constrained


class TurnoverConstraint:
    """
    Limit portfolio turnover between periods.
    
    Attributes:
        max_turnover: Maximum allowed turnover per period
    """
    
    def __init__(self, max_turnover: float = 0.50) -> None:
        """
        Initialize turnover constraint.
        
        Args:
            max_turnover: Maximum turnover (e.g., 0.50 for 50%)
        """
        self.max_turnover = max_turnover
    
    def apply(
        self,
        target_weights: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.DataFrame:
        """
        Apply turnover constraint.
        
        Args:
            target_weights: Target weights
            current_weights: Current portfolio weights
            
        Returns:
            Adjusted target weights
        """
        adjusted = target_weights.copy()
        
        for date in target_weights.index:
            target = target_weights.loc[date]
            
            # Calculate required turnover
            turnover = (target - current_weights).abs().sum() / 2
            
            if turnover > self.max_turnover:
                # Scale down the changes
                scale = self.max_turnover / turnover
                adjustment = (target - current_weights) * scale
                adjusted.loc[date] = current_weights + adjustment
            
            current_weights = adjusted.loc[date]
        
        return adjusted


class ConcentrationConstraint:
    """
    Limit portfolio concentration (Herfindahl index).
    
    Attributes:
        max_herfindahl: Maximum Herfindahl index
        max_top_n: Maximum weight in top N positions
    """
    
    def __init__(
        self,
        max_herfindahl: Optional[float] = None,
        max_top_n: Optional[tuple] = None,
    ) -> None:
        """
        Initialize concentration constraint.
        
        Args:
            max_herfindahl: Maximum Herfindahl index (sum of squared weights)
            max_top_n: Tuple of (n, max_weight) for top N positions
        """
        self.max_herfindahl = max_herfindahl
        self.max_top_n = max_top_n
    
    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply concentration constraints.
        
        Args:
            weights: Raw weights
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        if self.max_herfindahl is None and self.max_top_n is None:
            return constrained
        
        for date in weights.index:
            w = weights.loc[date].dropna()
            
            # Herfindahl constraint
            if self.max_herfindahl is not None:
                herf = (w ** 2).sum()
                
                while herf > self.max_herfindahl and len(w) > 1:
                    # Find largest position and reduce it
                    max_idx = w.abs().idxmax()
                    reduction = 0.01
                    w[max_idx] = w[max_idx] * (1 - reduction)
                    herf = (w ** 2).sum()
            
            # Top N constraint
            if self.max_top_n is not None:
                n, max_weight = self.max_top_n
                sorted_w = w.abs().sort_values(ascending=False)
                
                for i in range(min(n, len(sorted_w))):
                    idx = sorted_w.index[i]
                    if abs(w[idx]) > max_weight:
                        w[idx] = np.sign(w[idx]) * max_weight
            
            constrained.loc[date, w.index] = w.values
        
        return constrained
