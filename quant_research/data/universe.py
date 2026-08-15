"""
Universe Manager

Manages the asset universe to handle survivorship bias and ensure
point-in-time correct asset selection.
"""

from typing import Optional, Dict, Any, Union, List, Set
import pandas as pd
import numpy as np


class UniverseManager:
    """
    Manage the asset universe for backtesting.
    
    This class handles:
    - Dynamic universe composition (assets entering/leaving)
    - Survivorship bias mitigation
    - Point-in-time correct universe membership
    
    Attributes:
        prices: Price DataFrame
        min_history: Minimum history required for an asset to be included
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        min_history: int = 252,
    ) -> None:
        """
        Initialize the UniverseManager.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
            min_history: Minimum number of days of history required
        """
        self.prices = prices
        self.min_history = min_history
        self._universe_cache: Dict[pd.Timestamp, List[str]] = {}
    
    def get_universe_at_date(self, date: pd.Timestamp) -> List[str]:
        """
        Get the list of assets available at a specific date.
        
        An asset is included if it has sufficient price history
        up to and including the given date.
        
        Args:
            date: The date for which to get the universe
            
        Returns:
            List of ticker symbols in the universe
        """
        if date in self._universe_cache:
            return self._universe_cache[date]
        
        # Get prices up to and including the date
        mask = self.prices.index <= date
        prices_up_to_date = self.prices[mask]
        
        # Check which assets have sufficient history
        valid_assets = []
        for ticker in prices_up_to_date.columns:
            # Count non-null observations
            non_null_count = prices_up_to_date[ticker].notna().sum()
            if non_null_count >= self.min_history:
                # Also check that the asset has a price on the current date
                if pd.notna(prices_up_to_date.loc[date, ticker]):
                    valid_assets.append(ticker)
        
        self._universe_cache[date] = valid_assets
        return valid_assets
    
    def get_universe_series(self) -> pd.DataFrame:
        """
        Get a time series of universe membership.
        
        Returns:
            DataFrame with dates as index and boolean columns indicating membership
        """
        dates = self.prices.index
        tickers = self.prices.columns
        
        membership = pd.DataFrame(
            False,
            index=dates,
            columns=tickers,
        )
        
        for date in dates:
            universe = self.get_universe_at_date(date)
            membership.loc[date, universe] = True
        
        return membership
    
    def filter_to_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a DataFrame to only include assets in the universe at each date.
        
        Args:
            df: DataFrame to filter (should have same index as prices)
            
        Returns:
            Filtered DataFrame with NaN for assets not in universe
        """
        result = df.copy()
        membership = self.get_universe_series()
        
        # Ensure alignment
        membership = membership.reindex(result.index, columns=result.columns)
        
        # Set values to NaN for assets not in universe
        result[~membership] = np.nan
        
        return result


def create_point_in_time_universe(
    prices: pd.DataFrame,
    lookback_period: int = 252,
) -> pd.DataFrame:
    """
    Create a point-in-time correct universe mask.
    
    Args:
        prices: DataFrame of prices
        lookback_period: Required lookback period in days
        
    Returns:
        Boolean DataFrame indicating universe membership
    """
    manager = UniverseManager(prices, min_history=lookback_period)
    return manager.get_universe_series()


def detect_universe_changes(
    prices: pd.DataFrame,
    frequency: str = 'M',
) -> pd.DataFrame:
    """
    Detect changes in universe composition over time.
    
    Args:
        prices: DataFrame of prices
        frequency: Resampling frequency ('M' for month-end, 'Q' for quarter-end)
        
    Returns:
        DataFrame showing universe changes at each period
    """
    # Get universe at period ends
    period_dates = prices.resample(frequency).last().index
    
    changes = []
    prev_universe: Set[str] = set()
    
    for date in period_dates:
        current_prices = prices[prices.index <= date]
        current_universe = set(current_prices.dropna(axis=1, how='all').columns)
        
        added = current_universe - prev_universe
        removed = prev_universe - current_universe
        
        changes.append({
            'date': date,
            'universe_size': len(current_universe),
            'added_count': len(added),
            'removed_count': len(removed),
            'added_tickers': list(added),
            'removed_tickers': list(removed),
        })
        
        prev_universe = current_universe
    
    return pd.DataFrame(changes)
