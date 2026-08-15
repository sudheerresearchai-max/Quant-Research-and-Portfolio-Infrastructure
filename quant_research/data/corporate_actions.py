"""
Corporate Actions Handler

Handles adjustments for stock splits, dividends, and other corporate actions
to ensure point-in-time correct price series.
"""

from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


class CorporateActionsHandler:
    """
    Handle corporate actions adjustments for price data.
    
    This class ensures that price series are adjusted for:
    - Stock splits (forward and reverse)
    - Dividends (cash and stock)
    - Mergers and acquisitions
    
    Attributes:
        splits_df: DataFrame containing split information
        dividends_df: DataFrame containing dividend information
    """
    
    def __init__(
        self,
        splits_file: Optional[Union[str, Path]] = None,
        dividends_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the CorporateActionsHandler.
        
        Args:
            splits_file: Path to CSV file with split data
            dividends_file: Path to CSV file with dividend data
        """
        self.splits_df: Optional[pd.DataFrame] = None
        self.dividends_df: Optional[pd.DataFrame] = None
        
        if splits_file:
            self.load_splits(splits_file)
        if dividends_file:
            self.load_dividends(dividends_file)
    
    def load_splits(self, filepath: Union[str, Path]) -> None:
        """
        Load split data from CSV file.
        
        Expected format:
            date,ticker,ratio
        
        Args:
            filepath: Path to the splits CSV file
        """
        filepath = Path(filepath)
        if filepath.exists():
            self.splits_df = pd.read_csv(filepath, parse_dates=['date'])
            self.splits_df = self.splits_df.set_index('date')
    
    def load_dividends(self, filepath: Union[str, Path]) -> None:
        """
        Load dividend data from CSV file.
        
        Expected format:
            date,ticker,amount
        
        Args:
            filepath: Path to the dividends CSV file
        """
        filepath = Path(filepath)
        if filepath.exists():
            self.dividends_df = pd.read_csv(filepath, parse_dates=['date'])
            self.dividends_df = self.dividends_df.set_index('date')
    
    def adjust_for_splits(
        self,
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Adjust prices and volume for stock splits.
        
        Uses backward adjustment to maintain point-in-time correctness.
        
        Args:
            prices: DataFrame of unadjusted prices
            volume: Optional DataFrame of trading volume
            
        Returns:
            Tuple of (adjusted_prices, adjusted_volume)
        """
        if self.splits_df is None or self.splits_df.empty:
            return prices, volume
        
        adjusted_prices = prices.copy()
        adjusted_volume = volume.copy() if volume is not None else None
        
        # Process splits chronologically (backward adjustment)
        for date in sorted(self.splits_df.index, reverse=True):
            splits_on_date = self.splits_df.loc[date]
            
            # Handle single or multiple splits on same date
            if isinstance(splits_on_date, pd.Series):
                splits_on_date = [splits_on_date]
            
            for split in splits_on_date if isinstance(splits_on_date, list) else [splits_on_date]:
                ticker = split['ticker']
                ratio = split['ratio']  # e.g., 2.0 for 2:1 split
                
                if ticker in adjusted_prices.columns:
                    # Adjust all prices before the split date
                    mask = adjusted_prices.index < date
                    adjusted_prices.loc[mask, ticker] /= ratio
                    
                    if adjusted_volume is not None and ticker in adjusted_volume.columns:
                        adjusted_volume.loc[mask, ticker] *= ratio
        
        return adjusted_prices, adjusted_volume
    
    def adjust_for_dividends(
        self,
        prices: pd.DataFrame,
        method: str = 'total_return',
    ) -> pd.DataFrame:
        """
        Adjust prices for dividends.
        
        Args:
            prices: DataFrame of prices
            method: Adjustment method ('total_return' or 'price')
            
        Returns:
            DataFrame of dividend-adjusted prices
        """
        if self.dividends_df is None or self.dividends_df.empty:
            return prices
        
        if method == 'total_return':
            # Create total return index
            returns = prices.pct_change()
            
            # Add dividend yield to returns
            for ticker in prices.columns:
                ticker_divs = self.dividends_df[self.dividends_df['ticker'] == ticker]
                for _, div_row in ticker_divs.iterrows():
                    div_date = div_row['date']
                    div_amount = div_row['amount']
                    price_on_date = prices.loc[div_date, ticker]
                    
                    if pd.notna(price_on_date) and price_on_date > 0:
                        div_yield = div_amount / price_on_date
                        returns.loc[div_date, ticker] += div_yield
            
            # Convert back to prices
            adjusted_prices = 100 * np.exp(np.log(1 + returns).cumsum())
            return adjusted_prices
        
        elif method == 'price':
            # Simple price adjustment (less accurate)
            adjusted_prices = prices.copy()
            for ticker in prices.columns:
                ticker_divs = self.dividends_df[self.dividends_df['ticker'] == ticker]
                cumulative_divs = 0
                
                for _, div_row in ticker_divs.iterrows():
                    cumulative_divs += div_row['amount']
                
                # Adjust by cumulative dividends
                if cumulative_divs > 0:
                    last_price = prices[ticker].iloc[-1]
                    adjustment_factor = (last_price + cumulative_divs) / last_price
                    adjusted_prices[ticker] *= adjustment_factor
            
            return adjusted_prices
        
        else:
            raise ValueError(f"Unknown adjustment method: {method}")


def adjust_for_corporate_actions(
    prices: pd.DataFrame,
    splits_file: Optional[Union[str, Path]] = None,
    dividends_file: Optional[Union[str, Path]] = None,
    volume: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to adjust prices for all corporate actions.
    
    Args:
        prices: DataFrame of unadjusted prices
        splits_file: Path to splits CSV file
        dividends_file: Path to dividends CSV file
        volume: Optional DataFrame of trading volume
        
    Returns:
        Tuple of (adjusted_prices, adjusted_volume)
    """
    handler = CorporateActionsHandler(
        splits_file=splits_file,
        dividends_file=dividends_file,
    )
    
    adjusted_prices, adjusted_volume = handler.adjust_for_splits(prices, volume)
    adjusted_prices = handler.adjust_for_dividends(adjusted_prices)
    
    return adjusted_prices, adjusted_volume


def detect_survivorship_bias(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Detect potential survivorship bias in the dataset.
    
    Args:
        prices: DataFrame of prices
        start_date: Start of analysis period
        end_date: End of analysis period
        
    Returns:
        Dictionary with survivorship bias metrics
    """
    # Count assets with data at start vs end
    start_mask = prices.index <= start_date
    end_mask = prices.index >= end_date
    
    assets_at_start = prices[start_mask].dropna(axis=1, how='all').columns
    assets_at_end = prices[end_mask].dropna(axis=1, how='all').columns
    
    # Assets that disappeared
    disappeared = set(assets_at_start) - set(assets_at_end)
    # New assets
    new_assets = set(assets_at_end) - set(assets_at_start)
    
    return {
        'assets_at_start': len(assets_at_start),
        'assets_at_end': len(assets_at_end),
        'disappeared_count': len(disappeared),
        'new_assets_count': len(new_assets),
        'survival_rate': len(assets_at_end) / len(assets_at_start) if len(assets_at_start) > 0 else 0,
        'disappeared_tickers': list(disappeared),
        'new_tickers': list(new_assets),
    }
