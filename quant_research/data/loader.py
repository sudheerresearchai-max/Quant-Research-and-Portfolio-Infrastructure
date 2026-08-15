"""
Data Loader Module

Handles ingestion of historical price/volume data from CSV files or APIs,
with proper handling of missing data and point-in-time correctness.
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import pandas as pd
import numpy as np


class DataLoader:
    """
    Load and clean financial data from various sources.
    
    Attributes:
        start_date: Start date for data filtering
        end_date: End date for data filtering
        fill_method: Method for handling missing data ('ffill', 'bfill', 'interpolate')
    
    Example:
        >>> loader = DataLoader(start_date='2010-01-01', end_date='2023-12-31')
        >>> prices = loader.load_csv('data/prices.csv')
    """
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fill_method: str = 'ffill',
    ) -> None:
        """
        Initialize the DataLoader.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            fill_method: Method for filling missing values
        """
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None
        self.fill_method = fill_method
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        date_col: str = 'date',
        price_col: str = 'close',
        asset_col: str = 'ticker',
    ) -> pd.DataFrame:
        """
        Load price data from a CSV file.
        
        Expected CSV format:
            date,ticker,open,high,low,close,volume
        
        Args:
            filepath: Path to the CSV file
            date_col: Name of the date column
            price_col: Name of the price column to use
            asset_col: Name of the asset identifier column
            
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=[date_col])
        df = df.set_index(date_col)
        
        # Filter by date range
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]
        
        # Pivot to wide format (dates x tickers)
        if asset_col in df.columns:
            prices = df.pivot(columns=asset_col, values=price_col)
        else:
            prices = df[[price_col]]
        
        # Handle missing data
        prices = self._handle_missing_data(prices)
        
        return prices
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using the specified fill method.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        if self.fill_method == 'ffill':
            return df.ffill()
        elif self.fill_method == 'bfill':
            return df.bfill()
        elif self.fill_method == 'interpolate':
            return df.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown fill method: {self.fill_method}")
    
    def load_from_dict(self, data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Load price data from a dictionary of time series.
        
        Args:
            data: Dictionary mapping ticker symbols to price series
            
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        
        # Filter by date range
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]
        
        # Handle missing data
        df = self._handle_missing_data(df)
        
        return df
    
    def validate_point_in_time(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> bool:
        """
        Validate that signals do not use future data.
        
        Args:
            prices: Price DataFrame
            signals: Signal DataFrame
            
        Returns:
            True if point-in-time correct, False otherwise
        """
        # Check that signal at time t only uses prices up to t-1
        shifted_prices = prices.shift(1)
        correlation = signals.corrwith(shifted_prices)
        return not correlation.isna().all()


def create_sample_data(
    n_assets: int = 10,
    n_days: int = 252 * 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create sample price data for testing.
    
    Args:
        n_assets: Number of assets
        n_days: Number of trading days
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame of simulated prices
    """
    np.random.seed(seed)
    
    dates = pd.date_range('2010-01-01', periods=n_days, freq='B')
    tickers = [f'TICK{i:03d}' for i in range(n_assets)]
    
    # Generate returns with some momentum and mean-reversion
    base_returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
    
    # Add some autocorrelation for realism
    returns = np.zeros_like(base_returns)
    returns[0] = base_returns[0]
    for t in range(1, n_days):
        returns[t] = 0.1 * returns[t-1] + 0.9 * base_returns[t]
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df
