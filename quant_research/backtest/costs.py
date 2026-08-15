"""
Transaction Cost Model

Models trading costs including:
- Bid-ask spread
- Market impact/slippage
- Commissions/fees
"""

from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class TransactionCostModel:
    """
    Model transaction costs for portfolio trades.
    
    Costs include:
    - Spread: Half the bid-ask spread (paid on each trade)
    - Slippage: Market impact proportional to trade size
    - Commission: Fixed percentage fee
    
    Attributes:
        spread: Bid-ask spread (e.g., 0.001 for 10 bps)
        slippage: Market impact coefficient
        commission: Commission rate
    """
    
    def __init__(
        self,
        spread: float = 0.001,
        slippage: float = 0.0005,
        commission: float = 0.001,
        min_cost: float = 0.0,
        max_cost: Optional[float] = None,
    ) -> None:
        """
        Initialize transaction cost model.
        
        Args:
            spread: Bid-ask spread (as fraction of price)
            slippage: Slippage coefficient (cost per unit of turnover)
            commission: Commission rate (as fraction of trade value)
            min_cost: Minimum cost per trade
            max_cost: Maximum cost per trade (None for no limit)
        """
        self.spread = spread
        self.slippage = slippage
        self.commission = commission
        self.min_cost = min_cost
        self.max_cost = max_cost
    
    def calculate_costs(
        self,
        weights_prev: pd.Series,
        weights_target: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate total transaction cost for a rebalance.
        
        Args:
            weights_prev: Previous portfolio weights
            weights_target: Target portfolio weights
            prices: Optional current prices (for more accurate modeling)
            
        Returns:
            Total transaction cost as fraction of portfolio value
        """
        # Calculate turnover
        turnover = (weights_target - weights_prev).abs().sum() / 2
        
        # Spread cost (paid on half turnover for each side)
        spread_cost = turnover * self.spread
        
        # Slippage cost (increases with turnover)
        slippage_cost = turnover * self.slippage
        
        # Commission (on total turnover)
        commission_cost = turnover * self.commission
        
        # Total cost
        total_cost = spread_cost + slippage_cost + commission_cost
        
        # Apply bounds
        total_cost = max(total_cost, self.min_cost)
        if self.max_cost is not None:
            total_cost = min(total_cost, self.max_cost)
        
        return total_cost
    
    def calculate_turnover(
        self,
        weights_prev: pd.Series,
        weights_target: pd.Series,
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            
        Returns:
            Turnover as fraction of portfolio value
        """
        return (weights_target - weights_prev).abs().sum() / 2
    
    def get_cost_breakdown(
        self,
        weights_prev: pd.Series,
        weights_target: pd.Series,
    ) -> Dict[str, float]:
        """
        Get breakdown of transaction costs.
        
        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            
        Returns:
            Dictionary with cost components
        """
        turnover = self.calculate_turnover(weights_prev, weights_target)
        
        return {
            'turnover': turnover,
            'spread_cost': turnover * self.spread,
            'slippage_cost': turnover * self.slippage,
            'commission_cost': turnover * self.commission,
            'total_cost': self.calculate_costs(weights_prev, weights_target),
        }


class VariableTransactionCostModel(TransactionCostModel):
    """
    Transaction cost model with asset-specific costs.
    
    Allows different costs for different assets based on
    liquidity, market cap, or other characteristics.
    """
    
    def __init__(
        self,
        base_spread: float = 0.001,
        base_slippage: float = 0.0005,
        base_commission: float = 0.001,
        liquidity_multiplier: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize variable cost model.
        
        Args:
            base_spread: Base bid-ask spread
            base_slippage: Base slippage coefficient
            base_commission: Base commission rate
            liquidity_multiplier: Multiplier for each asset (higher = less liquid = more costly)
        """
        super().__init__(
            spread=base_spread,
            slippage=base_slippage,
            commission=base_commission,
        )
        self.liquidity_multiplier = liquidity_multiplier or {}
    
    def calculate_costs(
        self,
        weights_prev: pd.Series,
        weights_target: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate costs with asset-specific adjustments.
        
        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            prices: Optional prices
            
        Returns:
            Total transaction cost
        """
        total_cost = 0.0
        
        for ticker in weights_target.index:
            w_prev = weights_prev.get(ticker, 0)
            w_target = weights_target.get(ticker, 0)
            
            # Asset-specific turnover
            turnover = abs(w_target - w_prev)
            
            # Get liquidity multiplier
            liq_mult = self.liquidity_multiplier.get(ticker, 1.0)
            
            # Calculate cost for this asset
            asset_cost = turnover * (
                self.spread * liq_mult +
                self.slippage * liq_mult +
                self.commission
            )
            
            total_cost += asset_cost
        
        # Divide by 2 because we're counting both sides
        total_cost /= 2
        
        # Apply bounds
        total_cost = max(total_cost, self.min_cost)
        if self.max_cost is not None:
            total_cost = min(total_cost, self.max_cost)
        
        return total_cost


def estimate_annual_cost_drag(
    annual_turnover: float,
    spread: float = 0.001,
    slippage: float = 0.0005,
    commission: float = 0.001,
) -> float:
    """
    Estimate annual cost drag from turnover.
    
    Args:
        annual_turnover: Expected annual turnover
        spread: Average spread
        slippage: Average slippage
        commission: Average commission
        
    Returns:
        Estimated annual cost drag (as fraction)
    """
    return annual_turnover * (spread + slippage + commission)
