"""
Main Pipeline Entry Point

Runs the complete quant research pipeline from data loading through 
out-of-sample validation and report generation.
"""

from typing import Optional, Dict, Any, Union
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from quant_research.data.loader import DataLoader, create_sample_data
from quant_research.signals.momentum import MomentumSignal, CrossSectionalMomentum
from quant_research.signals.mean_reversion import MeanReversionSignal
from quant_research.signals.statistical_tests import (
    test_signal_significance,
    compute_ic_statistics,
)
from quant_research.portfolio.constructor import PortfolioConstructor
from quant_research.portfolio.risk_controls import (
    VolatilityTarget,
    DrawdownControl,
    PositionLimits,
)
from quant_research.backtest.engine import BacktestEngine, TransactionCostModel
from quant_research.analytics.metrics import PerformanceMetrics
from quant_research.analytics.risk import RiskAnalysis, DrawdownAnalysis


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = 'config/default.yaml',
    generate_report: bool = True,
    output_dir: str = 'reports/',
) -> Dict[str, Any]:
    """
    Run the complete research pipeline.
    
    Args:
        config_path: Path to configuration YAML file
        generate_report: Whether to generate output report
        output_dir: Directory for output files
        
    Returns:
        Dictionary containing all results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    np.random.seed(config.get('random_seed', 42))
    
    print("=" * 60)
    print("QUANT RESEARCH PIPELINE")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Data Loading
    # =========================================================================
    print("\n[1/6] Loading data...")
    
    # For demonstration, use simulated data
    prices = create_sample_data(n_assets=50, n_days=3500, seed=config['random_seed'])
    
    # Split into in-sample and out-of-sample periods
    in_sample_end = pd.Timestamp(config['periods']['in_sample_end'])
    out_of_sample_start = pd.Timestamp(config['periods']['out_of_sample_start'])
    
    prices_is = prices[prices.index <= in_sample_end]
    prices_oos = prices[prices.index >= out_of_sample_start]
    
    print(f"  Total period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  In-sample: {prices_is.index[0].date()} to {prices_is.index[-1].date()}")
    print(f"  Out-of-sample: {prices_oos.index[0].date()} to {prices_oos.index[-1].date()}")
    
    # Calculate returns
    returns = prices.pct_change()
    returns_is = returns[returns.index <= in_sample_end]
    returns_oos = returns[returns.index >= out_of_sample_start]
    
    # =========================================================================
    # STEP 2: Signal Computation
    # =========================================================================
    print("\n[2/6] Computing signals...")
    
    signal_config = config['signal']
    
    if signal_config['type'] == 'momentum':
        signal_obj = CrossSectionalMomentum(
            lookback=signal_config['lookback'],
            skip=signal_config['skip'],
            method=signal_config['method'],
        )
    elif signal_config['type'] == 'mean_reversion':
        signal_obj = MeanReversionSignal(
            window=signal_config.get('window', 21),
        )
    else:
        signal_obj = MomentumSignal(
            lookback=signal_config['lookback'],
            skip=signal_config['skip'],
        )
    
    signals = signal_obj.compute(prices)
    signals_is = signals[signals.index <= in_sample_end]
    signals_oos = signals[signals.index >= out_of_sample_start]
    
    # Standardize signals
    signals = signal_obj.standardize(signals, method=signal_config.get('standardization', 'zscore'))
    signals_is = signal_obj.standardize(signals_is, method=signal_config.get('standardization', 'zscore'))
    signals_oos = signal_obj.standardize(signals_oos, method=signal_config.get('standardization', 'zscore'))
    
    # Lag signals to prevent look-ahead bias
    signals = signal_obj.lag(signals, periods=1)
    signals_is = signal_obj.lag(signals_is, periods=1)
    signals_oos = signal_obj.lag(signals_oos, periods=1)
    
    print(f"  Signal type: {signal_config['type']}")
    print(f"  Signal range: [{signals.min().min():.4f}, {signals.max().max():.4f}]")
    
    # Test statistical significance (in-sample only)
    forward_returns = returns.shift(-1)  # Next day returns
    sig_test = test_signal_significance(
        signals_is,
        forward_returns[forward_returns.index <= in_sample_end],
        method=config['testing']['significance_method'],
        lags=config['testing']['newey_west_lags'],
    )
    
    print(f"\n  Signal Significance (In-Sample):")
    print(f"    Beta: {sig_test['beta']:.6f}")
    print(f"    T-stat: {sig_test['t_stat']:.3f}")
    print(f"    P-value: {sig_test['p_value']:.6f}")
    print(f"    R-squared: {sig_test['r_squared']:.6f}")
    
    # IC analysis
    ic_stats = compute_ic_statistics(
        signals_is,
        forward_returns[forward_returns.index <= in_sample_end],
        method=config['testing']['ic_method'],
    )
    
    print(f"\n  Information Coefficient Analysis:")
    print(f"    Mean IC: {ic_stats['mean_ic']:.4f}")
    print(f"    IC T-stat: {ic_stats['ic_tstat']:.3f}")
    print(f"    Hit Rate: {ic_stats['hit_rate']:.2%}")
    
    # =========================================================================
    # STEP 3: Portfolio Construction
    # =========================================================================
    print("\n[3/6] Constructing portfolio...")
    
    port_config = config['portfolio']
    
    constructor = PortfolioConstructor(
        method=port_config['method'],
        max_weight=port_config.get('max_weight'),
        target_gross=port_config.get('target_gross', 1.0),
    )
    
    weights = constructor.build(signals)
    weights_is = weights[weights.index <= in_sample_end]
    weights_oos = weights[weights.index >= out_of_sample_start]
    
    # Apply risk controls
    risk_config = config['risk_controls']
    
    if risk_config['volatility_target']['enabled']:
        vol_target = VolatilityTarget(
            target_vol=risk_config['volatility_target']['target_vol'],
            vol_window=risk_config['volatility_target']['vol_window'],
            max_leverage=risk_config['volatility_target']['max_leverage'],
        )
        weights = vol_target.apply(weights, returns)
    
    if risk_config['drawdown_control']['enabled']:
        dd_control = DrawdownControl(
            max_drawdown=risk_config['drawdown_control']['max_drawdown'],
            trigger_dd=risk_config['drawdown_control']['trigger_dd'],
        )
        # Will be applied dynamically during backtest
    
    print(f"  Construction method: {port_config['method']}")
    print(f"  Average gross exposure: {weights.abs().sum(axis=1).mean():.2f}")
    
    # =========================================================================
    # STEP 4: Backtest
    # =========================================================================
    print("\n[4/6] Running backtest...")
    
    cost_model = TransactionCostModel(
        spread=config['costs']['spread'],
        slippage=config['costs']['slippage'],
        commission=config['costs']['commission'],
    )
    
    engine = BacktestEngine(
        cost_model=cost_model,
        rebalance_frequency=port_config.get('rebalance_frequency', 'M'),
    )
    
    # Create benchmark (equal-weight market)
    benchmark_returns = returns.mean(axis=1)
    
    # Run full backtest
    results = engine.run(weights, prices, benchmark_returns)
    
    # Split results
    ret_is = results['portfolio_returns'][results['portfolio_returns'].index <= in_sample_end]
    ret_oos = results['portfolio_returns'][results['portfolio_returns'].index >= out_of_sample_start]
    
    bench_is = benchmark_returns[benchmark_returns.index <= in_sample_end]
    bench_oos = benchmark_returns[benchmark_returns.index >= out_of_sample_start]
    
    print(f"  Total transaction costs: {results['total_costs']:.4f}")
    print(f"  Average turnover: {results['average_turnover']:.4f}")
    
    # =========================================================================
    # STEP 5: Performance Analytics
    # =========================================================================
    print("\n[5/6] Calculating performance metrics...")
    
    # In-sample metrics
    metrics_is = PerformanceMetrics(ret_is)
    is_metrics = metrics_is.all_metrics()
    
    # Out-of-sample metrics
    metrics_oos = PerformanceMetrics(ret_oos)
    oos_metrics = metrics_oos.all_metrics()
    
    # Benchmark metrics
    bench_metrics = PerformanceMetrics(benchmark_returns)
    bench_full = bench_metrics.all_metrics()
    
    print("\n  PERFORMANCE SUMMARY")
    print("  " + "-" * 50)
    print(f"  {'Metric':<25} {'In-Sample':>12} {'OOS':>12} {'Benchmark':>12}")
    print("  " + "-" * 50)
    print(f"  {'Sharpe Ratio':<25} {is_metrics['sharpe_ratio']:>12.3f} {oos_metrics['sharpe_ratio']:>12.3f} {bench_full['sharpe_ratio']:>12.3f}")
    print(f"  {'Annualized Return':<25} {is_metrics['annualized_return']:>12.2%} {oos_metrics['annualized_return']:>12.2%} {bench_full['annualized_return']:>12.2%}")
    print(f"  {'Annualized Volatility':<25} {is_metrics['annualized_volatility']:>12.2%} {oos_metrics['annualized_volatility']:>12.2%} {bench_full['annualized_volatility']:>12.2%}")
    print(f"  {'Max Drawdown':<25} {is_metrics['max_drawdown']:>12.2%} {oos_metrics['max_drawdown']:>12.2%} {bench_full['max_drawdown']:>12.2%}")
    print(f"  {'Calmar Ratio':<25} {is_metrics['calmar_ratio']:>12.3f} {oos_metrics['calmar_ratio']:>12.3f} {bench_full['calmar_ratio']:>12.3f}")
    print(f"  {'Hit Rate':<25} {is_metrics['hit_rate']:>12.2%} {oos_metrics['hit_rate']:>12.2%} {bench_full['hit_rate']:>12.2%}")
    print("  " + "-" * 50)
    
    # IS vs OOS comparison
    sharpe_degradation = (is_metrics['sharpe_ratio'] - oos_metrics['sharpe_ratio']) / is_metrics['sharpe_ratio'] if is_metrics['sharpe_ratio'] != 0 else 0
    print(f"\n  Sharpe Ratio Degradation (IS to OOS): {sharpe_degradation:.2%}")
    
    if sharpe_degradation > 0.5:
        print("  ⚠️  WARNING: Significant degradation suggests potential overfitting!")
    elif sharpe_degradation > 0.3:
        print("  ⚡ Moderate degradation - monitor closely")
    else:
        print("  ✓ Good consistency between IS and OOS")
    
    # Risk analysis
    risk_analysis = RiskAnalysis(results['portfolio_returns'])
    risk_metrics = risk_analysis.risk_metrics()
    
    print(f"\n  RISK METRICS")
    print(f"    VaR (95%): {risk_metrics['var_95']:.2%}")
    print(f"    CVaR (95%): {risk_metrics['cvar_95']:.2%}")
    print(f"    Tail Ratio: {risk_metrics['tail_ratio']:.3f}")
    print(f"    Omega Ratio: {risk_metrics['omega_ratio']:.3f}")
    
    # Drawdown analysis
    dd_analysis = DrawdownAnalysis(results['portfolio_returns'])
    dd_summary = dd_analysis.summary()
    
    print(f"\n  DRAWDOWN ANALYSIS")
    print(f"    Max Drawdown: {dd_summary['max_drawdown']:.2%}")
    print(f"    Avg Drawdown: {dd_summary['avg_drawdown']:.2%}")
    print(f"    Max Duration: {dd_summary['max_duration_days']} days")
    print(f"    Max Recovery: {dd_summary['max_recovery_days']} days")
    
    # =========================================================================
    # STEP 6: Compile Results
    # =========================================================================
    print("\n[6/6] Compiling results...")
    
    all_results = {
        'config': config,
        'in_sample': {
            'returns': ret_is,
            'metrics': is_metrics,
            'signal_test': sig_test,
            'ic_stats': ic_stats,
        },
        'out_of_sample': {
            'returns': ret_oos,
            'metrics': oos_metrics,
        },
        'full_period': {
            'portfolio_returns': results['portfolio_returns'],
            'cumulative_returns': results['cumulative_returns'],
            'benchmark_returns': benchmark_returns,
            'weights': results['weights_history'],
            'turnover': results['turnover'],
            'costs': results['costs'],
        },
        'risk': {
            'metrics': risk_metrics,
            'drawdown': dd_summary,
        },
        'is_oos_comparison': {
            'sharpe_degradation': sharpe_degradation,
            'return_difference': is_metrics['annualized_return'] - oos_metrics['annualized_return'],
            'vol_difference': is_metrics['annualized_volatility'] - oos_metrics['annualized_volatility'],
        }
    }
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return all_results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Run Quant Research Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    args = parser.parse_args()
    
    results = run_pipeline(
        config_path=args.config,
        generate_report=not args.no_report,
        output_dir=args.output_dir,
    )
    
    return results


if __name__ == '__main__':
    main()
