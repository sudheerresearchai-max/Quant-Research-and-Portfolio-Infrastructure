# Quant Research Pipeline

A reproducible quantitative research pipeline in Python for systematic equity/multi-asset strategy development, from raw data to out-of-sample performance reporting.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Testing](#testing)
- [Generating Reports](#generating-reports)
- [Architecture Diagram](#architecture-diagram)

## Overview

This pipeline implements a complete quantitative research workflow:

1. **Data Pipeline**: Ingest, clean, and prepare historical price/volume data with point-in-time correctness
2. **Signal Research**: Define, compute, and statistically test trading signals
3. **Portfolio Construction**: Convert signals to positions with risk controls
4. **Backtest Engine**: Vectorized backtesting with transaction costs
5. **Performance Analytics**: Comprehensive metrics and risk analysis
6. **Out-of-Sample Validation**: Hold-out period testing to detect overfitting

### Key Features

- ✅ Point-in-time correct data handling
- ✅ Corporate actions adjustment (splits/dividends)
- ✅ Survivorship bias mitigation
- ✅ Look-ahead bias prevention
- ✅ Transaction cost modeling (spread + slippage + commission)
- ✅ Volatility targeting
- ✅ Drawdown controls
- ✅ Position sizing with constraints
- ✅ Statistical significance testing (Newey-West adjusted)
- ✅ Factor exposure analysis
- ✅ Reproducible configuration via YAML

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scipy statsmodels matplotlib seaborn pyyaml pytest jupyter notebook
```

### Requirements File

The `requirements.txt` file contains all necessary dependencies:

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.12.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
pytest>=6.2.0
jupyter>=1.0.0
notebook>=6.4.0
```

## Project Structure

```
quant_research/
├── __init__.py              # Package initialization
├── data/
│   ├── __init__.py
│   ├── loader.py            # Data ingestion and cleaning
│   ├── corporate_actions.py # Handle splits/dividends
│   └── universe.py          # Asset universe management
├── signals/
│   ├── __init__.py
│   ├── base.py              # Base signal class
│   ├── momentum.py          # Momentum signals
│   ├── mean_reversion.py    # Mean-reversion signals
│   └── statistical_tests.py # Signal significance testing
├── portfolio/
│   ├── __init__.py
│   ├── constructor.py       # Position sizing and weights
│   ├── risk_controls.py     # Vol targeting, drawdown controls
│   └── constraints.py       # Position/sector constraints
├── backtest/
│   ├── __init__.py
│   ├── engine.py            # Vectorized backtester
│   └── costs.py             # Transaction cost model
├── analytics/
│   ├── __init__.py
│   ├── metrics.py           # Performance metrics
│   ├── risk.py              # Risk analysis (VaR, CVaR)
│   └── factors.py           # Factor exposure analysis
├── reporting/
│   ├── __init__.py
│   ├── generator.py         # Report generation
│   └── templates/           # Report templates
├── tests/
│   ├── __init__.py
│   ├── test_data.py         # Data pipeline tests
│   ├── test_signals.py      # Signal calculation tests
│   ├── test_portfolio.py    # Portfolio construction tests
│   └── test_backtest.py     # Backtest engine tests
├── config/
│   ├── default.yaml         # Default configuration
│   └── strategies/          # Strategy-specific configs
├── notebooks/
│   └── research_report.ipynb # Jupyter notebook report
└── main.py                  # Main pipeline entry point
```

## Quick Start

```python
from quant_research.main import run_pipeline

# Run the full pipeline with default configuration
results = run_pipeline(config_path='config/default.yaml')

# Access results
print(f"In-Sample Sharpe: {results['in_sample']['sharpe_ratio']:.3f}")
print(f"Out-of-Sample Sharpe: {results['out_of_sample']['sharpe_ratio']:.3f}")
```

## Pipeline Components

### 1. Data Pipeline

Handles data ingestion, cleaning, and preparation:

```python
from quant_research.data.loader import DataLoader
from quant_research.data.corporate_actions import adjust_for_corporate_actions

# Load and clean data
loader = DataLoader()
prices = loader.load_csv('data/prices.csv')
adjusted_prices = adjust_for_corporate_actions(prices, splits_file, dividends_file)
```

**Key Features:**
- Missing data imputation with forward-fill
- Corporate actions adjustment
- Survivorship bias handling via universe reconstruction
- Point-in-time correct alignment

### 2. Signal Research

Define and test trading signals:

```python
from quant_research.signals.momentum import MomentumSignal
from quant_research.signals.statistical_tests import test_signal_significance

# Create momentum signal
signal = MomentumSignal(lookback=252, skip=21)
signal_values = signal.compute(prices)

# Test statistical significance
results = test_signal_significance(signal_values, forward_returns, method='newey_west')
```

**Available Signals:**
- Momentum (time-series and cross-sectional)
- Mean-reversion
- Volatility risk premium
- Custom signals via base class

### 3. Portfolio Construction

Convert signals to target positions:

```python
from quant_research.portfolio.constructor import PortfolioConstructor
from quant_research.portfolio.risk_controls import VolatilityTarget, DrawdownControl

# Build portfolio
constructor = PortfolioConstructor(method='rank_zscore')
target_weights = constructor.build(signal_values)

# Apply risk controls
vol_target = VolatilityTarget(target_vol=0.10)
scaled_weights = vol_target.apply(target_weights, realized_vol)

# Apply drawdown control
dd_control = DrawdownControl(max_dd=0.15)
final_weights = dd_control.apply(scaled_weights, cumulative_returns)
```

### 4. Backtest Engine

Run vectorized backtest:

```python
from quant_research.backtest.engine import BacktestEngine
from quant_research.backtest.costs import TransactionCostModel

# Configure backtester
cost_model = TransactionCostModel(spread=0.001, slippage=0.0005, commission=0.001)
engine = BacktestEngine(cost_model=cost_model)

# Run backtest
results = engine.run(weights, prices, benchmark_returns)
```

### 5. Performance Analytics

Compute comprehensive metrics:

```python
from quant_research.analytics.metrics import PerformanceMetrics
from quant_research.analytics.risk import RiskAnalysis

# Calculate metrics
metrics = PerformanceMetrics(results['portfolio_returns'])
sharpe = metrics.sharpe_ratio()
max_dd = metrics.max_drawdown()
calmar = metrics.calmar_ratio()

# Risk analysis
risk = RiskAnalysis(results['portfolio_returns'])
var_95 = risk.var(confidence=0.95)
cvar_95 = risk.cvar(confidence=0.95)
```

### 6. Out-of-Sample Validation

Automatic IS/OOS split and comparison:

```python
# Configuration handles split dates
config = {
    'in_sample_end': '2018-12-31',
    'out_of_sample_start': '2019-01-01'
}

# Pipeline automatically validates on OOS period
results = run_pipeline(config)
comparison = results['is_oos_comparison']
```

## Configuration

All parameters are configurable via YAML:

```yaml
# config/default.yaml
data:
  price_file: "data/prices.csv"
  start_date: "2010-01-01"
  end_date: "2023-12-31"
  
periods:
  in_sample_end: "2018-12-31"
  out_of_sample_start: "2019-01-01"
  
signal:
  type: "momentum"
  lookback: 252
  skip: 21
  
portfolio:
  method: "rank_zscore"
  max_weight: 0.10
  target_volatility: 0.10
  max_drawdown: 0.15
  
costs:
  spread: 0.001
  slippage: 0.0005
  commission: 0.001
  
random_seed: 42
```

## Running the Pipeline

### Command Line

```bash
# Run with default config
python -m quant_research.main

# Run with custom config
python -m quant_research.main --config config/my_strategy.yaml

# Run specific component
python -m quant_research.main --step data
python -m quant_research.main --step signals
python -m quant_research.main --step backtest
```

### Python API

```python
from quant_research.main import run_pipeline

results = run_pipeline(
    config_path='config/default.yaml',
    generate_report=True,
    output_dir='reports/'
)
```

## Testing

Run all tests:

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=quant_research --cov-report=html
```

Run specific test module:

```bash
pytest tests/test_signals.py -v
pytest tests/test_backtest.py -v
```

## Generating Reports

### Jupyter Notebook Report

```bash
jupyter nbconvert --to html notebooks/research_report.ipynb
```

### Programmatic Report

```python
from quant_research.reporting.generator import ReportGenerator

generator = ReportGenerator(results)
generator.generate_html('report.html')
generator.generate_pdf('report.pdf')
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANT RESEARCH PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   DATA LAYER  │          │  SIGNAL LAYER │          │ PORTFOLIO     │
│               │          │               │          │ LAYER         │
│ • Data Loader │          │ • Signal Gen  │          │ • Constructor │
│ • Cleaning    │─────────▶│ • Stat Tests  │─────────▶│ • Risk Ctrl   │
│ • Adjustments │          │ • Significance│          │ • Constraints │
└───────────────┘          └───────────────┘          └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │  BACKTEST ENGINE  │
                          │                   │
                          │ • Vectorized Exec │
                          │ • Cost Modeling   │
                          │ • P&L Tracking    │
                          └───────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │    ANALYTICS      │
                          │                   │
                          │ • Perf Metrics    │
                          │ • Risk Analysis   │
                          │ • Factor Exposure │
                          └───────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │    REPORTING      │
                          │                   │
                          │ • HTML/PDF Report │
                          │ • Charts/Tables   │
                          │ • IS vs OOS       │
                          └───────────────────┘
```

### Data Flow Diagram

```
Raw Data ──▶ Clean Data ──▶ Signal ──▶ Weights ──▶ Returns ──▶ Metrics
   │            │             │           │            │          │
   │            │             │           │            │          │
   ▼            ▼             ▼           ▼            ▼          ▼
┌─────┐      ┌─────┐       ┌─────┐     ┌─────┐      ┌─────┐    ┌─────┐
│ CSV │      │ NaN │       │ Mom │     │ Rank│      │ Net │    │Sharpe│
│ API │      │ Fwd │       │ Rev │     │ Opt │      │ Ret │    │ MDD │
└─────┘      └─────┘       └─────┘     └─────┘      └─────┘    └─────┘
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```
Quant Research Pipeline. (2024). 
GitHub repository: https://github.com/yourusername/quant-research
```
