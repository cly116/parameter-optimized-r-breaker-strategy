# R Breaker Strategy

## Introduction

The R Breaker strategy is a sophisticated intraday trading system that identifies potential breakout and reversal points based on the previous day's price action. This implementation features automatic parameter optimization to find the best settings for any given stock.

### Strategy Overview

The strategy calculates six key price levels each day based on the previous day's data:

1. **Observation Prices**
   - Ssetup (Sell Setup) = High + a × (Close - Low)
   - Bsetup (Buy Setup) = Low - a × (High - Close)

2. **Reversal Prices**
   - Senter (Sell Enter) = b/2 × (High + Low) - c × Low
   - Benter (Buy Enter) = b/2 × (High + Low) - c × High

3. **Breakout Prices**
   - Sbreak (Sell Break) = Ssetup - d × (Ssetup - Bsetup)
   - Bbreak (Buy Break) = Bsetup + d × (Ssetup - Bsetup)

### Trading Rules

1. **Trend Following Signals**
   - **Long Signal**: If price breaks above Bbreak with no existing position → Open long
   - **Short Signal**: If price breaks below Sbreak with no existing position → Open short

2. **Reversal Trading Signals**
   - **Long to Short**: When holding long, if daily high exceeds Ssetup then price drops below Senter → Reverse to short
   - **Short to Long**: When holding short, if daily low drops below Bsetup then price rises above Benter → Reverse to long

3. **Risk Management**
   - **Intraday Close**: All positions are closed 5 minutes before market close to avoid overnight risk

### Parameter Optimization

The strategy automatically finds optimal parameters (a, b, c, d) through exhaustive backtesting:

- Tests 1,440 different parameter combinations
- Evaluates each combination using 60 days of 5-minute historical data
- Selects parameters that maximize Sharpe ratio while considering total return, maximum drawdown, and win rate
- Displays real-time progress during optimization

The optimization ranges are:
- a: 0.3-0.6 (controls observation price distance)
- b: 0.4-0.8 (controls reversal price midpoint)
- c: 0.05-0.3 (controls reversal price offset)
- d: 0.2-0.5 (controls breakout price distance)

## Download Stock Data

First, download 60-day 5-minute K-line data from Yahoo Finance:

```bash
python download_data.py <TICKER>
```

### Examples
```bash
python download_data.py AAPL    # Downloads AAPL_60d_5m.csv
python download_data.py TSLA    # Downloads TSLA_60d_5m.csv
python download_data.py MSFT    # Downloads MSFT_60d_5m.csv
```

## Run Strategy

After downloading data, run the R Breaker strategy:

```bash
python r_breaker_strategy.py <data_file.csv>
```

### Examples
```bash
python r_breaker_strategy.py TSLA_60d_5m.csv
python r_breaker_strategy.py AAPL_60d_5m.csv
python r_breaker_strategy.py MSFT_60d_5m.csv
```

You can use any CSV file with the required format (datetime, open, high, low, close, volume columns).

This will:
1. Optimize parameters with progress bar
2. Generate a single PDF report containing strategy analysis and trade records

## Output

- `r_breaker_report.pdf` - Complete report with performance metrics and trade history
