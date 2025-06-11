# RSI Trading Strategy with Stooq Data

A comprehensive Python script that implements a **Relative Strength Index (RSI)** based trading strategy using free financial data from Stooq. This tool calculates RSI using Wilder's exponential smoothing method, generates trading signals, simulates portfolio performance, and provides detailed analysis with visualizations.

## 🎯 Features

- **Free Data Source**: Uses Stooq (no API keys required & far fewer rate-limits than Yahoo Finance)
- **Wilder's RSI Calculation**: Implements the authentic RSI formula with exponential smoothing
- **Trading Simulation**: Complete backtesting with entry/exit tracking
- **Performance Metrics**: Total return, CAGR, maximum drawdown, win rate
- **Visualization**: Multi-panel charts showing equity curves, price signals, and RSI indicator
- **Benchmark Comparison**: Optional comparison with SPY or other benchmarks
- **Educational Code**: Extensive comments for learning purposes
- **Flexible Parameters**: Customizable RSI periods, thresholds, and date ranges

## 📋 Requirements

### Python Version
- Python 3.10+ (compatible with 3.11, 3.12)

### Dependencies
```bash
pip install pandas numpy pandas-datareader matplotlib
```

## 🚀 Quick Start

### Basic Usage (Default Parameters)
```bash
python rsi_stooq.py
```

**Default Settings:**
- Ticker: AAPL
- Date Range: 2024-01-01 to 2024-12-31
- RSI Period: 14 days
- Oversold Threshold: 30
- Overbought Threshold: 70
- Starting Capital: $10,000

### Custom Parameters
```bash
python rsi_stooq.py --ticker TSLA --start 2020-01-01 --end 2023-12-31 --period 21 --lower 25 --upper 75 --capital 50000
```

### With Benchmark Comparison
```bash
python rsi_stooq.py --ticker NVDA --bench SPY --capital 25000
```

## 📊 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ticker` | str | "AAPL" | Stock ticker symbol |
| `--start` | str | "2024-01-01" | Start date (YYYY-MM-DD) |
| `--end` | str | "2024-12-31" | End date (YYYY-MM-DD) |
| `--period` | int | 14 | RSI lookback period |
| `--lower` | float | 30.0 | Oversold threshold (buy signal) |
| `--upper` | float | 70.0 | Overbought threshold (sell signal) |
| `--capital` | float | 10000.0 | Starting capital ($) |
| `--bench` | str | None | Benchmark ticker (e.g., SPY) |

## 🎓 Trading Strategy Logic

### RSI Calculation (Wilder's Method)
1. **Price Changes**: Calculate daily price differences
2. **Separate Moves**: Split into upward and downward movements
3. **Exponential Smoothing**: Apply Wilder's smoothing (α = 1/period)
4. **Relative Strength**: RS = Average Up / Average Down
5. **RSI Formula**: RSI = 100 - (100 / (1 + RS))

### Trading Rules
- **Enter Long Position**: When RSI crosses **below** the oversold threshold (default: 30)
- **Exit to Cash**: When RSI crosses **above** the overbought threshold (default: 70)
- **No Shorting**: Strategy only takes long positions or holds cash
- **Full Position**: Uses all available capital for each trade

### Performance Metrics
- **Total Return**: Overall percentage gain/loss
- **CAGR**: Compound Annual Growth Rate (geometric mean)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 📁 Output Files

The script creates an `output/` directory with the following files:

### 1. `daily_equity.csv`
Daily portfolio values for strategy comparison:
```csv
date,equity_rsi,ticker_buy_hold
2020-01-02,10000.0,10000.0
2020-01-03,10000.0,9903.10
...
```

### 2. `trade_log.csv`
Individual trade details:
```csv
entry_date,exit_date,entry_price,exit_price,return_pct
2020-02-27,2020-06-05,66.38,80.67,21.53
```

### 3. `rsi_strategy_analysis.png`
Comprehensive visualization with three panels:
- **Top**: Equity curves comparison (Strategy vs Buy & Hold vs Benchmark)
- **Middle**: Stock price with buy/sell signal markers
- **Bottom**: RSI indicator with threshold lines

### 4. `rsi_calculation.xlsx`
Step-by-step Wilder-RSI calculation sheet exported by the script (Close, ΔPrice, Up/Down moves, Wilder averages, RS and RSI).

## 💡 Usage Examples

### Example 1: Conservative Strategy
```bash
python rsi_stooq.py --ticker SPY --period 21 --lower 25 --upper 75
```
*Uses longer RSI period and wider thresholds for fewer, more confident signals*

### Example 2: Aggressive Strategy
```bash
python rsi_stooq.py --ticker QQQ --period 10 --lower 40 --upper 60
```
*Uses shorter period and tighter thresholds for more frequent trading*

### Example 3: Volatile Stock Analysis
```bash
python rsi_stooq.py --ticker TSLA --start 2020-01-01 --end 2022-12-31 --bench SPY --capital 100000
```
*Analyzes Tesla during high volatility period with SPY benchmark*

### Example 4: Crypto-Adjacent Stock
```bash
python rsi_stooq.py --ticker MSTR --period 14 --lower 30 --upper 70 --bench BTC-USD
```
*Note: Bitcoin benchmark may not work with Stooq format*

## 📈 Sample Results

### AAPL 2020 (COVID Period)
```
📈 PERFORMANCE SUMMARY
Total Return: 21.53%
CAGR: 21.61%
Maximum Drawdown: -25.89%
Win Rate: 100.0% (1 total trades)

📊 Buy & Hold AAPL Return: 78.25%
```

### TSLA 2020-2021 (High Volatility)
```
📈 PERFORMANCE SUMMARY
Total Return: 192.08%
CAGR: 71.09%
Maximum Drawdown: -26.09%
Win Rate: 100.0% (2 total trades)

📊 Buy & Hold TSLA Return: 1128.07%
📊 Buy & Hold SPY Return: 50.92%
```

## 🔧 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'pandas'"**
   ```bash
   pip install pandas numpy pandas-datareader matplotlib
   ```

2. **"Error downloading [TICKER]"**
   - Check ticker symbol spelling
   - Ensure date range is valid (not weekends/holidays)
   - Try a different ticker

3. **"No trades executed"**
   - RSI may not have hit thresholds in the date range
   - Try more sensitive thresholds (e.g., `--lower 40 --upper 60`)
   - Use a more volatile stock or longer time period

4. **Python Version Issues**
   ```bash
   python --version  # Should be 3.10+
   python3 rsi_stooq.py  # Try python3 instead of python
   ```

### Data Limitations
- **Stooq Coverage**: Primarily US stocks, some international markets
- **Symbol Format**: Stooq uses lowercase + ".us" (script handles conversion)
- **Date Range**: Limited to available trading days
- **Real-time Data**: Stooq provides delayed data

## 🎯 Educational Value

This script is designed for learning and demonstrates:

### Technical Analysis Concepts
- RSI calculation and interpretation
- Overbought/oversold conditions
- Mean reversion strategies
- Backtesting methodology

### Programming Concepts
- Financial data handling with pandas
- Time series analysis
- Object-oriented design patterns
- Data visualization with matplotlib
- Command-line argument parsing

### Risk Management
- Maximum drawdown analysis
- Win/loss ratio calculation
- Portfolio simulation
- Benchmark comparison

## ⚠️ Disclaimer

**This script is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading involves risk of financial loss
- Always do your own research before making investment decisions
- Consider transaction costs and taxes in real trading
- RSI strategies may underperform in trending markets

## 🤝 Contributing

Feel free to enhance the script by:
- Adding more technical indicators
- Implementing stop-loss functionality
- Supporting additional data sources
- Improving visualization features
- Adding more sophisticated position sizing

## 📚 Further Reading

- [Wilder's RSI Original Paper](https://en.wikipedia.org/wiki/Relative_strength_index)
- [Technical Analysis Fundamentals](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Pandas Financial Data Analysis](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Stooq Data Documentation](https://stooq.com/)

---

*Last Updated: June 2024*
*Compatible with Python 3.10+ | Requires: pandas, numpy, pandas-datareader, matplotlib* 