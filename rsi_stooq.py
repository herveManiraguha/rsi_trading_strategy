#!/usr/bin/env python3
"""
RSI Trading Strategy with Stooq Data

This script implements a Relative Strength Index (RSI) based trading strategy
using data from Stooq. It calculates RSI using Wilder's method, generates
trading signals, simulates equity curves, and provides performance analysis.

Compatible with Python 3.10+
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os


def parse_arguments():
    """Parse command line arguments for the RSI trading strategy."""
    parser = argparse.ArgumentParser(
        description="RSI Trading Strategy using Stooq data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--ticker", default="AAPL", 
                       help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2024-01-01", 
                       help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end", default="2024-12-31", 
                       help="End date YYYY-MM-DD (default: 2024-12-31)")
    parser.add_argument("--period", type=int, default=14, 
                       help="RSI lookback period (default: 14)")
    parser.add_argument("--lower", type=float, default=30.0, 
                       help="Oversold threshold (default: 30)")
    parser.add_argument("--upper", type=float, default=70.0, 
                       help="Overbought threshold (default: 70)")
    parser.add_argument("--capital", type=float, default=10000.0, 
                       help="Starting capital (default: 10000)")
    parser.add_argument("--bench", 
                       help="Benchmark ticker (e.g., SPY) for comparison")
    
    return parser.parse_args()


def download_stooq_data(ticker, start_date, end_date):
    """
    Download daily adjusted close prices from Stooq.
    Stooq uses lowercase ticker + '.us' format for US stocks.
    """
    # Convert ticker to Stooq format (lowercase + .us)
    stooq_symbol = f"{ticker.lower()}.us"
    
    print(f"Downloading {ticker} data from Stooq ({start_date} to {end_date})...")
    
    try:
        # Download data using pandas_datareader
        data = web.DataReader(stooq_symbol, "stooq", start_date, end_date)
        
        # Sort ascending (Stooq returns newest-first)
        data = data.sort_index()
        
        # Extract adjusted close prices
        if 'Close' in data.columns:
            prices = data['Close'].dropna()
            print(f"✅ Successfully downloaded {len(prices)} trading days")
            return prices
        else:
            raise ValueError(f"No 'Close' column found for {ticker}")
            
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None


def calculate_rsi_wilder(prices, period):
    """
    Calculate RSI using Wilder's exponential smoothing method **and** build a
    step-by-step calculation table that can be exported to Excel.

    Returns
    -------
    rsi : pd.Series
        Series with final RSI values.
    calc_df : pd.DataFrame
        DataFrame containing Close, ΔPrice, UpMove, DownMove, Wilder averages,
        RS and RSI for each date – useful for audit or teaching purposes.
    """
    # Calculate price changes (today's close - yesterday's close)
    changes = prices.diff()
    
    # Separate upward and downward movements
    up_moves = np.maximum(changes, 0)      # Positive changes only
    down_moves = np.maximum(-changes, 0)   # Absolute value of negative changes
    
    # Initialize arrays for exponential moving averages
    avg_up = pd.Series(index=prices.index, dtype=float)
    avg_down = pd.Series(index=prices.index, dtype=float)
    
    # Calculate initial simple moving averages for the first period
    first_avg_up = up_moves.iloc[1:period+1].mean()
    first_avg_down = down_moves.iloc[1:period+1].mean()
    
    # Set the first calculated values
    avg_up.iloc[period] = first_avg_up
    avg_down.iloc[period] = first_avg_down
    
    # Apply Wilder's exponential smoothing (alpha = 1/period)
    alpha = 1.0 / period
    
    # Calculate exponential moving averages for the rest of the data
    for i in range(period + 1, len(prices)):
        avg_up.iloc[i] = alpha * up_moves.iloc[i] + (1 - alpha) * avg_up.iloc[i-1]
        avg_down.iloc[i] = alpha * down_moves.iloc[i] + (1 - alpha) * avg_down.iloc[i-1]
    
    # Calculate Relative Strength (RS) and RSI
    # Avoid division by zero
    rs = avg_up / (avg_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # NEW ➜ Build DataFrame that mirrors every intermediate calculation
    calc_df = pd.DataFrame({
        'Close': prices,
        'Change': changes,
        'UpMove': up_moves,
        'DownMove': down_moves,
        'AvgUp': avg_up,
        'AvgDown': avg_down,
        'RS': rs,
        'RSI': rsi
    })

    # Return both final RSI series and the detailed table
    return rsi, calc_df


def generate_trading_signals(rsi, lower_threshold, upper_threshold):
    """
    Generate trading signals based on RSI thresholds.
    
    Rules:
    - Enter long (position = 1) when RSI crosses below lower threshold (oversold)
    - Exit to cash (position = 0) when RSI crosses above upper threshold (overbought)
    - No shorting allowed
    """
    signals = pd.Series(0, index=rsi.index)  # Start with no position
    position = 0  # Track current position (0 = cash, 1 = long)
    
    for i in range(1, len(rsi)):
        prev_rsi = rsi.iloc[i-1]
        curr_rsi = rsi.iloc[i]
        
        # Entry rule ▶ when RSI crosses BELOW oversold threshold (mean-reversion) ──┐
        if position == 0 and prev_rsi >= lower_threshold and curr_rsi < lower_threshold:
            position = 1  # long                                                            │
        # Exit rule  ▼ when RSI crosses ABOVE overbought threshold─────────────────────────┘
        elif position == 1 and prev_rsi <= upper_threshold and curr_rsi > upper_threshold:
            position = 0  # flat / cash
        
        signals.iloc[i] = position
    
    return signals


def simulate_trading(prices, signals, initial_capital):
    """
    Simulate the trading strategy and calculate equity curve.
    
    Track position value, daily P/L, and cumulative equity.
    Record individual trades with entry/exit details.
    """
    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    equity_curve = []
    trades = []
    
    # Track trade details
    entry_date = None
    entry_price = None
    
    for date, price in prices.items():
        current_signal = signals.loc[date]
        previous_signal = signals.shift(1).loc[date] if date != signals.index[0] else 0
        
        # Check for position changes
        if current_signal == 1 and previous_signal == 0:
            # Enter long position - buy with all available cash
            shares = cash / price
            cash = 0
            entry_date = date
            entry_price = price
            
        elif current_signal == 0 and previous_signal == 1:
            # Exit position - sell all shares
            cash = shares * price
            
            # Record the completed trade
            if entry_date is not None and entry_price is not None:
                trade_return = (price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'return_pct': trade_return * 100
                })
            
            shares = 0
            entry_date = None
            entry_price = None
        
        # Calculate current portfolio value
        portfolio_value = cash + (shares * price)
        equity_curve.append(portfolio_value)
    
    # Convert to pandas Series
    equity_series = pd.Series(equity_curve, index=prices.index)
    trades_df = pd.DataFrame(trades)
    
    return equity_series, trades_df


def calculate_performance_metrics(equity_curve, trades_df, initial_capital):
    """
    Calculate key performance metrics for the trading strategy.
    
    Metrics include: total return, CAGR, maximum drawdown, win rate.
    """
    # Total return percentage
    final_value = equity_curve.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Calculate CAGR (Compound Annual Growth Rate)
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    cagr_pct = cagr * 100
    
    # Calculate maximum drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    if len(trades_df) > 0:
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades * 100
    else:
        win_rate = 0
        total_trades = 0
    
    return {
        'total_return': total_return,
        'cagr': cagr_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades
    }


def calculate_buy_hold_return(prices, initial_capital):
    """Calculate buy-and-hold equity curve for comparison."""
    # Buy shares with initial capital at first price
    shares = initial_capital / prices.iloc[0]
    
    # Calculate portfolio value over time
    buy_hold_equity = shares * prices
    
    return buy_hold_equity


def save_results(equity_curve, buy_hold_equity, trades_df, ticker):
    """Save trading results to CSV files in ./output/ directory."""
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Save daily equity curve comparison
    daily_data = pd.DataFrame({
        'date': equity_curve.index,
        'equity_rsi': equity_curve.values,
        f'{ticker.lower()}_buy_hold': buy_hold_equity.values
    })
    daily_data.to_csv('./output/daily_equity.csv', index=False)
    print("💾 Saved daily equity data to ./output/daily_equity.csv")
    
    # Save trade log
    if len(trades_df) > 0:
        trades_df.to_csv('./output/trade_log.csv', index=False)
        print("💾 Saved trade log to ./output/trade_log.csv")
    else:
        print("📝 No trades executed - no trade log saved")


def create_visualization(equity_curve, buy_hold_equity, rsi, signals, prices, 
                        ticker, metrics, benchmark_equity=None, benchmark_name=None):
    """Create comprehensive visualization of the trading strategy results."""
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Equity curves comparison
    ax1.plot(equity_curve.index, equity_curve.values, label='RSI Strategy', 
             linewidth=2, color='blue')
    ax1.plot(buy_hold_equity.index, buy_hold_equity.values, 
             label=f'{ticker} Buy & Hold', linewidth=2, color='orange')
    
    # Add benchmark if provided
    if benchmark_equity is not None:
        ax1.plot(benchmark_equity.index, benchmark_equity.values, 
                 label=f'{benchmark_name} Buy & Hold', linewidth=2, color='green')
    
    ax1.set_title(f'RSI Trading Strategy vs Buy & Hold - {ticker}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add performance metrics as text
    metrics_text = (f"Total Return: {metrics['total_return']:.1f}%\n"
                   f"CAGR: {metrics['cagr']:.1f}%\n"
                   f"Max Drawdown: {metrics['max_drawdown']:.1f}%\n"
                   f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['total_trades']} trades)")
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Stock price with buy/sell signals
    ax2.plot(prices.index, prices.values, label=f'{ticker} Price', color='black', linewidth=1)
    
    # Mark buy and sell signals
    buy_signals = signals[signals.diff() == 1]
    sell_signals = signals[signals.diff() == -1]
    
    for date in buy_signals.index:
        ax2.scatter(date, prices.loc[date], color='green', marker='^', s=100, zorder=5)
    
    for date in sell_signals.index:
        ax2.scatter(date, prices.loc[date], color='red', marker='v', s=100, zorder=5)
    
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Price Chart with Buy/Sell Signals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI with thresholds
    ax3.plot(rsi.index, rsi.values, label='RSI', color='purple', linewidth=1.5)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3.fill_between(rsi.index, 70, 100, alpha=0.1, color='red')
    ax3.fill_between(rsi.index, 0, 30, alpha=0.1, color='green')
    
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')
    ax3.set_title('RSI Indicator with Trading Thresholds')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('./output/rsi_strategy_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Saved strategy visualization to ./output/rsi_strategy_analysis.png")
    plt.show()


def main():
    """Main function to execute the RSI trading strategy analysis."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 RSI Trading Strategy Analysis")
    print("=" * 50)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.start} to {args.end}")
    print(f"RSI Period: {args.period}")
    print(f"Thresholds: {args.lower} (oversold) / {args.upper} (overbought)")
    print(f"Starting Capital: ${args.capital:,.0f}")
    print("=" * 50)
    
    # Download stock price data from Stooq
    prices = download_stooq_data(args.ticker, args.start, args.end)
    if prices is None:
        print("❌ Failed to download price data. Exiting.")
        return
    
    # Calculate RSI using Wilder's method
    print(f"\n📊 Calculating RSI with {args.period}-period lookback...")
    rsi, rsi_calc_df = calculate_rsi_wilder(prices, args.period)
    
    # Generate trading signals based on RSI thresholds
    print("🎯 Generating trading signals...")
    signals = generate_trading_signals(rsi, args.lower, args.upper)
    
    # Simulate the trading strategy
    print("💰 Simulating trading strategy...")
    equity_curve, trades_df = simulate_trading(prices, signals, args.capital)
    
    # Calculate buy-and-hold comparison
    buy_hold_equity = calculate_buy_hold_return(prices, args.capital)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_curve, trades_df, args.capital)
    
    # Download benchmark data if specified
    benchmark_equity = None
    benchmark_name = None
    if args.bench:
        print(f"📈 Downloading benchmark data for {args.bench}...")
        benchmark_prices = download_stooq_data(args.bench, args.start, args.end)
        if benchmark_prices is not None:
            benchmark_equity = calculate_buy_hold_return(benchmark_prices, args.capital)
            benchmark_name = args.bench
        else:
            print(f"⚠️  Could not download benchmark data for {args.bench}")
    
    # Print performance summary
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"CAGR: {metrics['cagr']:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['total_trades']} total trades)")
    
    # Compare with buy-and-hold
    bh_return = (buy_hold_equity.iloc[-1] - args.capital) / args.capital * 100
    print(f"\n📊 Buy & Hold {args.ticker} Return: {bh_return:.2f}%")
    
    if benchmark_equity is not None:
        bench_return = (benchmark_equity.iloc[-1] - args.capital) / args.capital * 100
        print(f"📊 Buy & Hold {benchmark_name} Return: {bench_return:.2f}%")
    
    print("=" * 50)
    
    # Save results to CSV files
    save_results(equity_curve, buy_hold_equity, trades_df, args.ticker)
    
    # NEW ➜ Export the detailed RSI worksheet to Excel for transparency / learning
    os.makedirs('./output', exist_ok=True)
    rsi_calc_df.to_excel('./output/rsi_calculation.xlsx')
    print("📚 Saved step-by-step RSI calculations to ./output/rsi_calculation.xlsx")
    
    # Create and display visualization
    create_visualization(equity_curve, buy_hold_equity, rsi, signals, prices, 
                        args.ticker, metrics, benchmark_equity, benchmark_name)
    
    print("\n✅ Analysis complete! Check ./output/ directory for detailed results.")


if __name__ == "__main__":
    main() 