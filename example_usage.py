#!/usr/bin/env python3
"""
Example Usage Scripts for RSI Trading Strategy

This file contains example code snippets showing different ways to use
the RSI trading strategy script. You can run these examples directly
or use them as a reference for your own analysis.
"""

import subprocess
import sys

def run_example(description, command):
    """Run an example command with description."""
    print(f"\n{'='*60}")
    print(f"📊 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("Running...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(line)
        else:
            print("❌ Error:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Failed to run: {e}")

def main():
    """Run example analyses."""
    
    print("🚀 RSI Trading Strategy - Example Usage")
    print("This script demonstrates various ways to use the RSI strategy.")
    
    examples = [
        (
            "Basic AAPL Analysis (Default Parameters)", 
            "python rsi_stooq.py"
        ),
        (
            "Conservative SPY Strategy", 
            "python rsi_stooq.py --ticker SPY --period 21 --lower 25 --upper 75"
        ),
        (
            "Aggressive QQQ Strategy", 
            "python rsi_stooq.py --ticker QQQ --period 10 --lower 40 --upper 60"
        ),
        (
            "Tesla vs SPY Benchmark (2020-2021)", 
            "python rsi_stooq.py --ticker TSLA --start 2020-01-01 --end 2021-12-31 --bench SPY --capital 50000"
        ),
        (
            "Microsoft with Custom Thresholds", 
            "python rsi_stooq.py --ticker MSFT --lower 35 --upper 65 --capital 25000"
        )
    ]
    
    # Ask user which examples to run
    print(f"\nAvailable Examples:")
    for i, (desc, _) in enumerate(examples, 1):
        print(f"{i}. {desc}")
    
    print("\nOptions:")
    print("- Enter numbers (e.g., '1,3,5') to run specific examples")
    print("- Enter 'all' to run all examples")
    print("- Enter 'quit' to exit")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'quit':
        print("Goodbye! 👋")
        return
    elif choice == 'all':
        selected = list(range(len(examples)))
    else:
        try:
            selected = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [i for i in selected if 0 <= i < len(examples)]
        except ValueError:
            print("❌ Invalid input. Please use numbers separated by commas.")
            return
    
    if not selected:
        print("❌ No valid examples selected.")
        return
    
    # Run selected examples
    for i in selected:
        desc, cmd = examples[i]
        run_example(desc, cmd)
    
    print(f"\n{'='*60}")
    print("🎉 Example analysis complete!")
    print("📁 Check the 'output/' directory for detailed results.")
    print("📊 View the generated visualization: output/rsi_strategy_analysis.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 