"""
Download 60-day 5-minute K-line data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

def download_stock_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        days: Number of days to download (default: 60)
    
    Returns:
        DataFrame with stock data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Downloading {days} days of 5-minute data for {ticker}...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Download data from Yahoo Finance
        # Note: yfinance limits intraday data to last 60 days
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start_date,
            end=end_date,
            interval='5m',
            actions=False  # Exclude dividends and stock splits
        )
        
        if df.empty:
            raise ValueError("No data retrieved. Please check the ticker symbol.")
        
        # Reset index to get datetime as a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Datetime': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Select only required columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Remove timezone info if present
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        # Add Adj Close column (same as close for intraday data)
        df['Adj Close'] = df['close']
        
        # Reorder columns to match TSLA_60d_5m.csv format
        df = df[['datetime', 'Adj Close', 'close', 'high', 'low', 'open', 'volume']]
        
        print(f"Successfully downloaded {len(df)} data points")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None


def main():
    """Main function"""
    # Check if ticker is provided
    if len(sys.argv) < 2:
        print("Error: No ticker specified!")
        print("\nUsage: python download_data.py <TICKER>")
        print("\nExamples:")
        print("  python download_data.py AAPL")
        print("  python download_data.py TSLA")
        print("  python download_data.py MSFT")
        print("\nNote: Yahoo Finance limits 5-minute data to the last 60 days")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    # Download data
    df = download_stock_data(ticker)
    
    if df is not None:
        # Save to CSV file
        filename = f"{ticker}_60d_5m.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to: {filename}")
        print(f"Total rows: {len(df)}")
        print(f"\nYou can now run: python r_breaker_strategy.py {filename}")
    else:
        print("\nFailed to download data. Please check the ticker symbol and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
