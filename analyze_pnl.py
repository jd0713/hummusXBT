#!/usr/bin/env python3
import pandas as pd
import os
import locale
from datetime import datetime

def analyze_pnl(csv_file):
    """
    Analyze the provided CSV file to find trades with the largest and smallest Realized PnL.
    
    Args:
        csv_file (str): Path to the CSV file containing trade data
    """
    # Set locale for proper formatting of currency values
    locale.setlocale(locale.LC_ALL, '')
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Make sure PnL_Numeric column exists, if not create it
    if 'PnL_Numeric' not in df.columns:
        # Try to extract numeric values from Realized PnL column
        df['PnL_Numeric'] = df['Realized PnL'].str.replace(' USDT', '').str.replace(',', '').astype(float)
    
    # Find the trade with the largest (most positive) Realized PnL
    max_pnl_idx = df['PnL_Numeric'].idxmax()
    max_pnl_trade = df.loc[max_pnl_idx]
    
    # Find the trade with the smallest (most negative) Realized PnL
    min_pnl_idx = df['PnL_Numeric'].idxmin()
    min_pnl_trade = df.loc[min_pnl_idx]
    
    # Print results
    print("\n===== 거래 분석 결과 (TRADE ANALYSIS RESULTS) =====")
    print("\n🔼 가장 큰 실현 손익(Realized PnL)의 거래:")
    print_trade_details(max_pnl_trade)
    
    print("\n🔽 가장 작은 실현 손익(Realized PnL)의 거래:")
    print_trade_details(min_pnl_trade)

def print_trade_details(trade):
    """
    Print details of a specific trade in a readable format.
    
    Args:
        trade (pandas.Series): Trade data to print
    """
    # Format PnL with appropriate sign
    pnl_value = trade['PnL_Numeric']
    pnl_formatted = f"+{pnl_value:,.2f}" if pnl_value > 0 else f"{pnl_value:,.2f}"
    
    # Calculate duration
    try:
        open_time = datetime.strptime(trade['Open_Time_UTC'], '%Y-%m-%d %H:%M:%S')
        close_time = datetime.strptime(trade['Close_Time_UTC'], '%Y-%m-%d %H:%M:%S')
        duration = close_time - open_time
        duration_str = f"{duration.days} days, {duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes"
    except:
        duration_str = "Unknown"
    
    # Print info
    print(f"심볼 (Symbol): {trade['Symbol']}")
    print(f"방향 (Direction): {trade['Direction']}")
    print(f"진입가 (Entry Price): {trade['Entry Price']}")
    print(f"포지션 크기 (Position Size): {trade['Max Size']} ({trade['Max Size USDT']})")
    print(f"실현 손익 (Realized PnL): {pnl_formatted} USDT ({trade['Realized PnL %']})")
    print(f"오픈 시간 (Open Time KST): {trade['Open_Time_KST_Str']}")
    print(f"종료 시간 (Close Time KST): {trade['Close_Time_KST_Str']}")
    print(f"거래 지속 기간 (Duration): {duration_str}")
    print(f"최대 수익 (Runup): {trade['Runup']}")
    print(f"최대 손실 (Drawdown): {trade['Drawdown']}")

if __name__ == "__main__":
    csv_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "analysis_results/ONLYCANDLE/overall/analyzed_data.csv"
    )
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        exit(1)
    
    analyze_pnl(csv_file)
