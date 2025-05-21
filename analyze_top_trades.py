import pandas as pd
import re
import datetime as dt
from tabulate import tabulate

# Load the CSV file
file_path = 'analysis_results/u_guys_r_rly_dum/overall/analyzed_data.csv'
df = pd.read_csv(file_path)

# Function to extract numeric value from formatted USDT string
def extract_usdt_value(usdt_str):
    if pd.isna(usdt_str):
        return None
    match = re.search(r'([\d,\.]+)\s*USDT', str(usdt_str))
    if match:
        return float(match.group(1).replace(',', ''))
    return None

# Function to extract numeric value from Realized PnL string
def extract_pnl_value(pnl_str):
    if pd.isna(pnl_str):
        return None
    match = re.search(r'([\-\d,\.]+)\s*USDT', str(pnl_str))
    if match:
        return float(match.group(1).replace(',', ''))
    return None

# Function to calculate time difference in minutes
def calculate_time_diff(open_time, close_time):
    try:
        open_dt = pd.to_datetime(open_time)
        close_dt = pd.to_datetime(close_time)
        diff_seconds = (close_dt - open_dt).total_seconds()
        return diff_seconds
    except:
        return None

# Function to format duration in a human-readable format
def format_duration(seconds):
    if seconds is None:
        return "Unknown"
    
    days = int(seconds // (24 * 3600))
    remaining_seconds = seconds % (24 * 3600)
    hours = int(remaining_seconds // 3600)
    remaining_seconds %= 3600
    minutes = int(remaining_seconds // 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

# Extract numeric values if not already available
if 'Size_USDT_Numeric' not in df.columns:
    df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(extract_usdt_value)

if 'PnL_Numeric' not in df.columns:
    df['PnL_Numeric'] = df['Realized PnL'].apply(extract_pnl_value)

# Calculate time difference in seconds
df['Duration_Seconds'] = df.apply(
    lambda row: calculate_time_diff(row['Open_Time_UTC'], row['Close_Time_UTC']), 
    axis=1
)

# Format duration in a readable way
df['Duration'] = df['Duration_Seconds'].apply(format_duration)

# Create display function for tables
def display_trades(trades_df, title, criterion):
    # Create a prettier display DataFrame with selected columns
    display_df = trades_df[[
        'Symbol', 'Direction', 'Max Size USDT', 'Size_USDT_Numeric',
        'Realized PnL', 'PnL_Numeric', 'Realized PnL %', 
        'Open_Time_UTC', 'Close_Time_UTC', 'Duration'
    ]]

    # Rename columns for better display
    display_df = display_df.rename(columns={
        'Size_USDT_Numeric': 'Size (USDT)',
        'PnL_Numeric': 'PnL (USDT)',
        'Open_Time_UTC': 'Open Time',
        'Close_Time_UTC': 'Close Time',
        'Realized PnL %': 'PnL %'
    })

    # Calculate some statistics
    total_size = trades_df['Size_USDT_Numeric'].sum()
    total_pnl = trades_df['PnL_Numeric'].sum()
    avg_pnl_pct = total_pnl / total_size * 100 if total_size > 0 else 0
    
    if criterion == 'pnl_top':
        profitable_trades = len(trades_df[trades_df['PnL_Numeric'] > 0])
        print(f"\n{'=' * 100}")
        print(f"PnL 상위 100개 거래 분석")
        print(f"{'=' * 100}")
        print(f"Top 100 거래 총 규모: {total_size:,.2f} USDT")
        print(f"Top 100 거래 PnL 합계: {total_pnl:,.2f} USDT (평균 수익률: {avg_pnl_pct:.2f}%)")
        print(f"수익 거래 비율: {profitable_trades}/100 ({profitable_trades}%)")
    elif criterion == 'pnl_bottom':
        losing_trades = len(trades_df[trades_df['PnL_Numeric'] < 0])
        print(f"\n{'=' * 100}")
        print(f"PnL 하위 100개 거래 분석")
        print(f"{'=' * 100}")
        print(f"Bottom 100 거래 총 규모: {total_size:,.2f} USDT")
        print(f"Bottom 100 거래 PnL 합계: {total_pnl:,.2f} USDT (평균 수익률: {avg_pnl_pct:.2f}%)")
        print(f"손실 거래 비율: {losing_trades}/100 ({losing_trades}%)")
    elif criterion == 'duration':
        avg_duration = trades_df['Duration_Seconds'].mean()
        print(f"\n{'=' * 100}")
        print(f"Duration 상위 100개 거래 분석 (가장 오래 지속된 거래)")
        print(f"{'=' * 100}")
        print(f"Top 100 거래 총 규모: {total_size:,.2f} USDT")
        print(f"Top 100 거래 PnL 합계: {total_pnl:,.2f} USDT (평균 수익률: {avg_pnl_pct:.2f}%)")
        print(f"평균 지속 시간: {format_duration(avg_duration)}")
    
    print(f"{'=' * 100}\n")

    # Use tabulate to display the DataFrame in a nice format
    table = tabulate(display_df, headers='keys', tablefmt='pretty', showindex=True, 
                    numalign='right', stralign='left')
    print(table)

# 1. PnL 상위 100개 거래
top_pnl_df = df.sort_values(by='PnL_Numeric', ascending=False).head(100)
display_trades(top_pnl_df, "PnL 상위 100개 거래", "pnl_top")

# 2. PnL 하위 100개 거래
bottom_pnl_df = df.sort_values(by='PnL_Numeric', ascending=True).head(100)
display_trades(bottom_pnl_df, "PnL 하위 100개 거래", "pnl_bottom")

# 3. Duration 상위 100개 거래
top_duration_df = df.sort_values(by='Duration_Seconds', ascending=False).head(100)
display_trades(top_duration_df, "Duration 상위 100개 거래", "duration")
