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
        
        # Convert to human readable format
        days = int(diff_seconds // (24 * 3600))
        remaining_seconds = diff_seconds % (24 * 3600)
        hours = int(remaining_seconds // 3600)
        remaining_seconds %= 3600
        minutes = int(remaining_seconds // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except:
        return "Unknown"

# Extract numeric values if not already available
if 'Size_USDT_Numeric' not in df.columns:
    df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(extract_usdt_value)

if 'PnL_Numeric' not in df.columns:
    df['PnL_Numeric'] = df['Realized PnL'].apply(extract_pnl_value)

# Calculate PnL percentage if not already available (using PnL_Numeric and Size_USDT_Numeric for accuracy)
if 'PnL_Percentage' not in df.columns:
    df['PnL_Percentage'] = (df['PnL_Numeric'] / df['Size_USDT_Numeric'] * 100)

# Calculate duration in a human-readable format
df['Duration'] = df.apply(
    lambda row: calculate_time_diff(row['Open_Time_UTC'], row['Close_Time_UTC']), 
    axis=1
)

# Sort by Size_USDT_Numeric in descending order and take top 100
largest_trades_df = df.sort_values(by='Size_USDT_Numeric', ascending=False).head(100)

# Format PnL with colors and signs for console output
def format_pnl_for_display(pnl):
    if pnl > 0:
        return f"+{pnl:.2f}"
    else:
        return f"{pnl:.2f}"

largest_trades_df['PnL_Display'] = largest_trades_df['PnL_Numeric'].apply(format_pnl_for_display)

# Create a prettier display DataFrame with selected columns
display_df = largest_trades_df[[
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

# Calculate some statistics for the largest 100 trades
total_size = largest_trades_df['Size_USDT_Numeric'].sum()
total_pnl = largest_trades_df['PnL_Numeric'].sum()
avg_pnl_pct = total_pnl / total_size * 100
profitable_trades = largest_trades_df[largest_trades_df['PnL_Numeric'] > 0]
losing_trades = largest_trades_df[largest_trades_df['PnL_Numeric'] < 0]
profit_sum = profitable_trades['PnL_Numeric'].sum()
loss_sum = losing_trades['PnL_Numeric'].sum()

# Print summary
print(f"\n{'=' * 100}")
print(f"Max Size USDT 상위 100개 거래 분석")
print(f"{'=' * 100}")
print(f"Top 100 거래 총 규모: {total_size:,.2f} USDT")
print(f"Top 100 거래 PnL 합계: {total_pnl:,.2f} USDT (평균 수익률: {avg_pnl_pct:.2f}%)")
print(f"수익 거래: {len(profitable_trades)}개 (총 {profit_sum:,.2f} USDT)")
print(f"손실 거래: {len(losing_trades)}개 (총 {loss_sum:,.2f} USDT)")
print(f"승률: {len(profitable_trades) / 100:.2%}")
print(f"{'=' * 100}\n")

# Use tabulate to display the DataFrame in a nice format
table = tabulate(display_df, headers='keys', tablefmt='pretty', showindex=True, 
                 numalign='right', stralign='left')
print(table)

# Optional: Save to CSV
# largest_trades_df.to_csv('analysis_results/u_guys_r_rly_dum/overall/largest_trades_analysis.csv', index=False)
