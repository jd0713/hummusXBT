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
        diff_minutes = (close_dt - open_dt).total_seconds() / 60
        return diff_minutes
    except:
        return None

# Extract numeric values if not already available
if 'Size_USDT_Numeric' not in df.columns:
    df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(extract_usdt_value)

if 'PnL_Numeric' not in df.columns:
    df['PnL_Numeric'] = df['Realized PnL'].apply(extract_pnl_value)

# Calculate time difference in minutes
df['Time_Diff_Minutes'] = df.apply(
    lambda row: calculate_time_diff(row['Open_Time_UTC'], row['Close_Time_UTC']), 
    axis=1
)

# Filter for quick trades (10 minutes or less)
quick_trades_df = df[df['Time_Diff_Minutes'] <= 10].copy()

# Calculate total PnL for quick trades
total_quick_pnl = quick_trades_df['PnL_Numeric'].sum()
total_quick_trades = len(quick_trades_df)
total_positions = len(df)

# Sort by Time_Diff_Minutes (ascending) and Size_USDT_Numeric (descending)
quick_trades_df = quick_trades_df.sort_values(by=['Time_Diff_Minutes', 'Size_USDT_Numeric'], 
                                             ascending=[True, False])

# Convert time difference to a string with proper formatting
quick_trades_df['Duration'] = quick_trades_df['Time_Diff_Minutes'].apply(
    lambda x: f"{int(x)} min {int((x % 1) * 60)} sec"
)

# Format PnL with colors and signs for console output
def format_pnl_for_display(pnl):
    if pnl > 0:
        return f"+{pnl:.2f}"
    else:
        return f"{pnl:.2f}"

quick_trades_df['PnL_Display'] = quick_trades_df['PnL_Numeric'].apply(format_pnl_for_display)

# Create a prettier display DataFrame with selected columns
display_df = quick_trades_df[[
    'Symbol', 'Direction', 'Max Size USDT', 'Size_USDT_Numeric',
    'Realized PnL', 'PnL_Numeric', 'Realized PnL %', 
    'Open_Time_UTC', 'Close_Time_UTC', 'Duration'
]]

# Rename columns for better display
display_df = display_df.rename(columns={
    'Size_USDT_Numeric': 'Size (USDT)',
    'PnL_Numeric': 'PnL (USDT)',
    'Open_Time_UTC': 'Open Time',
    'Close_Time_UTC': 'Close Time'
})

# Print summary
print(f"\n{'=' * 80}")
print(f"빠른 거래 분석 (10분 이내)")
print(f"{'=' * 80}")
print(f"총 빠른 거래 수: {total_quick_trades} (전체 {total_positions}개 중 {total_quick_trades/total_positions:.2%})")
print(f"빠른 거래 PnL 합계: {total_quick_pnl:.2f} USDT")
print(f"{'=' * 80}\n")

# Use tabulate to display the DataFrame in a nice format
table = tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False, 
                 numalign='right', stralign='left')
print(table)

# Additional analysis - profit vs loss distribution
profitable_trades = quick_trades_df[quick_trades_df['PnL_Numeric'] > 0]
losing_trades = quick_trades_df[quick_trades_df['PnL_Numeric'] < 0]
breakeven_trades = quick_trades_df[quick_trades_df['PnL_Numeric'] == 0]

profit_sum = profitable_trades['PnL_Numeric'].sum()
loss_sum = losing_trades['PnL_Numeric'].sum()

print(f"\n{'=' * 80}")
print(f"빠른 거래 수익/손실 분석")
print(f"{'=' * 80}")
print(f"수익 거래: {len(profitable_trades)}개 (총 {profit_sum:.2f} USDT)")
print(f"손실 거래: {len(losing_trades)}개 (총 {loss_sum:.2f} USDT)")
print(f"손익분기 거래: {len(breakeven_trades)}개")
print(f"승률: {len(profitable_trades) / total_quick_trades:.2%}")
print(f"{'=' * 80}")

# Optional: Save to CSV
# quick_trades_df.to_csv('analysis_results/u_guys_r_rly_dum/overall/quick_trades_analysis.csv', index=False)
