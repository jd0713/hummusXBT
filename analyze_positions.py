import pandas as pd
import datetime as dt
import re

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

# Function to extract numeric value from Realized PnL column
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

# Extract numeric values from 'Max Size USDT' column
df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(extract_usdt_value)

# Extract numeric values from 'Realized PnL' column if not already available
if 'PnL_Numeric' not in df.columns:
    df['PnL_Numeric'] = df['Realized PnL'].apply(extract_pnl_value)

# Calculate time difference in minutes
df['Time_Diff_Minutes'] = df.apply(
    lambda row: calculate_time_diff(row['Open_Time_UTC'], row['Close_Time_UTC']), 
    axis=1
)

# Create filters for each condition
small_size_filter = df['Size_USDT_Numeric'] <= 1000
short_time_filter = df['Time_Diff_Minutes'] <= 10
both_criteria_filter = small_size_filter & short_time_filter

# Union of the filters (positions that satisfy either condition 1, condition 2, or both)
union_filter = small_size_filter | short_time_filter

# Count positions meeting criteria
total_positions = len(df)
small_size_positions = len(df[small_size_filter])
short_time_positions = len(df[short_time_filter])
both_criteria_positions = len(df[both_criteria_filter])
union_positions = len(df[union_filter])

# Calculate sum of Realized PnL for each group
small_size_pnl_sum = df.loc[small_size_filter, 'PnL_Numeric'].sum()
short_time_pnl_sum = df.loc[short_time_filter, 'PnL_Numeric'].sum()
both_criteria_pnl_sum = df.loc[both_criteria_filter, 'PnL_Numeric'].sum()
union_pnl_sum = df.loc[union_filter, 'PnL_Numeric'].sum()

# Print results
print(f"분석 결과:")
print(f"전체 포지션 수: {total_positions}")
print(f"1. Max Size USDT가 1000 USDT 이하인 포지션: {small_size_positions} ({small_size_positions/total_positions:.2%})")
print(f"   - Realized PnL 합계: {small_size_pnl_sum:.2f} USDT")
print(f"2. 오픈/클로즈 시간 간격이 10분 이하인 포지션: {short_time_positions} ({short_time_positions/total_positions:.2%})")
print(f"   - Realized PnL 합계: {short_time_pnl_sum:.2f} USDT")
print(f"3. 위 두 조건을 모두 만족하는 포지션: {both_criteria_positions} ({both_criteria_positions/total_positions:.2%})")
print(f"   - Realized PnL 합계: {both_criteria_pnl_sum:.2f} USDT")
print(f"4. 위 조건들의 합집합(1번 또는 2번 조건 만족): {union_positions} ({union_positions/total_positions:.2%})")
print(f"   - Realized PnL 합계: {union_pnl_sum:.2f} USDT")

# Save detailed results to a new CSV
result_df = df.copy()
result_df['Small_Size'] = result_df['Size_USDT_Numeric'] <= 1000
result_df['Short_Duration'] = result_df['Time_Diff_Minutes'] <= 10
result_df['Both_Criteria'] = result_df['Small_Size'] & result_df['Short_Duration']
result_df['Union_Criteria'] = result_df['Small_Size'] | result_df['Short_Duration']

# Save to CSV with detailed analysis (uncomment if needed)
# result_df.to_csv('analysis_results/u_guys_r_rly_dum/overall/position_analysis_results.csv', index=False)

# To see detailed positions that meet specific criteria, uncomment and modify as needed
# Display a few small size positions
print("\n작은 포지션 예시 (최대 5개):")
print(df[df['Size_USDT_Numeric'] <= 1000][['Symbol', 'Direction', 'Max Size USDT', 'Open_Time_UTC', 'Close_Time_UTC', 'Time_Diff_Minutes']].head(5))

# Display a few quick trades
print("\n빠른 거래 예시 (최대 5개):")
print(df[df['Time_Diff_Minutes'] <= 10][['Symbol', 'Direction', 'Max Size USDT', 'Open_Time_UTC', 'Close_Time_UTC', 'Time_Diff_Minutes']].head(5))

# Display positions that meet both criteria
print("\n작은 포지션 & 빠른 거래 (최대 5개):")
print(df[(df['Size_USDT_Numeric'] <= 1000) & (df['Time_Diff_Minutes'] <= 10)][['Symbol', 'Direction', 'Max Size USDT', 'Open_Time_UTC', 'Close_Time_UTC', 'Time_Diff_Minutes']].head(5))
