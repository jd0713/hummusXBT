import pandas as pd
import os
from datetime import datetime

def extract_capital_data(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter data from September 2024 onwards
    sept_2024 = pd.Timestamp('2024-09-01')
    filtered_df = df[df['time'] >= sept_2024].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    
    if filtered_df.empty:
        print("No data found from September 2024 onwards.")
        return
    
    # Create result dataframe with time and capital columns
    result = []
    
    # First row of September data: Use the start_capital
    result.append({
        'time': filtered_df.iloc[0]['time'],
        'capital': filtered_df.iloc[0]['start_capital']
    })
    
    # For all subsequent rows: Use the end_capital
    for i in range(1, len(filtered_df)):  # Start from the second row (index 1)
        result.append({
            'time': filtered_df.iloc[i]['time'],
            'capital': filtered_df.iloc[i]['end_capital']
        })
    
    # Convert to dataframe
    result_df = pd.DataFrame(result)
    
    # Remove any duplicates based on time
    result_df = result_df.drop_duplicates(subset=['time'])
    
    # Sort by time
    result_df = result_df.sort_values('time')
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Data extracted and saved to {output_path}")
    print(f"Total rows: {len(result_df)}")

if __name__ == "__main__":
    # Input file path
    input_file = '/Users/jdpark/Downloads/binance_leaderboard_analysis/overlap_analysis/output/model_portfolio/model_portfolio_capital.csv'
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/jdpark/Downloads/binance_leaderboard_analysis/overlap_analysis/output/model_portfolio'
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, 'capital_time_series.csv')
    
    # Extract data
    extract_capital_data(input_file, output_file)
