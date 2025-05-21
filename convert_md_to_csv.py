#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from datetime import datetime
import argparse
import glob

def parse_md_file(md_file_path):
    """
    Parse the markdown file containing position cycle data and extract information.
    """
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract trader name from file path
    trader_name = os.path.basename(os.path.dirname(md_file_path)).replace('gmx_', '')
    
    # Parse position cycles
    positions = []
    cycles = re.split(r'### 사이클 #\d+', content)[1:]  # Split by cycle headers, ignore first part which is summary
    
    for cycle in cycles:
        # Extract cycle information
        position_direction = None
        position_symbol = None
        cycle_pnl = None
        cycle_pnl_percent = None
        open_time = None
        close_time = None
        
        # Find basic cycle info
        basic_info_match = re.search(r'- \*\*기간\*\*: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ~ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', cycle)
        if basic_info_match:
            open_time = basic_info_match.group(1)
            close_time = basic_info_match.group(2)
        
        # Extract Max Size USDT
        max_size_match = re.search(r'- \*\*최대 포지션 규모\*\*: \$(([0-9]{1,3},)*[0-9]+\.[0-9]+)', cycle)
        max_size_usdt = max_size_match.group(1).replace(',', '') if max_size_match else ""
        
        # Extract Total Trading Volume
        volume_match = re.search(r'- \*\*총 거래 규모\*\*: \$(([0-9]{1,3},)*[0-9]+\.[0-9]+)', cycle)
        volume = volume_match.group(1).replace(',', '') if volume_match else ""
        
        pnl_match = re.search(r'- \*\*손익\*\*: \$([0-9,.,-]+) \((이익|손실)\)', cycle)
        if pnl_match:
            cycle_pnl = pnl_match.group(1).replace(',', '')
        
        pnl_percent_match = re.search(r'- \*\*수익률\*\*: ([0-9,.,-]+)%', cycle)
        if pnl_percent_match:
            cycle_pnl_percent = pnl_percent_match.group(1)
        
        # Extract transaction details
        trade_table_match = re.search(r'\| 날짜 \| 액션 \| 규모 \| 마켓 \| PnL \|\n\|[- ]+\|[- ]+\|[- ]+\|[- ]+\|[- ]+\|(.*?)(?:\n\n|\Z)', cycle, re.DOTALL)
        
        if trade_table_match:
            trade_rows = [row.strip() for row in trade_table_match.group(1).strip().split('\n')]
            
            # Find first trade to determine symbol and direction
            if trade_rows:
                first_trade = re.match(r'\| (.*?) \| (.*?) \| \$(.*?) \| (Long|Short) (.*?) \|', trade_rows[0])
                if first_trade:
                    position_direction = first_trade.group(4)
                    symbol_raw = first_trade.group(5)
                    
                    # Convert symbol format from BTC/USD to BTCUSD
                    symbol = symbol_raw.replace('/', '')
                    position_symbol = symbol
            
            # Entry price is not needed
            
            # Add the position to our list
            positions.append({
                'Symbol': position_symbol,
                'Direction': position_direction,
                'Max Size USDT': f"{max_size_usdt} USDT" if max_size_usdt else "",
                'Volume': f"{volume} USDT" if volume else "",
                'Realized PnL': f"{cycle_pnl} USDT" if cycle_pnl else "",
                'Realized PnL %': f"{cycle_pnl_percent} %" if cycle_pnl_percent else "",
                'Open Time': format_time_for_csv(open_time) if open_time else "",
                'Close Time': format_time_for_csv(close_time) if close_time else "",
                'Trader': trader_name
            })
    
    return positions

def format_time_for_csv(time_str):
    """
    Convert time from markdown format to CSV format.
    Input: '2024-03-17 16:37:17'
    Output: '03/17/2024 16:37 UTC'
    """
    if not time_str:
        return ""
    
    try:
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%m/%d/%Y %H:%M UTC')
    except ValueError:
        return time_str

def write_to_csv(positions, output_file):
    """
    Write parsed positions to CSV file in the target format.
    """
    fieldnames = [
        'Symbol', 'Direction', 'Max Size USDT', 'Volume', 'Realized PnL', 
        'Realized PnL %', 'Open Time', 'Close Time'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for position in positions:
            # Filter to include only the fields we need
            row = {field: position.get(field, "") for field in fieldnames}
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Convert markdown position cycle data to CSV format')
    parser.add_argument('--trader', type=str, help='Trader name (e.g., btcinsider, majorswinger)')
    args = parser.parse_args()

    if args.trader:
        md_files = [f"/Users/jdpark/Downloads/binance_leaderboard_analysis/trader_data/gmx_{args.trader}/trader*_position_cycles.md"]
    else:
        # Process all markdown files in the trader_data directory
        md_files = glob.glob("/Users/jdpark/Downloads/binance_leaderboard_analysis/trader_data/gmx_*/trader*_position_cycles.md")

    for md_file in md_files:
        if not os.path.exists(md_file):
            continue
            
        trader_name = os.path.basename(os.path.dirname(md_file)).replace('gmx_', '')
        positions = parse_md_file(md_file)
        
        # Create output directory
        output_dir = f"/Users/jdpark/Downloads/binance_leaderboard_analysis/trader_data/{trader_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/{trader_name}_closed_positions_{timestamp}.csv"
        
        write_to_csv(positions, output_file)
        print(f"Converted {md_file} to {output_file}")

if __name__ == "__main__":
    main()
