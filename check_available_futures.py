#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check which symbols from analyzed_data.csv are not available as perpetual futures on Binance and Bybit.
"""

import pandas as pd
import ccxt
import os
import time
from datetime import datetime

def load_symbols_from_csv(csv_path):
    """Load unique symbols from the analyzed data CSV file that were traded within the last year."""
    df = pd.read_csv(csv_path)
    
    # Convert Open_Time_UTC to datetime
    df['Open_Time_UTC'] = pd.to_datetime(df['Open_Time_UTC'])
    
    # Calculate the date one year ago from today
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
    
    # Filter trades that occurred within the last year
    recent_df = df[df['Open_Time_UTC'] >= one_year_ago]
    
    # Extract symbols and standardize format from recent trades only
    symbols = recent_df['Symbol'].unique()
    
    print(f"Total unique symbols in CSV: {len(df['Symbol'].unique())}")
    print(f"Symbols traded within the last year: {len(symbols)}")
    
    return symbols

def get_exchange_perpetual_futures(exchange_id):
    """Get all perpetual futures symbols from the specified exchange."""
    try:
        # Initialize the exchange
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        # Load markets
        markets = exchange.load_markets()
        
        # Filter for perpetual futures only
        perpetual_symbols = []
        raw_symbols = []
        
        for symbol, market in markets.items():
            # Store the raw symbol for debugging
            raw_symbols.append(symbol)
            
            # Different exchanges have different ways to identify perpetuals
            is_perpetual = False
            
            if exchange_id == 'binance':
                if (market.get('linear') and market.get('type') == 'swap') or \
                   (market.get('future') and market.get('settle') == 'USDT') or \
                   (market.get('contract') and market.get('linear')):
                    is_perpetual = True
            elif exchange_id == 'bybit':
                if (market.get('linear') and market.get('type') == 'swap') or \
                   (market.get('swap') and market.get('settle') == 'USDT') or \
                   (market.get('contract') and market.get('linear')):
                    is_perpetual = True
            
            if is_perpetual:
                # 원래 심볼도 저장 (ETH/BTC와 같은 특수 케이스를 위해)
                perpetual_symbols.append(symbol)
                
                # Extract the base symbol
                base = None
                quote = None
                
                if ':' in symbol:  # Format like BTC/USDT:USDT
                    parts = symbol.split('/')
                    if len(parts) > 0:
                        base = parts[0]
                        if len(parts) > 1:
                            quote_parts = parts[1].split(':')
                            quote = quote_parts[0] if quote_parts else None
                elif '/' in symbol:  # Format like BTC/USDT
                    parts = symbol.split('/')
                    if len(parts) > 0:
                        base = parts[0]
                        if len(parts) > 1:
                            quote = parts[1]
                
                if base:
                    # USDT 페어인 경우 표준 형식으로 추가
                    if quote == 'USDT':
                        standardized = f"{base}USDT"
                        perpetual_symbols.append(standardized)
                    
                    # BTC 페어인 경우 별도 형식으로 추가
                    elif quote == 'BTC':
                        standardized = f"{base}BTC"
                        perpetual_symbols.append(standardized)
                        # 바이비트에서 사용하는 ETHBTCUSDT 형식도 추가
                        if base == 'ETH':
                            perpetual_symbols.append("ETHBTCUSDT")
                    
                    # 특수 케이스: SHIB1000, SATS1000 등
                    if 'SHIB' in base or '1000SHIB' in base:
                        perpetual_symbols.append("SHIBUSDT")
                        perpetual_symbols.append("1000SHIBUSDT")
                        perpetual_symbols.append("SHIB1000USDT")
                        perpetual_symbols.append("SHIB1000")
                    
                    if 'SATS' in base or '1000SATS' in base:
                        perpetual_symbols.append("SATSUSDT")
                        perpetual_symbols.append("1000SATSUSDT")
                        perpetual_symbols.append("SATS1000USDT")
                        perpetual_symbols.append("SATS1000")
        
        # 중복 제거
        perpetual_symbols = list(set(perpetual_symbols))
        
        # Print some debug info for the first few symbols
        print(f"Sample raw symbols from {exchange_id}: {raw_symbols[:5]}")
        print(f"Sample standardized perpetual symbols from {exchange_id} (first 5): {perpetual_symbols[:5]}")
        print(f"Total perpetual symbols from {exchange_id}: {len(perpetual_symbols)}")
        
        # 특정 심볼이 있는지 확인 (디버깅용)
        special_symbols = ['ETHBTC', 'ETH/BTC', 'ETHBTCUSDT', 'MATICUSDT', 'POLYUSDT', '1000SHIBUSDT', 'SHIB1000USDT', 'SHIB1000']
        for s in special_symbols:
            if s in perpetual_symbols:
                print(f"  Found {s} in {exchange_id}")
        
        return perpetual_symbols
    except Exception as e:
        print(f"Error fetching {exchange_id} markets: {e}")
        return []

def normalize_symbol(symbol):
    """Normalize symbol to a standard format for comparison."""
    # Handle special cases
    if symbol == "1000SHIBUSDT":
        # 바이낸스: SHIBUSDT, 바이비트: SHIB1000USDT 또는 SHIB1000
        return ["SHIBUSDT", "SHIB1000USDT", "SHIB1000"]
    elif symbol == "1000SATSUSDT":
        # 바이낸스: SATSUSDT, 바이비트: SATS1000USDT 또는 SATS1000
        return ["SATSUSDT", "SATS1000USDT", "SATS1000"]
    elif symbol == "ETHBTC":
        # ETH/BTC 페어는 특별 처리
        # 바이비트에서는 ETHBTCUSDT 형식으로 표기될 수 있음
        return ["ETHBTC", "ETH/BTC", "ETHBTCUSDT"]
    elif symbol == "MATICUSDT":
        # MATIC은 Polygon(MATIC)의 리브랜딩으로 POLYUSDT로도 표기될 수 있음
        return ["MATICUSDT", "POLYUSDT"]
    
    # 기본 심볼 반환
    return [symbol]

def main():
    # Path to the analyzed data CSV
    csv_path = '/Users/jdpark/Downloads/binance_leaderboard_analysis/analysis_results/hummusXBT/overall/analyzed_data.csv'
    
    # Get symbols from CSV (only those traded within the last year)
    symbols = load_symbols_from_csv(csv_path)
    
    # Get perpetual futures from exchanges
    print("Fetching Binance perpetual futures...")
    binance_perpetuals = get_exchange_perpetual_futures('binance')
    print(f"Found {len(binance_perpetuals)} perpetual futures on Binance")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    print("Fetching Bybit perpetual futures...")
    bybit_perpetuals = get_exchange_perpetual_futures('bybit')
    print(f"Found {len(bybit_perpetuals)} perpetual futures on Bybit")
    
    # Normalize symbols for comparison
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols]
    
    # Check availability on exchanges
    not_on_binance = []
    not_on_bybit = []
    not_on_either = []
    
    for i, symbol in enumerate(symbols):
        normalized_variations = normalized_symbols[i]
        
        # 각 심볼에 대해 정규화된 모든 변형을 확인
        on_binance = False
        on_bybit = False
        
        # 원본 심볼 확인
        if symbol in binance_perpetuals:
            on_binance = True
        if symbol in bybit_perpetuals:
            on_bybit = True
        
        # 정규화된 변형 심볼 확인
        for norm_symbol in normalized_variations:
            if norm_symbol in binance_perpetuals:
                on_binance = True
            if norm_symbol in bybit_perpetuals:
                on_bybit = True
        
        # 결과 기록
        if not on_binance:
            not_on_binance.append(symbol)
        
        if not on_bybit:
            not_on_bybit.append(symbol)
        
        if not on_binance and not on_bybit:
            not_on_either.append(symbol)
            
        # 디버깅: 특정 심볼에 대한 상세 정보 출력
        if symbol in ['ETHBTC', '1000SHIBUSDT', 'MATICUSDT']:
            print(f"\nDEBUG for {symbol}:")
            print(f"  Normalized variations: {normalized_variations}")
            print(f"  Available on Binance: {on_binance}")
            print(f"  Available on Bybit: {on_bybit}")
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Symbols not available on Binance ({len(not_on_binance)}):") 
    for symbol in not_on_binance:
        print(f"  - {symbol}")
    
    print(f"\nSymbols not available on Bybit ({len(not_on_bybit)}):") 
    for symbol in not_on_bybit:
        print(f"  - {symbol}")
    
    print(f"\nSymbols not available on either exchange ({len(not_on_either)}):") 
    for symbol in not_on_either:
        print(f"  - {symbol}")
    
    # Save results to a CSV file
    results_dir = os.path.dirname(csv_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"unavailable_futures_last_year_{timestamp}.csv")
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Symbol': list(symbols),
        'Normalized_Symbol': normalized_symbols,
        'Available_on_Binance': [symbol not in not_on_binance for symbol in symbols],
        'Available_on_Bybit': [symbol not in not_on_bybit for symbol in symbols],
        'Available_on_Either': [symbol not in not_on_either for symbol in symbols]
    })
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
