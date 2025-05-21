#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
import argparse
import json
from pathlib import Path

# 유동성 분석을 위한 기준값 설정
VOLUME_THRESHOLD_USD = 10000000  # 하루 1000만 달러 이하면 유동성 낮음으로 판단
SPREAD_THRESHOLD_PERCENT = 0.3   # 스프레드가 0.3% 이상이면 유동성 낮음으로 판단
DEPTH_THRESHOLD_USD = 500000     # 주문장 깊이가 50만 달러 이하면 유동성 낮음으로 판단

def load_trader_data(file_path):
    """트레이더 데이터 파일을 로드하는 함수"""
    df = pd.read_csv(file_path)
    return df

def extract_unique_symbols(df):
    """데이터프레임에서 고유한 거래 심볼 추출"""
    return df['Symbol'].unique()

def setup_exchange():
    """Binance USDM 거래소 연결 설정"""
    exchange = ccxt.binanceusdm()
    return exchange

def convert_symbol_format(original_symbol):
    """CSV 파일의 심볼 형식을 CCXT API 형식으로 변환"""
    # 이미 CCXT 형식인 경우 그대로 반환
    if '/' in original_symbol:
        return original_symbol
    
    # USDT 페어 처리
    if original_symbol.endswith('USDT'):
        base = original_symbol[:-4]
        quote = 'USDT'
        return f"{base}/{quote}"
    
    # 다른 형식의 심볼은 그대로 반환
    return original_symbol

def get_market_data(exchange, original_symbol):
    """거래소로부터 심볼의 시장 데이터 가져오기"""
    try:
        # 심볼 형식 변환
        symbol = convert_symbol_format(original_symbol)
        
        # 티커 정보 가져오기
        ticker = exchange.fetch_ticker(symbol)
        
        # 주문장 정보 가져오기
        order_book = exchange.fetch_order_book(symbol, limit=20)
        
        # 스프레드 계산
        # 주문장에서 최고 매수호가와 최저 매도호가 가져오기
        best_bid = order_book['bids'][0][0] if order_book['bids'] else None
        best_ask = order_book['asks'][0][0] if order_book['asks'] else None
        
        if best_bid and best_ask:
            spread = ((best_ask - best_bid) / best_bid) * 100
        else:
            spread = None
        
        # 주문장 깊이 계산 (상위 20개 주문의 총 주문량)
        bids_depth = sum(bid[0] * bid[1] for bid in order_book['bids'][:20]) if order_book['bids'] else 0
        asks_depth = sum(ask[0] * ask[1] for ask in order_book['asks'][:20]) if order_book['asks'] else 0
        total_depth = bids_depth + asks_depth
        
        # 24시간 거래량
        volume_24h = ticker['quoteVolume']
        
        return {
            'symbol': original_symbol,
            'last_price': ticker['last'],
            'volume_24h': volume_24h,
            'spread_percent': spread,
            'depth_usd': total_depth,
            'bid': best_bid,
            'ask': best_ask,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        print(f"오류 발생 ({original_symbol}): {str(e)}")
        return {
            'symbol': original_symbol,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

def analyze_liquidity(market_data):
    """심볼의 유동성 분석"""
    if 'error' in market_data:
        return {
            'symbol': market_data['symbol'],
            'liquidity_status': 'unknown',
            'reason': f"오류: {market_data['error']}",
            'timestamp': market_data['timestamp'],
        }
    
    low_liquidity_reasons = []
    
    # 거래량 체크
    if market_data['volume_24h'] < VOLUME_THRESHOLD_USD:
        low_liquidity_reasons.append(f"24시간 거래량({market_data['volume_24h']:,.2f} USD)이 기준치({VOLUME_THRESHOLD_USD:,} USD) 미만")
    
    # 스프레드 체크
    if market_data['spread_percent'] and market_data['spread_percent'] > SPREAD_THRESHOLD_PERCENT:
        low_liquidity_reasons.append(f"스프레드({market_data['spread_percent']:.2f}%)가 기준치({SPREAD_THRESHOLD_PERCENT:.2f}%) 초과")
    
    # 주문장 깊이 체크
    if market_data['depth_usd'] < DEPTH_THRESHOLD_USD:
        low_liquidity_reasons.append(f"주문장 깊이({market_data['depth_usd']:,.2f} USD)가 기준치({DEPTH_THRESHOLD_USD:,} USD) 미만")
    
    liquidity_status = "low" if low_liquidity_reasons else "normal"
    
    return {
        'symbol': market_data['symbol'],
        'liquidity_status': liquidity_status,
        'reasons': ', '.join(low_liquidity_reasons) if low_liquidity_reasons else "유동성 충분",
        'volume_24h': market_data.get('volume_24h', 0),
        'spread_percent': market_data.get('spread_percent', 0),
        'depth_usd': market_data.get('depth_usd', 0),
        'last_price': market_data.get('last_price', 0),
        'timestamp': market_data['timestamp'],
    }

def save_results(results, output_path):
    """분석 결과를 파일로 저장"""
    # 데이터프레임으로 변환
    results_df = pd.DataFrame(results)
    
    # CSV 파일로 저장
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"분석 결과가 {output_path}에 저장되었습니다.")
    
    # 유동성이 낮은 심볼만 필터링하여 별도 저장
    low_liquidity_df = results_df[results_df['liquidity_status'] == 'low']
    low_liquidity_path = output_path.replace('.csv', '_low_liquidity.csv')
    low_liquidity_df.to_csv(low_liquidity_path, index=False, encoding='utf-8-sig')
    print(f"유동성이 낮은 심볼의 정보가 {low_liquidity_path}에 저장되었습니다.")
    
    return results_df, low_liquidity_df

def analyze_trader_symbols(trader_name, trader_file, output_dir):
    """트레이더의 거래 심볼 유동성 분석"""
    print(f"\n{trader_name} 트레이더의 거래 심볼 유동성 분석 시작...")
    
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 트레이더 데이터 로드
    df = load_trader_data(trader_file)
    
    # 고유 심볼 추출
    symbols = extract_unique_symbols(df)
    print(f"{trader_name} 트레이더가 거래한 고유 심볼 수: {len(symbols)}")
    
    # 거래소 설정
    exchange = setup_exchange()
    
    # 각 심볼 분석
    results = []
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {symbol} 분석 중...")
        
        # 시장 데이터 가져오기
        market_data = get_market_data(exchange, symbol)
        
        # 유동성 분석
        liquidity_analysis = analyze_liquidity(market_data)
        
        # 심볼의 거래 횟수, 평균 포지션 크기 등 추가 분석
        symbol_trades = df[df['Symbol'] == symbol]
        liquidity_analysis.update({
            'trade_count': len(symbol_trades),
            'avg_position_size_usdt': symbol_trades['Size_USDT_Abs'].mean(),
            'max_position_size_usdt': symbol_trades['Size_USDT_Abs'].max(),
            'total_pnl': symbol_trades['PnL_Numeric'].sum(),
            'avg_pnl_percent': symbol_trades['Realized PnL %'].str.rstrip(' %').astype(float).mean() if 'Realized PnL %' in symbol_trades.columns else 0,
        })
        
        results.append(liquidity_analysis)
        
        # API 호출 제한을 피하기 위해 지연
        time.sleep(0.5)
    
    # 결과 저장
    output_path = os.path.join(output_dir, f"{trader_name}_liquidity_analysis.csv")
    results_df, low_liquidity_df = save_results(results, output_path)
    
    print(f"{trader_name} 트레이더 분석 완료. 총 {len(symbols)} 심볼 중 {len(low_liquidity_df)} 심볼이 유동성이 낮은 것으로 판단됨.")
    
    # 유동성이 낮은 심볼의 거래 비중 분석
    if len(low_liquidity_df) > 0:
        low_liquidity_symbols = low_liquidity_df['symbol'].tolist()
        low_liquidity_trades = df[df['Symbol'].isin(low_liquidity_symbols)]
        
        total_trade_volume = df['Size_USDT_Abs'].sum()
        low_liquidity_volume = low_liquidity_trades['Size_USDT_Abs'].sum()
        volume_percentage = (low_liquidity_volume / total_trade_volume) * 100 if total_trade_volume > 0 else 0
        
        print(f"유동성이 낮은 심볼의 거래 비중: {volume_percentage:.2f}% (거래량 기준)")
        print(f"유동성이 낮은 심볼의 거래 비중: {len(low_liquidity_trades) / len(df) * 100:.2f}% (거래 횟수 기준)")
    
    return results_df, low_liquidity_df

def main():
    parser = argparse.ArgumentParser(description='트레이더의 거래 심볼 유동성 분석')
    parser.add_argument('--traders', nargs='+', default=['hummusXBT', 'Cyborg0578', 'ONLYCANDLE'],
                        help='분석할 트레이더 이름 (기본값: hummusXBT, Cyborg0578, ONLYCANDLE)')
    args = parser.parse_args()
    
    # 각 트레이더 분석
    all_results = {}
    for trader_name in args.traders:
        trader_file = f"analysis_results/{trader_name}/overall/analyzed_data.csv"
        output_dir = f"analysis_results/{trader_name}/liquidity"
        
        if not os.path.exists(trader_file):
            print(f"파일을 찾을 수 없음: {trader_file}")
            continue
        
        results_df, low_liquidity_df = analyze_trader_symbols(trader_name, trader_file, output_dir)
        all_results[trader_name] = {
            'results': results_df,
            'low_liquidity': low_liquidity_df
        }
    
    # 모든 트레이더의 결과 비교 분석
    if len(all_results) > 1:
        print("\n=== 트레이더 간 유동성 낮은 심볼 비교 ===")
        
        # 각 트레이더별 유동성이 낮은 심볼 집합
        low_liquidity_sets = {
            trader: set(data['low_liquidity']['symbol']) 
            for trader, data in all_results.items() 
            if len(data['low_liquidity']) > 0
        }
        
        # 모든 트레이더가 공통으로 거래한 유동성이 낮은 심볼
        if len(low_liquidity_sets) > 1:
            common_symbols = set.intersection(*low_liquidity_sets.values())
            print(f"모든 트레이더가 공통으로 거래한 유동성이 낮은 심볼: {common_symbols}")
            
            # 트레이더별 유동성이 낮은 심볼 수
            for trader, symbols in low_liquidity_sets.items():
                print(f"{trader}: 총 {len(symbols)}개의 유동성이 낮은 심볼 거래")
        
    print("\n모든 트레이더의 유동성 분석이 완료되었습니다.")

if __name__ == "__main__":
    main()
