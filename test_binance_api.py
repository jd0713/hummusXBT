#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import time
from datetime import datetime

def test_binance_usdm_api():
    """바이낸스 USDM API 테스트"""
    print("바이낸스 USDM API 테스트 시작...")
    
    # 바이낸스 USDM 선물 거래소 객체 생성
    exchange = ccxt.binanceusdm()
    
    # 테스트할 심볼 목록
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NEIRO/USDT', 'GOAT/USDT']
    
    for symbol in test_symbols:
        print(f"\n{symbol} 정보 조회 중...")
        
        try:
            # 1. 티커 정보 조회
            ticker = exchange.fetch_ticker(symbol)
            print(f"티커 정보:")
            print(f"  - 마지막 가격: {ticker['last']}")
            print(f"  - 24시간 거래량(Base): {ticker['volume']}")
            print(f"  - 24시간 거래량(Quote): {ticker['quoteVolume']} USDT")
            print(f"  - 최고 매수호가: {ticker['bid']}")
            print(f"  - 최저 매도호가: {ticker['ask']}")
            
            # 스프레드 계산
            spread = ((ticker['ask'] - ticker['bid']) / ticker['bid']) * 100 if ticker['bid'] else 0
            print(f"  - 스프레드: {spread:.4f}%")
            
            # 2. 주문장(Order Book) 정보 조회
            order_book = exchange.fetch_order_book(symbol)
            bids = order_book['bids']
            asks = order_book['asks']
            
            # 주문장 깊이 계산 (상위 10개 주문)
            bids_depth = sum([bid[0] * bid[1] for bid in bids[:10]]) if bids else 0
            asks_depth = sum([ask[0] * ask[1] for ask in asks[:10]]) if asks else 0
            
            print(f"주문장 정보:")
            print(f"  - 매수 주문 수: {len(bids)}")
            print(f"  - 매도 주문 수: {len(asks)}")
            print(f"  - 상위 10개 매수 주문 깊이: {bids_depth:.2f} USDT")
            print(f"  - 상위 10개 매도 주문 깊이: {asks_depth:.2f} USDT")
            print(f"  - 총 주문장 깊이(상위 10개): {bids_depth + asks_depth:.2f} USDT")
            
            # 3. 24시간 거래 정보 조회
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1)
            if ohlcv:
                daily_data = ohlcv[0]
                print(f"24시간 OHLCV 정보:")
                print(f"  - 시가: {daily_data[1]}")
                print(f"  - 고가: {daily_data[2]}")
                print(f"  - 저가: {daily_data[3]}")
                print(f"  - 종가: {daily_data[4]}")
                print(f"  - 거래량: {daily_data[5]}")
            
            # 4. 시장 정보 조회
            market = exchange.market(symbol)
            print(f"시장 정보:")
            print(f"  - 최소 주문 수량: {market.get('limits', {}).get('amount', {}).get('min', 'N/A')}")
            print(f"  - 가격 단위(틱 사이즈): {market.get('precision', {}).get('price', 'N/A')}")
            print(f"  - 수량 단위: {market.get('precision', {}).get('amount', 'N/A')}")
            
            # 유동성 판단 기준
            is_low_liquidity = False
            reasons = []
            
            # 거래량 기준 (1천만 달러 미만)
            if ticker['quoteVolume'] < 10000000:
                is_low_liquidity = True
                reasons.append(f"낮은 거래량 ({ticker['quoteVolume']:.2f} USDT)")
            
            # 스프레드 기준 (0.3% 이상)
            if spread > 0.3:
                is_low_liquidity = True
                reasons.append(f"높은 스프레드 ({spread:.4f}%)")
            
            # 주문장 깊이 기준 (50만 달러 미만)
            total_depth = bids_depth + asks_depth
            if total_depth < 500000:
                is_low_liquidity = True
                reasons.append(f"얕은 주문장 깊이 ({total_depth:.2f} USDT)")
            
            if is_low_liquidity:
                print(f"\n⚠️ {symbol}은(는) 유동성이 낮습니다!")
                print(f"  - 이유: {', '.join(reasons)}")
            else:
                print(f"\n✅ {symbol}은(는) 유동성이 충분합니다.")
            
        except Exception as e:
            print(f"❌ {symbol} 조회 중 오류 발생: {str(e)}")
        
        # API 호출 제한을 피하기 위한 지연
        time.sleep(1)
    
    print("\n바이낸스 USDM API 테스트 완료!")

def test_symbol_format_conversion():
    """CSV 파일에서 추출한 심볼 형식을 바이낸스 API 형식으로 변환 테스트"""
    print("\n심볼 형식 변환 테스트:")
    
    test_symbols = [
        'BTCUSDT', 
        'ETHUSDT', 
        'SOLUSDT', 
        'NEIROUSDT', 
        'GOATUSDT',
        '1000PEPEUSDT',
        'NEIROETHUSDT'
    ]
    
    for original_symbol in test_symbols:
        # USDT 페어 처리
        if original_symbol.endswith('USDT'):
            base = original_symbol[:-4]
            quote = 'USDT'
            formatted_symbol = f"{base}/{quote}"
        # ETH 페어 처리 (NEIROETHUSDT -> NEIRO/ETH:USDT)
        elif 'ETH' in original_symbol and original_symbol.endswith('USDT'):
            parts = original_symbol.split('ETH')
            base = parts[0]
            formatted_symbol = f"{base}/ETH:USDT"
        else:
            formatted_symbol = original_symbol
        
        print(f"원본: {original_symbol} -> 변환: {formatted_symbol}")

if __name__ == "__main__":
    print("=== 바이낸스 USDM API 테스트 스크립트 ===")
    
    # 심볼 형식 변환 테스트
    test_symbol_format_conversion()
    
    # API 테스트
    test_binance_usdm_api()
