#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ccxt
import json
import pprint

def print_api_response_structure():
    """바이낸스 USDM API 응답 구조 출력"""
    print("바이낸스 USDM API 응답 구조 확인")
    
    # 바이낸스 USDM 선물 거래소 객체 생성
    exchange = ccxt.binanceusdm()
    
    # 테스트할 심볼
    symbol = 'BTC/USDT'
    
    print(f"\n===== {symbol} 티커 정보 =====")
    ticker = exchange.fetch_ticker(symbol)
    print("티커 응답 구조:")
    pprint.pprint(ticker)
    
    print(f"\n===== {symbol} 주문장 정보 =====")
    order_book = exchange.fetch_order_book(symbol, limit=5)
    print("주문장 응답 구조:")
    pprint.pprint(order_book)
    
    print(f"\n===== {symbol} 24시간 거래 정보 =====")
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1)
    print("OHLCV 응답 구조:")
    pprint.pprint(ohlcv)
    
    print(f"\n===== {symbol} 마켓 정보 =====")
    market = exchange.market(symbol)
    print("마켓 정보 응답 구조:")
    pprint.pprint(market)
    
    # 마켓 리스트 확인 (일부만)
    print("\n===== 마켓 리스트 (일부) =====")
    markets = exchange.load_markets()
    market_list = list(markets.keys())[:10]  # 처음 10개만
    print(f"마켓 목록 (처음 10개): {market_list}")
    
    # 거래량 정보 확인을 위한 추가 API 호출
    print(f"\n===== {symbol} 24시간 거래량 정보 =====")
    try:
        tickers = exchange.fetch_tickers([symbol])
        print("티커스 응답 구조:")
        pprint.pprint(tickers)
    except Exception as e:
        print(f"티커스 조회 중 오류 발생: {str(e)}")
    
    # 심볼 정보 확인
    print(f"\n===== 심볼 정보 변환 테스트 =====")
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'NEIROETHUSDT', '1000PEPEUSDT']
    
    for original_symbol in test_symbols:
        try:
            # CCXT 형식으로 변환 시도
            ccxt_symbol = exchange.market_id(original_symbol)
            print(f"원본: {original_symbol} -> CCXT 변환: {ccxt_symbol}")
        except Exception as e:
            print(f"원본: {original_symbol} -> 변환 오류: {str(e)}")
            # 수동 변환 시도
            if original_symbol.endswith('USDT'):
                base = original_symbol[:-4]
                quote = 'USDT'
                formatted_symbol = f"{base}/{quote}"
                print(f"  수동 변환: {formatted_symbol}")

if __name__ == "__main__":
    print_api_response_structure()
