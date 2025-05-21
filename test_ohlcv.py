import ccxt
import time

def fetch_futures_ohlcv(symbol='BTCUSDT', timeframe='1h', limit=100):
    # CCXT 바이낸스 선물 인스턴스 생성
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 중요: 선물 시장
        }
    })
    
    try:
        # 마켓 정보 로드
        exchange.load_markets()
        
        # 사용 가능한 심볼 예시 출력
        print(f"사용 가능한 심볼 예시: {list(exchange.markets.keys())[:5]}")
        
        # 심볼이 존재하는지 확인
        if symbol not in exchange.markets:
            print(f"심볼 '{symbol}'은(는) 바이낸스 선물 마켓에 없습니다.")
            # 테스트를 위해 BTC로 대체
            symbol = 'BTCUSDT'
            print(f"테스트를 위해 '{symbol}'로 대체합니다.")
        
        # OHLCV 데이터 가져오기
        print(f"{symbol}의 {timeframe} 데이터를 가져오는 중...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        # 결과 출력
        print(f"가져온 캔들 수: {len(ohlcv)}\n")
        for candle in ohlcv[:5]:  # 처음 5개만 출력
            timestamp, open_, high, low, close, volume = candle
            print(f"{exchange.iso8601(timestamp)} O:{open_} H:{high} L:{low} C:{close} V:{volume}")
        print("...")
        
        return ohlcv
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def main():
    # 몇 가지 심볼 테스트
    symbols_to_test = [
        'BTCUSDT',          # 표준 비트코인 선물
        'FARTCOIN/USDT',    # 슬래시 포함 포맷
        'FARTCOIN/USDT:USDT' # 사용자가 언급한 형식
    ]
    
    for symbol in symbols_to_test:
        print(f"\n===== {symbol} 테스트 =====")
        fetch_futures_ohlcv(symbol)
        time.sleep(1)  # API 제한 방지

# 메인 함수 실행
if __name__ == "__main__":
    main()