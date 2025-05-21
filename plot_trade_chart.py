#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import pytz
import time

def generate_mock_ohlcv(start_time, end_time, timeframe='1h'):
    """
    Generate mock OHLCV data for demonstration when real data is not available
    """
    # Create a date range from start_time to end_time with hourly intervals
    date_range = pd.date_range(start=start_time, end=end_time, freq=timeframe)
    
    # Generate mock price data with a reasonable trend
    base_price = 100.0  # Starting price
    np.random.seed(42)  # For reproducibility
    
    # Create price movements with some randomness but overall trend
    price_changes = np.random.normal(0, 1, len(date_range))
    cumulative_changes = np.cumsum(price_changes) * 2  # Scale the changes
    
    # Create a dataframe with mock OHLCV data
    mock_data = pd.DataFrame(index=date_range)
    mock_data['close'] = base_price + cumulative_changes
    
    # Generate open, high, low based on close
    mock_data['open'] = mock_data['close'].shift(1).fillna(base_price)
    daily_volatility = mock_data['close'] * 0.02  # 2% daily volatility
    mock_data['high'] = mock_data[['open', 'close']].max(axis=1) + daily_volatility
    mock_data['low'] = mock_data[['open', 'close']].min(axis=1) - daily_volatility
    
    # Generate mock volume
    mock_data['volume'] = np.random.lognormal(10, 1, len(date_range))
    
    return mock_data

def fetch_ohlcv_data(symbol, start_time, end_time, timeframe='1h'):
    """
    Fetch OHLCV data from Binance using CCXT.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT' or 'BTC/USDT')
        start_time (datetime): Start time for data collection
        end_time (datetime): End time for data collection
        timeframe (str): Timeframe for candlesticks ('1h', '4h', '1d', etc.)
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Initialize Binance Futures client
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # Use futures API
            'adjustForTimeDifference': True,
        },
        'enableRateLimit': True,
    })
    
    # Convert datetime to milliseconds timestamp
    since = int(start_time.timestamp() * 1000)
    until = int(end_time.timestamp() * 1000)
    
    # Fetch all data in chunks to handle API limits
    all_ohlcv = []
    current_since = since
    
    try:
        # 마켓 정보 로드
        exchange.load_markets()
        print(f"사용 가능한 심볼 예시: {list(exchange.markets.keys())[:5]}")
        
        # 심볼이 있는지 확인하고 없으면 그대로 사용 시도 (test_ohlcv.py 결과 참고)
        # FARTCOIN/USDT:USDT와 같은 형식이 직접 작동한다고 확인됨
        if symbol not in exchange.markets and ':' not in symbol:
            # 슬래시가 있는 형식은 유지
            if '/' in symbol:
                # FARTCOIN/USDT -> FARTCOIN/USDT:USDT 형식 시도
                modified_symbol = f"{symbol}:USDT"
                print(f"원본 심볼 {symbol}을(를) {modified_symbol}로 시도합니다")
                symbol = modified_symbol
        
        print(f"{symbol}의 {timeframe} 데이터를 가져오는 중...")
    except Exception as e:
        print(f"시장 정보 로드 중 오류: {e}")
    
    while current_since < until:
        try:
            print(f"{symbol} 데이터 가져오기: {datetime.fromtimestamp(current_since/1000)}")
            
            # 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000  # Maximum limit per request
            )
            
            if not ohlcv:
                print("반환된 데이터 없음, 종료합니다")
                break
                
            all_ohlcv.extend(ohlcv)
            print(f"캔들 {len(ohlcv)}개 가져옴, 총 {len(all_ohlcv)}개")
            
            # 다음 페이지를 위한 타임스탬프 업데이트
            current_since = ohlcv[-1][0] + 1
            
            # API 제한 준수
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"데이터 가져오기 오류: {e}")
            
            # 만약 에러가 발생했다면 모의 데이터 생성
            if len(all_ohlcv) == 0:
                print("모의 데이터를 생성합니다")
                mock_data = generate_mock_ohlcv(start_time, end_time)
                return mock_data
            else:
                # 일부 데이터라도 받았다면 중단
                print("일부 데이터를 받았으므로 진행합니다")
                break
    
    # 데이터 프레임으로 변환
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 요청된 시간 범위로 필터링
        df = df[(df.index >= pd.to_datetime(start_time)) & (df.index <= pd.to_datetime(end_time))]
        
        print(f"최종 캔들 수: {len(df)}")
        return df
    else:
        # 데이터를 가져오지 못했다면 모의 데이터 생성
        print("데이터를 가져오지 못했으므로 모의 데이터를 생성합니다")
        return generate_mock_ohlcv(start_time, end_time)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Filter to requested time range
    df = df[(df.index >= pd.to_datetime(start_time)) & (df.index <= pd.to_datetime(end_time))]
    
    return df

def parse_trade_info(trade_info):
    """
    Parse trade information from the string format.
    
    Args:
        trade_info (str): Trade information in string format
        
    Returns:
        dict: Dictionary with parsed trade information
    """
    trade_data = {}
    
    # Parse symbol
    symbol_line = [line for line in trade_info.split('\n') if "심볼 (Symbol)" in line][0]
    trade_data['symbol'] = symbol_line.split(': ')[1].strip()
    
    # Parse direction
    direction_line = [line for line in trade_info.split('\n') if "방향 (Direction)" in line][0]
    trade_data['direction'] = direction_line.split(': ')[1].strip()
    
    # Parse entry price
    entry_price_line = [line for line in trade_info.split('\n') if "진입가 (Entry Price)" in line][0]
    entry_price_str = entry_price_line.split(': ')[1].strip().replace(" USDT", "").replace(",", "")
    trade_data['entry_price'] = float(entry_price_str)
    
    # Parse open and close times
    open_time_line = [line for line in trade_info.split('\n') if "오픈 시간 (Open Time KST)" in line][0]
    close_time_line = [line for line in trade_info.split('\n') if "종료 시간 (Close Time KST)" in line][0]
    
    open_time_str = open_time_line.split(': ')[1].strip()
    close_time_str = close_time_line.split(': ')[1].strip()
    
    # Convert KST times to UTC
    kst_tz = pytz.timezone('Asia/Seoul')
    utc_tz = pytz.UTC
    
    open_time = kst_tz.localize(datetime.strptime(open_time_str.replace(' KST', ''), '%Y-%m-%d %H:%M'))
    close_time = kst_tz.localize(datetime.strptime(close_time_str.replace(' KST', ''), '%Y-%m-%d %H:%M'))
    
    trade_data['open_time'] = open_time.astimezone(utc_tz).replace(tzinfo=None)
    trade_data['close_time'] = close_time.astimezone(utc_tz).replace(tzinfo=None)
    
    # Parse realized PnL
    pnl_line = [line for line in trade_info.split('\n') if "실현 손익 (Realized PnL)" in line][0]
    pnl_str = pnl_line.split(': ')[1].split(' USDT')[0].replace(",", "").replace("+", "")
    trade_data['realized_pnl'] = float(pnl_str)
    
    return trade_data

def create_trade_chart(trade_info):
    """
    Create a candlestick chart with trade entry and exit points.
    
    Args:
        trade_info (str): Trade information in string format
    """
    # Parse trade information
    trade_data = parse_trade_info(trade_info)
    
    # test_ohlcv.py에서 확인된 대로 FARTCOIN/USDT:USDT 형식 직접 사용
    symbol = trade_data['symbol']
    
    print(f"Using symbol format: {symbol}")
    
    # Calculate time range (2 weeks before entry to 2 weeks after exit)
    start_time = trade_data['open_time'] - timedelta(days=14)
    end_time = trade_data['close_time'] + timedelta(days=14)
    
    # Fetch OHLCV data
    ohlcv_data = fetch_ohlcv_data(symbol, start_time, end_time)
    
    if ohlcv_data.empty:
        print(f"No data found for {symbol} between {start_time} and {end_time}")
        return
    
    # Create figure with volume subplot (no volume label)
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.8, 0.2],
        subplot_titles=(f"{trade_data['symbol']} Price", "")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_data.index,
            open=ohlcv_data['open'],
            high=ohlcv_data['high'],
            low=ohlcv_data['low'],
            close=ohlcv_data['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=ohlcv_data.index,
            y=ohlcv_data['volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # Find nearest data points to entry and exit times
    entry_time = trade_data['open_time']
    exit_time = trade_data['close_time']
    
    # Find nearest data points to entry and exit times
    try:
        entry_idx = ohlcv_data.index[ohlcv_data.index.get_indexer([entry_time], method='nearest')[0]]
        exit_idx = ohlcv_data.index[ohlcv_data.index.get_indexer([exit_time], method='nearest')[0]]
        
        entry_price = float(ohlcv_data.loc[entry_idx, 'close'])
        exit_price = float(ohlcv_data.loc[exit_idx, 'close'])
    except (IndexError, KeyError) as e:
        print(f"Error finding entry/exit points: {e}")
        # Use the first and last points as fallback
        entry_idx = ohlcv_data.index[0]
        exit_idx = ohlcv_data.index[-1]
        entry_price = float(ohlcv_data.iloc[0]['close'])
        exit_price = float(ohlcv_data.iloc[-1]['close'])
    
    # Add entry and exit markers
    marker_color = 'green' if trade_data['direction'] == 'Long' else 'red'
    opposite_color = 'red' if trade_data['direction'] == 'Long' else 'green'
    
    # Entry marker
    fig.add_trace(
        go.Scatter(
            x=[entry_idx],
            y=[entry_price],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if trade_data['direction'] == 'Long' else 'triangle-down',
                size=15,
                color=marker_color,
                line=dict(width=2, color='black')
            ),
            name=f"Entry ({trade_data['direction']})",
            text=f"Entry: {entry_price:.4f}",
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # Exit marker
    fig.add_trace(
        go.Scatter(
            x=[exit_idx],
            y=[exit_price],
            mode='markers',
            marker=dict(
                symbol='triangle-down' if trade_data['direction'] == 'Long' else 'triangle-up',
                size=15,
                color=opposite_color,
                line=dict(width=2, color='black')
            ),
            name="Exit",
            text=f"Exit: {exit_price:.4f}",
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # Add shapes for entry and exit lines (safer than vlines with pandas timestamps)
    fig.add_shape(
        type="line",
        x0=entry_idx,
        x1=entry_idx,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=marker_color, width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=exit_idx,
        x1=exit_idx,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=opposite_color, width=1, dash="dash"),
    )
    
    # Add annotations for entry and exit
    fig.add_annotation(
        x=entry_idx,
        y=1.05,
        yref="paper",
        text="Entry",
        showarrow=False,
        font=dict(color=marker_color),
    )
    
    fig.add_annotation(
        x=exit_idx,
        y=1.05,
        yref="paper",
        text="Exit",
        showarrow=False,
        font=dict(color=opposite_color),
    )
    
    # Format the chart
    pnl_sign = "+" if trade_data['realized_pnl'] > 0 else ""
    fig.update_layout(
        title=f"{trade_data['symbol']} {trade_data['direction']} Trade - PnL: {pnl_sign}{trade_data['realized_pnl']:.2f} USDT",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=800
    )
    
    # Save the chart
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize symbol name for filename (remove special characters)
    safe_symbol = trade_data['symbol'].replace('/', '_').replace(':', '_').replace('\\', '_')
    
    trade_type = "largest" if trade_data['realized_pnl'] > 0 else "smallest"
    output_file = os.path.join(output_dir, f"{safe_symbol}_{trade_type}_pnl_trade.html")
    
    fig.write_html(output_file)
    print(f"Chart saved to {output_file}")

def main():
    # Trade information from analyzed data
    max_pnl_trade_info = """🔼 가장 큰 실현 손익(Realized PnL)의 거래:
심볼 (Symbol): ETHUSDT
방향 (Direction): Long
진입가 (Entry Price): 2,623.283 USDT
포지션 크기 (Position Size): 556.902 (1,460,911.35 USDT)
실현 손익 (Realized PnL): +370,418.00 USDT (25.36 %)
오픈 시간 (Open Time KST): 2024-11-08 17:48 KST
종료 시간 (Close Time KST): 2024-11-12 09:43 KST
거래 지속 기간 (Duration): 3 days, 15 hours, 55 minutes
최대 수익 (Runup): 29.31 % 3,392 USDT
최대 손실 (Drawdown): nan"""

    min_pnl_trade_info = """🔽 가장 작은 실현 손익(Realized PnL)의 거래:
심볼 (Symbol): XRPUSDT
방향 (Direction): Long
진입가 (Entry Price): 2.645 USDT
포지션 크기 (Position Size): 770,000 (2,036,601.43 USDT)
실현 손익 (Realized PnL): -125,337.10 USDT (-6.15 %)
오픈 시간 (Open Time KST): 2025-02-04 16:54 KST
종료 시간 (Close Time KST): 2025-02-07 00:29 KST
거래 지속 기간 (Duration): 2 days, 7 hours, 35 minutes
최대 수익 (Runup): 2.40 % 2.71 USDT
최대 손실 (Drawdown): -11.86 % 2.33 USDT"""

    # Create charts
    print("Creating chart for the trade with largest PnL...")
    create_trade_chart(max_pnl_trade_info)
    
    print("\nCreating chart for the trade with smallest PnL...")
    create_trade_chart(min_pnl_trade_info)

if __name__ == "__main__":
    main()
