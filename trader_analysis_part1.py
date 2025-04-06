#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import os
import matplotlib.font_manager as fm
import shutil

# 한글 폰트 설정 (맥OS에서 사용 가능한 한글 폰트)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 원금과 기간 구분 날짜는 트레이더 설정에서 가져옴
# 이제 하드코딩된 값을 사용하지 않음

# 결과 저장 경로
RESULT_DIR = "analysis_results"

# 트레이더별 결과 디렉토리 경로 반환 함수
def get_trader_result_dirs(trader_id=None):
    """트레이더 ID에 따른 결과 디렉토리 경로 반환"""
    if trader_id:
        # 트레이더 설정 가져오기
        from trader_config import get_trader
        trader = get_trader(trader_id)
        use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
        
        trader_dir = os.path.join(RESULT_DIR, trader_id)
        overall_dir = os.path.join(trader_dir, "overall")
        
        # 기간 구분이 있는 트레이더만 period1, period2 디렉토리 경로 반환
        if use_periods:
            period1_dir = os.path.join(trader_dir, "period1")
            period2_dir = os.path.join(trader_dir, "period2")
        else:
            # 기간 구분이 없는 트레이더는 period1, period2 디렉토리 경로를 None으로 반환
            period1_dir = None
            period2_dir = None
    else:
        # 기본값 (하위 호환성 유지)
        trader_dir = RESULT_DIR
        period1_dir = os.path.join(RESULT_DIR, "period1")
        period2_dir = os.path.join(RESULT_DIR, "period2")
        overall_dir = os.path.join(RESULT_DIR, "overall")
    
    return trader_dir, period1_dir, period2_dir, overall_dir

def parse_datetime(dt_str):
    """UTC 시간 문자열을 datetime 객체로 변환"""
    try:
        # 날짜 형식: MM/DD/YYYY HH:MM UTC
        return datetime.strptime(dt_str.replace(' UTC', ''), '%m/%d/%Y %H:%M')
    except:
        return None

def convert_to_kst(dt):
    """UTC 시간을 KST(UTC+9)로 변환"""
    if dt is None:
        return None
    return dt + timedelta(hours=9)

def format_kst_time(dt):
    """KST 시간을 문자열로 포맷팅"""
    if dt is None:
        return "Unknown"
    return dt.strftime('%Y-%m-%d %H:%M') + " KST"

def parse_pnl(pnl_str):
    """PnL 문자열을 숫자로 변환"""
    try:
        # 콤마와 USDT 제거
        return float(pnl_str.replace(',', '').replace(' USDT', '').replace('-', '').strip())
    except:
        return 0

def parse_size_usdt(size_str):
    """거래 금액(USDT) 문자열을 숫자로 변환"""
    try:
        # 콤마와 USDT 제거, 음수 부호 처리
        clean_str = size_str.replace(',', '').replace(' USDT', '').strip()
        if clean_str.startswith('-'):
            return -float(clean_str[1:])
        return float(clean_str)
    except:
        return 0

def calculate_mdd(cumulative_returns):
    """최대 낙폭(MDD) 계산"""
    # 누적 최대값
    peak = cumulative_returns.cummax()
    # 현재 값과 현재까지의 최대값의 차이 계산
    drawdown = (cumulative_returns - peak) / peak
    # 최대 낙폭
    mdd = drawdown.min()
    return mdd

def calculate_annualized_return(initial_capital, final_capital, start_date, end_date):
    """연율화된 수익률 계산"""
    # 기간(년) 계산
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0
    
    # 총 수익률
    total_return = (final_capital - initial_capital) / initial_capital
    
    # 연율화된 수익률
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    return annualized_return

def analyze_pnl_by_periods(csv_file, period_split_date=None, trader_id=None):
    """트레이더의 PnL 분석 (기간별)"""
    # 기간 구분 날짜가 없으면 기본값 사용
    if period_split_date is None:
        period_split_date = PERIOD_SPLIT_DATE
    
    # 트레이더 설정 가져오기
    from trader_config import get_trader
    trader = get_trader(trader_id)
    use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
        
    # 트레이더별 결과 디렉토리 가져오기
    trader_dir, period1_dir, period2_dir, overall_dir = get_trader_result_dirs(trader_id)
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 날짜 파싱 및 KST로 변환
    df['Open_Time_UTC'] = df['Open Time'].apply(parse_datetime)
    df['Close_Time_UTC'] = df['Close Time'].apply(parse_datetime)
    df['Open_Time_KST'] = df['Open_Time_UTC'].apply(convert_to_kst)
    df['Close_Time_KST'] = df['Close_Time_UTC'].apply(convert_to_kst)
    
    # 문자열 형태의 KST 시간 추가
    df['Open_Time_KST_Str'] = df['Open_Time_KST'].apply(format_kst_time)
    df['Close_Time_KST_Str'] = df['Close_Time_KST'].apply(format_kst_time)
    
    # PnL 숫자로 변환
    df['PnL_Value'] = df['Realized PnL'].apply(lambda x: parse_pnl(x))
    df['PnL_Sign'] = df['Realized PnL'].apply(lambda x: -1 if '-' in x else 1)
    df['PnL_Numeric'] = df['PnL_Value'] * df['PnL_Sign']
    
    # 거래 금액 숫자로 변환
    df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(parse_size_usdt)
    df['Size_USDT_Abs'] = df['Size_USDT_Numeric'].abs()
    
    # 시간순 정렬 (오픈 시간 기준)
    df_sorted = df.sort_values('Open_Time_UTC').reset_index(drop=True)
    
    # 누적 PnL 계산
    df_sorted['Cumulative_PnL'] = df_sorted['PnL_Numeric'].cumsum()
    
    # 기간 구분
    df_sorted['Period'] = df_sorted['Close_Time_UTC'].apply(
        lambda x: 1 if x < period_split_date else 2
    )
    
    # 기간별 데이터 분리
    df_period1 = df_sorted[df_sorted['Period'] == 1].copy()
    df_period2 = df_sorted[df_sorted['Period'] == 2].copy()
    
    # 각 기간별 누적 PnL 재계산 (기간 시작점을 0으로)
    if not df_period1.empty:
        initial_pnl_period1 = 0
        df_period1.loc[:, 'Period_Cumulative_PnL'] = df_period1['PnL_Numeric'].cumsum() - initial_pnl_period1
    
    if not df_period2.empty:
        initial_pnl_period2 = 0
        df_period2.loc[:, 'Period_Cumulative_PnL'] = df_period2['PnL_Numeric'].cumsum() - initial_pnl_period2
    
    # 결과 디렉토리 생성
    os.makedirs(overall_dir, exist_ok=True)
    
    # 기간 구분이 있는 트레이더인 경우에만 period1, period2 디렉토리 생성
    if use_periods and period1_dir and period2_dir:
        os.makedirs(period1_dir, exist_ok=True)
        os.makedirs(period2_dir, exist_ok=True)
        
        # 결과 저장
        result_file_overall = os.path.join(overall_dir, "analyzed_data.csv")
        result_file_period1 = os.path.join(period1_dir, "analyzed_data.csv")
        result_file_period2 = os.path.join(period2_dir, "analyzed_data.csv")
        
        df_sorted.to_csv(result_file_overall, index=False)
        if not df_period1.empty:
            df_period1.to_csv(result_file_period1, index=False)
        if not df_period2.empty:
            df_period2.to_csv(result_file_period2, index=False)
        
        print(f"전체 분석 결과가 {result_file_overall}에 저장되었습니다.")
        if not df_period1.empty:
            print(f"기간 1 분석 결과가 {result_file_period1}에 저장되었습니다.")
        if not df_period2.empty:
            print(f"기간 2 분석 결과가 {result_file_period2}에 저장되었습니다.")
    else:
        # 기간 구분이 없는 트레이더인 경우
        result_file_overall = os.path.join(overall_dir, "analyzed_data.csv")
        df_sorted.to_csv(result_file_overall, index=False)
        print(f"분석 결과가 {result_file_overall}에 저장되었습니다.")
    
    return df_sorted, df_period1, df_period2


def analyze_pnl_without_periods(csv_file, trader_id=None):
    """기간 구분 없이 트레이더의 PnL 분석"""
    # 트레이더별 결과 디렉토리 가져오기
    trader_dir, period1_dir, period2_dir, overall_dir = get_trader_result_dirs(trader_id)
    
    # 결과 디렉토리 생성
    os.makedirs(overall_dir, exist_ok=True)
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 날짜 파싱 및 KST로 변환
    df['Open_Time_UTC'] = df['Open Time'].apply(parse_datetime)
    df['Close_Time_UTC'] = df['Close Time'].apply(parse_datetime)
    df['Open_Time_KST'] = df['Open_Time_UTC'].apply(convert_to_kst)
    df['Close_Time_KST'] = df['Close_Time_UTC'].apply(convert_to_kst)
    
    # 문자열 형태의 KST 시간 추가
    df['Open_Time_KST_Str'] = df['Open_Time_KST'].apply(format_kst_time)
    df['Close_Time_KST_Str'] = df['Close_Time_KST'].apply(format_kst_time)
    
    # PnL 숫자로 변환
    df['PnL_Value'] = df['Realized PnL'].apply(lambda x: parse_pnl(x))
    df['PnL_Sign'] = df['Realized PnL'].apply(lambda x: -1 if '-' in x else 1)
    df['PnL_Numeric'] = df['PnL_Value'] * df['PnL_Sign']
    
    # 거래 금액 숫자로 변환
    df['Size_USDT_Numeric'] = df['Max Size USDT'].apply(parse_size_usdt)
    df['Size_USDT_Abs'] = df['Size_USDT_Numeric'].abs()
    
    # 시간순 정렬 (오픈 시간 기준)
    df_sorted = df.sort_values('Open_Time_UTC').reset_index(drop=True)
    
    # 누적 PnL 계산
    df_sorted['Cumulative_PnL'] = df_sorted['PnL_Numeric'].cumsum()
    df_sorted['Period_Cumulative_PnL'] = df_sorted['Cumulative_PnL']  # 기간 구분 없이 동일한 값 사용
    
    # 결과 저장
    result_file_overall = os.path.join(overall_dir, "analyzed_data.csv")
    df_sorted.to_csv(result_file_overall, index=False)
    print(f"전체 분석 결과가 {result_file_overall}에 저장되었습니다.")
    
    return df_sorted

def calculate_total_trading_volume(df):
    """총 거래 금액 계산"""
    return df['Size_USDT_Abs'].sum()

def calculate_performance_metrics_without_capital(df_period, period_name):
    """원금 정보 없이 성과 지표 계산"""
    if df_period.empty:
        print(f"{period_name}에 해당하는 데이터가 없습니다.")
        return {}
    
    # 총 수익 계산
    total_pnl = df_period['PnL_Numeric'].sum()
    
    # 총 거래 금액
    total_volume = calculate_total_trading_volume(df_period)
    
    # 시작일과 종료일
    start_date = df_period['Close_Time_UTC'].min()
    end_date = df_period['Close_Time_UTC'].max()
    
    # 기간 (일)
    days = (end_date - start_date).days if start_date and end_date else 0
    
    # 일별 수익 계산을 위한 데이터 준비
    df_period.loc[:, 'Close_Date'] = df_period['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
    daily_pnl = df_period.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
    
    # 수익/손실 거래 비율
    win_trades = df_period[df_period['PnL_Numeric'] > 0]
    loss_trades = df_period[df_period['PnL_Numeric'] < 0]
    win_rate = len(win_trades) / len(df_period) * 100 if len(df_period) > 0 else 0
    
    # 평균 수익/손실
    avg_win = win_trades['PnL_Numeric'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['PnL_Numeric'].mean() if len(loss_trades) > 0 else 0
    
    # 일평균 수익
    daily_avg_pnl = total_pnl / days if days > 0 else 0
    
    # 심볼별 수익
    symbol_pnl = df_period.groupby('Symbol')['PnL_Numeric'].sum()
    top_symbols = symbol_pnl.nlargest(5)
    bottom_symbols = symbol_pnl.nsmallest(5)
    
    # 방향별 수익
    direction_pnl = df_period.groupby('Direction')['PnL_Numeric'].sum()
    
    return {
        'Period': period_name,
        'Total PnL': total_pnl,
        'Total Volume': total_volume,
        'Start Date': start_date,
        'End Date': end_date,
        'Trading Days': days,
        'Daily Avg PnL': daily_avg_pnl,
        'Win Rate (%)': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Top Symbols': top_symbols,
        'Bottom Symbols': bottom_symbols,
        'Direction PnL': direction_pnl
    }

def calculate_performance_metrics(df_period, initial_capital, period_name):
    """성과 지표 계산"""
    if df_period.empty:
        print(f"{period_name}에 해당하는 데이터가 없습니다.")
        return {}
    
    # 누적 수익 계산
    total_pnl = df_period['PnL_Numeric'].sum()
    
    # 총 거래 금액
    total_volume = calculate_total_trading_volume(df_period)
    
    # 시작일과 종료일
    start_date = df_period['Close_Time_UTC'].min()
    end_date = df_period['Close_Time_UTC'].max()
    
    # 기간 (일)
    days = (end_date - start_date).days if start_date and end_date else 0
    
    # 누적 수익률
    total_return = total_pnl / initial_capital
    
    # 일별 수익률 계산을 위한 데이터 준비
    df_period.loc[:, 'Close_Date'] = df_period['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
    daily_pnl = df_period.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
    
    # 일별 수익률
    daily_pnl['Daily_Return'] = daily_pnl['PnL_Numeric'] / initial_capital
    
    # 누적 수익률 계산
    daily_pnl['Cumulative_Return'] = (1 + daily_pnl['Daily_Return']).cumprod() - 1
    
    # 자산 가치 계산 (초기 자산 + 누적 PnL)
    daily_pnl['Cumulative_PnL'] = daily_pnl['PnL_Numeric'].cumsum()
    daily_pnl['Asset_Value'] = daily_pnl['Cumulative_PnL'] + initial_capital
    
    # 올바른 MDD 계산
    peak = daily_pnl['Asset_Value'].cummax()
    drawdown = (daily_pnl['Asset_Value'] - peak) / peak
    mdd = drawdown.min()
    
    # 연율화된 수익률
    annualized_return = calculate_annualized_return(
        initial_capital, 
        initial_capital * (1 + total_return), 
        start_date, 
        end_date
    )
    
    # 샤프 비율 (간단한 계산, 무위험 수익률 0% 가정)
    daily_std = daily_pnl['Daily_Return'].std()
    sharpe_ratio = np.sqrt(365) * daily_pnl['Daily_Return'].mean() / daily_std if daily_std > 0 else 0
    
    # 승률
    win_rate = (df_period['PnL_Numeric'] > 0).mean() * 100
    
    # 평균 거래 크기
    avg_trade_size = df_period['Size_USDT_Abs'].mean()
    
    # 최대 거래 크기
    max_trade_size = df_period['Size_USDT_Abs'].max()
    
    # 결과 저장
    metrics = {
        '기간': period_name,
        '원금 (USD)': initial_capital,
        '시작일 (KST)': convert_to_kst(start_date).strftime('%Y-%m-%d') if start_date else 'N/A',
        '종료일 (KST)': convert_to_kst(end_date).strftime('%Y-%m-%d') if end_date else 'N/A',
        '기간 (일)': days,
        '총 트레이드 수': len(df_period),
        '총 거래 금액 (USD)': total_volume,
        '평균 거래 크기 (USD)': avg_trade_size,
        '최대 거래 크기 (USD)': max_trade_size,
        '총 수익 (USD)': total_pnl,
        '총 수익률 (%)': total_return * 100,
        '연율화 수익률 (%)': annualized_return * 100,
        'MDD (%)': mdd * 100,
        '샤프 비율': sharpe_ratio,
        '승률 (%)': win_rate,
        '일평균 수익 (USD)': total_pnl / days if days > 0 else 0,
        '일평균 수익률 (%)': (total_return / days) * 100 if days > 0 else 0
    }
    
    return metrics

def print_performance_comparison(metrics_period1, metrics_period2, trader_id=None):
    """성과 지표 비교 출력"""
    # 트레이더별 결과 디렉토리 가져오기
    result_dir, period1_dir, period2_dir, overall_dir = get_trader_result_dirs(trader_id)
    
    print("\n===== 기간별 성과 비교 =====")
    
    # 데이터프레임으로 변환하여 출력
    metrics_df = pd.DataFrame({
        '지표': list(metrics_period1.keys()) if metrics_period1 else list(metrics_period2.keys()),
        '기간 1': list(metrics_period1.values()) if metrics_period1 else ['N/A'] * len(metrics_period2),
        '기간 2': list(metrics_period2.values()) if metrics_period2 else ['N/A'] * len(metrics_period1)
    })
    
    print(metrics_df)
    
    # 결과를 텍스트 파일로 저장
    with open(os.path.join(overall_dir, 'performance_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("===== 기간별 성과 비교 =====\n\n")
        f.write(metrics_df.to_string(index=False))
    
    print(f"\n성과 비교가 {os.path.join(overall_dir, 'performance_comparison.txt')}에 저장되었습니다.")

def evaluate_trader_skill(metrics_period1, metrics_period2, trader_id=None):
    """트레이더 실력 평가"""
    # 트레이더별 결과 디렉토리 가져오기
    result_dir, period1_dir, period2_dir, overall_dir = get_trader_result_dirs(trader_id)
    # 평가 기준
    evaluation_points = 0
    max_points = 10
    comments = []
    
    # 기간 1 평가
    if metrics_period1:
        # 수익률 평가
        if metrics_period1['총 수익률 (%)'] > 100:
            evaluation_points += 2
            comments.append("✓ 기간 1: 100% 이상의 매우 높은 수익률 달성")
        elif metrics_period1['총 수익률 (%)'] > 50:
            evaluation_points += 1
            comments.append("✓ 기간 1: 50% 이상의 높은 수익률 달성")
        
        # 샤프 비율 평가
        if metrics_period1['샤프 비율'] > 3:
            evaluation_points += 1
            comments.append("✓ 기간 1: 매우 높은 샤프 비율 (3 이상)")
        elif metrics_period1['샤프 비율'] > 2:
            evaluation_points += 0.5
            comments.append("✓ 기간 1: 좋은 샤프 비율 (2 이상)")
        
        # MDD 평가
        if abs(metrics_period1['MDD (%)']) < 20:
            evaluation_points += 1
            comments.append("✓ 기간 1: 낮은 MDD (20% 미만)")
        elif abs(metrics_period1['MDD (%)']) < 30:
            evaluation_points += 0.5
            comments.append("✓ 기간 1: 적절한 MDD (30% 미만)")
        else:
            comments.append("✗ 기간 1: 높은 MDD (30% 이상)")
    
    # 기간 2 평가
    if metrics_period2:
        # 수익률 평가
        if metrics_period2['총 수익률 (%)'] > 50:
            evaluation_points += 2
            comments.append("✓ 기간 2: 50% 이상의 매우 높은 수익률 달성 (큰 원금 고려)")
        elif metrics_period2['총 수익률 (%)'] > 25:
            evaluation_points += 1
            comments.append("✓ 기간 2: 25% 이상의 높은 수익률 달성 (큰 원금 고려)")
        
        # 샤프 비율 평가
        if metrics_period2['샤프 비율'] > 3:
            evaluation_points += 1
            comments.append("✓ 기간 2: 매우 높은 샤프 비율 (3 이상)")
        elif metrics_period2['샤프 비율'] > 2:
            evaluation_points += 0.5
            comments.append("✓ 기간 2: 좋은 샤프 비율 (2 이상)")
        
        # MDD 평가
        if abs(metrics_period2['MDD (%)']) < 20:
            evaluation_points += 1
            comments.append("✓ 기간 2: 낮은 MDD (20% 미만)")
        elif abs(metrics_period2['MDD (%)']) < 30:
            evaluation_points += 0.5
            comments.append("✓ 기간 2: 적절한 MDD (30% 미만)")
    
    # 전체 평가
    # 거래 횟수 평가
    total_trades = (metrics_period1.get('총 트레이드 수', 0) if metrics_period1 else 0) + \
                   (metrics_period2.get('총 트레이드 수', 0) if metrics_period2 else 0)
    if total_trades > 2000:
        evaluation_points += 1
        comments.append(f"✓ 충분한 거래 횟수 ({total_trades}회)")
    elif total_trades > 1000:
        evaluation_points += 0.5
        comments.append(f"✓ 적절한 거래 횟수 ({total_trades}회)")
    
    # 거래 금액 평가
    total_volume = (metrics_period1.get('총 거래 금액 (USD)', 0) if metrics_period1 else 0) + \
                   (metrics_period2.get('총 거래 금액 (USD)', 0) if metrics_period2 else 0)
    if total_volume > 50000000:  # 5천만 달러
        evaluation_points += 1
        comments.append(f"✓ 매우 큰 거래 금액 (${total_volume:,.2f})")
    elif total_volume > 10000000:  # 1천만 달러
        evaluation_points += 0.5
        comments.append(f"✓ 큰 거래 금액 (${total_volume:,.2f})")
    
    # 종합 평가
    rating = evaluation_points / max_points * 10  # 10점 만점
    
    evaluation_result = {
        "점수": f"{rating:.1f}/10",
        "평가": comments
    }
    
    # 종합 의견
    if rating >= 8:
        evaluation_result["종합 의견"] = "이 트레이더는 탁월한 실력을 갖춘 탑 트레이더로 평가됩니다. 높은 수익률, 효과적인 위험 관리, 일관된 성과를 보여주고 있습니다."
    elif rating >= 6:
        evaluation_result["종합 의견"] = "이 트레이더는 우수한 실력을 갖춘 트레이더로 평가됩니다. 좋은 수익률과 위험 관리 능력을 보여주고 있습니다."
    elif rating >= 4:
        evaluation_result["종합 의견"] = "이 트레이더는 평균 이상의 실력을 갖춘 트레이더로 평가됩니다. 개선의 여지가 있지만 전반적으로 긍정적인 성과를 보여주고 있습니다."
    else:
        evaluation_result["종합 의견"] = "이 트레이더는 추가적인 개선이 필요한 것으로 보입니다. 위험 관리와 수익률 측면에서 더 나은 결과를 위한 전략 조정이 권장됩니다."
    
    # 결과 저장
    with open(os.path.join(overall_dir, 'trader_evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write("===== 트레이더 실력 평가 =====\n\n")
        f.write(f"종합 점수: {evaluation_result['점수']}\n\n")
        f.write("세부 평가:\n")
        for comment in evaluation_result["평가"]:
            f.write(f"- {comment}\n")
        f.write(f"\n종합 의견:\n{evaluation_result['종합 의견']}")
    
    print(f"\n트레이더 평가가 {os.path.join(overall_dir, 'trader_evaluation.txt')}에 저장되었습니다.")
    
    return evaluation_result
