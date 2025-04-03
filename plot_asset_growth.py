#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import os
from trader_analysis_part1 import *

def plot_asset_growth_by_periods(df, df_period1, df_period2):
    """기간별 자산 성장 그래프 생성"""
    plt.figure(figsize=(15, 15))
    
    # 전체 기간 자산 그래프
    plt.subplot(3, 1, 1)
    
    # 초기 자산 설정
    initial_capital = INITIAL_CAPITAL_PERIOD1
    
    # 누적 PnL에 초기 자산 더하기
    asset_values = df['Cumulative_PnL'] + initial_capital
    
    # 원금 변경 시점에 자산 조정
    split_date_idx = df[df['Close_Time_UTC'] >= PERIOD_SPLIT_DATE].index.min()
    if not pd.isna(split_date_idx):
        # 원금 변경 시점의 누적 PnL
        pnl_at_split = df.loc[split_date_idx-1, 'Cumulative_PnL'] if split_date_idx > 0 else 0
        
        # 원금 변경 시점 이후의 자산 조정
        asset_values.loc[split_date_idx:] = df.loc[split_date_idx:, 'Cumulative_PnL'] - \
                                           df.loc[split_date_idx, 'Cumulative_PnL'] + \
                                           pnl_at_split + INITIAL_CAPITAL_PERIOD2
    
    plt.plot(df['Close_Time_KST'], asset_values, marker='o', linestyle='-', color='blue')
    plt.axvline(x=convert_to_kst(PERIOD_SPLIT_DATE), color='red', linestyle='--', label='원금 변경 시점')
    plt.title('전체 기간 자산 가치 변화 (한국 시간 기준)', fontsize=16)
    plt.ylabel('자산 가치 (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 기간 1 자산 그래프
    if not df_period1.empty:
        plt.subplot(3, 1, 2)
        
        # 기간 1 자산 계산
        asset_values_period1 = df_period1['Period_Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD1
        
        plt.plot(df_period1['Close_Time_KST'], asset_values_period1, 
                marker='o', linestyle='-', color='green')
        plt.title(f'기간 1 자산 가치 변화 (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # 수익률 표시 (%)
        final_asset = asset_values_period1.iloc[-1] if not asset_values_period1.empty else INITIAL_CAPITAL_PERIOD1
        total_return_pct_period1 = ((final_asset / INITIAL_CAPITAL_PERIOD1) - 1) * 100
        plt.annotate(f'총 수익률: {total_return_pct_period1:.2f}%', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 기간 2 자산 그래프
    if not df_period2.empty:
        plt.subplot(3, 1, 3)
        
        # 기간 2 자산 계산
        asset_values_period2 = df_period2['Period_Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD2
        
        plt.plot(df_period2['Close_Time_KST'], asset_values_period2, 
                marker='o', linestyle='-', color='purple')
        plt.title(f'기간 2 자산 가치 변화 (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # 수익률 표시 (%)
        final_asset = asset_values_period2.iloc[-1] if not asset_values_period2.empty else INITIAL_CAPITAL_PERIOD2
        total_return_pct_period2 = ((final_asset / INITIAL_CAPITAL_PERIOD2) - 1) * 100
        plt.annotate(f'총 수익률: {total_return_pct_period2:.2f}%', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OVERALL_DIR, 'asset_growth_by_periods.png'), dpi=300, bbox_inches='tight')
    print(f"기간별 자산 성장 그래프가 {os.path.join(OVERALL_DIR, 'asset_growth_by_periods.png')}에 저장되었습니다.")
    
    # 각 기간별 자산 그래프 개별 저장
    if not df_period1.empty:
        plt.figure(figsize=(12, 8))
        plt.plot(df_period1['Close_Time_KST'], asset_values_period1, 
                marker='o', linestyle='-', color='green')
        plt.title(f'기간 1 자산 가치 변화 (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.annotate(f'총 수익률: {total_return_pct_period1:.2f}%', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD1_DIR, 'asset_growth.png'), dpi=300, bbox_inches='tight')
        print(f"기간 1 자산 성장 그래프가 {os.path.join(PERIOD1_DIR, 'asset_growth.png')}에 저장되었습니다.")
    
    if not df_period2.empty:
        plt.figure(figsize=(12, 8))
        plt.plot(df_period2['Close_Time_KST'], asset_values_period2, 
                marker='o', linestyle='-', color='purple')
        plt.title(f'기간 2 자산 가치 변화 (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.annotate(f'총 수익률: {total_return_pct_period2:.2f}%', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD2_DIR, 'asset_growth.png'), dpi=300, bbox_inches='tight')
        print(f"기간 2 자산 성장 그래프가 {os.path.join(PERIOD2_DIR, 'asset_growth.png')}에 저장되었습니다.")

def plot_daily_asset_growth(df_period1, df_period2):
    """일별 자산 성장 및 수익률 그래프 생성"""
    # 기간 1 일별 자산 성장
    if not df_period1.empty:
        df_period1_copy = df_period1.copy()
        df_period1_copy.loc[:, 'Close_Date'] = df_period1_copy['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
        
        # 일별 PnL 합계
        daily_pnl1 = df_period1_copy.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
        
        # 누적 PnL 계산
        daily_pnl1['Cumulative_PnL'] = daily_pnl1['PnL_Numeric'].cumsum()
        
        # 자산 가치 계산
        daily_pnl1['Asset_Value'] = daily_pnl1['Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD1
        
        # 일별 수익률 계산
        daily_pnl1['Daily_Return_Pct'] = daily_pnl1['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD1 * 100
        
        # 그래프 그리기
        plt.figure(figsize=(15, 10))
        
        # 일별 자산 가치
        plt.subplot(2, 1, 1)
        plt.plot(daily_pnl1['Close_Date'], daily_pnl1['Asset_Value'], marker='o', linestyle='-', color='green')
        plt.title(f'기간 1 일별 자산 가치 (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 일별 수익률
        plt.subplot(2, 1, 2)
        colors = ['green' if ret >= 0 else 'red' for ret in daily_pnl1['Daily_Return_Pct']]
        plt.bar(daily_pnl1['Close_Date'], daily_pnl1['Daily_Return_Pct'], color=colors, alpha=0.7)
        plt.title(f'기간 1 일별 수익률 (%)', fontsize=16)
        plt.ylabel('일별 수익률 (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD1_DIR, 'daily_asset_growth.png'), dpi=300, bbox_inches='tight')
        print(f"기간 1 일별 자산 성장 그래프가 {os.path.join(PERIOD1_DIR, 'daily_asset_growth.png')}에 저장되었습니다.")
    
    # 기간 2 일별 자산 성장
    if not df_period2.empty:
        df_period2_copy = df_period2.copy()
        df_period2_copy.loc[:, 'Close_Date'] = df_period2_copy['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
        
        # 일별 PnL 합계
        daily_pnl2 = df_period2_copy.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
        
        # 누적 PnL 계산
        daily_pnl2['Cumulative_PnL'] = daily_pnl2['PnL_Numeric'].cumsum()
        
        # 자산 가치 계산
        daily_pnl2['Asset_Value'] = daily_pnl2['Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD2
        
        # 일별 수익률 계산
        daily_pnl2['Daily_Return_Pct'] = daily_pnl2['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD2 * 100
        
        # 그래프 그리기
        plt.figure(figsize=(15, 10))
        
        # 일별 자산 가치
        plt.subplot(2, 1, 1)
        plt.plot(daily_pnl2['Close_Date'], daily_pnl2['Asset_Value'], marker='o', linestyle='-', color='purple')
        plt.title(f'기간 2 일별 자산 가치 (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        plt.ylabel('자산 가치 (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 일별 수익률
        plt.subplot(2, 1, 2)
        colors = ['green' if ret >= 0 else 'red' for ret in daily_pnl2['Daily_Return_Pct']]
        plt.bar(daily_pnl2['Close_Date'], daily_pnl2['Daily_Return_Pct'], color=colors, alpha=0.7)
        plt.title(f'기간 2 일별 수익률 (%)', fontsize=16)
        plt.ylabel('일별 수익률 (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD2_DIR, 'daily_asset_growth.png'), dpi=300, bbox_inches='tight')
        print(f"기간 2 일별 자산 성장 그래프가 {os.path.join(PERIOD2_DIR, 'daily_asset_growth.png')}에 저장되었습니다.")

def plot_asset_growth_without_periods(df, initial_capital=100000):
    """기간 구분이 없는 트레이더의 자산 성장 그래프 생성"""
    plt.figure(figsize=(15, 10))
    
    # 자산 가치 계산
    if initial_capital is not None:
        asset_values = df['Cumulative_PnL'] + initial_capital
        title_suffix = f"(초기 원금: {initial_capital:,} USD)"
    else:
        # 원금 정보가 없는 경우 누적 PnL만 표시
        asset_values = df['Cumulative_PnL']
        title_suffix = "(원금 정보 없음)"
    
    # 자산 가치 그래프
    plt.subplot(2, 1, 1)
    plt.plot(df['Close_Time_KST'], asset_values, marker='o', linestyle='-', color='blue')
    plt.title(f'자산 가치 변화 {title_suffix}', fontsize=16)
    plt.ylabel('자산 가치 (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 일별 자산 변화
    plt.subplot(2, 1, 2)
    
    # 일별 데이터 준비
    df['Close_Date'] = df['Close_Time_KST'].dt.date
    daily_pnl = df.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
    daily_pnl['Close_Date'] = pd.to_datetime(daily_pnl['Close_Date'])
    
    # 일별 PnL 그래프
    plt.bar(daily_pnl['Close_Date'], daily_pnl['PnL_Numeric'], color='green')
    plt.title('일별 PnL', fontsize=16)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OVERALL_DIR, 'asset_growth.png'), dpi=300, bbox_inches='tight')
    print(f"자산 성장 그래프가 {os.path.join(OVERALL_DIR, 'asset_growth.png')}에 저장되었습니다.")


def generate_asset_growth_charts(trader_id=None, csv_file=None, output_dir=None):
    """트레이더 ID와 CSV 파일을 기반으로 자산 성장 그래프 생성"""
    from trader_config import get_trader, get_all_trader_ids
    
    # 트레이더 선택 또는 확인
    if trader_id is None:
        # 기본 트레이더 사용
        trader_id = "hummusXBT"
    
    # 트레이더 정보 가져오기
    trader = get_trader(trader_id)
    trader_name = trader['name']
    
    # 트레이더 설정에 따라 분석 방식 결정
    use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
    
    # 결과 폴더 설정
    global RESULT_DIR, PERIOD1_DIR, PERIOD2_DIR, OVERALL_DIR
    
    # 출력 디렉토리가 지정된 경우 사용
    if output_dir:
        RESULT_DIR = output_dir
    else:
        RESULT_DIR = f"analysis_results/{trader_id}"
    
    OVERALL_DIR = os.path.join(RESULT_DIR, "overall")
    
    # 기본 폴더 생성
    for directory in [RESULT_DIR, OVERALL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"폴더 생성: {directory}")
    
    # 기간 구분이 있는 트레이더인 경우에만 period1, period2 디렉토리 생성
    if use_periods:
        PERIOD1_DIR = os.path.join(RESULT_DIR, "period1")
        PERIOD2_DIR = os.path.join(RESULT_DIR, "period2")
        
        for directory in [PERIOD1_DIR, PERIOD2_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"폴더 생성: {directory}")
    else:
        # 기간 구분이 없는 트레이더인 경우 None으로 설정
        PERIOD1_DIR = None
        PERIOD2_DIR = None
    
    # CSV 파일 경로 가져오기
    if csv_file is None:
        # 최신 CSV 파일 찾기
        trader_data_dir = f"trader_data/{trader_id}"
        if not os.path.exists(trader_data_dir):
            print(f"오류: {trader_id} 트레이더의 데이터 디렉토리를 찾을 수 없습니다.")
            return
        
        # 디렉토리에서 CSV 파일 찾기
        csv_files = [f for f in os.listdir(trader_data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"오류: {trader_id} 트레이더의 CSV 파일을 찾을 수 없습니다.")
            return
        
        # 가장 최근 파일 선택
        csv_files.sort(reverse=True)
        csv_file = os.path.join(trader_data_dir, csv_files[0])
    
    print(f"\n===== {trader_name} 트레이더 자산 성장 그래프 생성 시작 =====")
    print(f"분석할 파일: {csv_file}")
    
    # 트레이더 설정에 따라 분석 방식 결정
    use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
    
    try:
        if use_periods:
            # 기간 구분이 있는 경우 (hummusXBT 같은 트레이더)
            initial_capital_period1 = trader.get('initial_capital_period1', INITIAL_CAPITAL_PERIOD1)
            initial_capital_period2 = trader.get('initial_capital_period2', INITIAL_CAPITAL_PERIOD2)
            period_split_date_str = trader.get('period_split_date', '2024-06-15')
            
            # 기간 구분 날짜 설정
            period_split_date = datetime.strptime(period_split_date_str, '%Y-%m-%d')
            
            # PnL 분석 (기간별)
            df, df_period1, df_period2 = analyze_pnl_by_periods(csv_file, period_split_date, trader_id)
            
            # 자산 성장 그래프 생성
            plot_asset_growth_by_periods(df, df_period1, df_period2)
            
            # 일별 자산 성장 그래프 생성
            plot_daily_asset_growth(df_period1, df_period2)
        else:
            # 기간 구분이 없는 경우 (TRADERT22 같은 트레이더)
            initial_capital = trader.get('initial_capital', None)  # 기본 원금 설정
            
            # 전체 기간 분석
            df = analyze_pnl_without_periods(csv_file, trader_id)
            
            # 자산 성장 그래프 생성
            plot_asset_growth_without_periods(df, initial_capital)
        
        print("\n===== 자산 성장 그래프 생성 완료 =====")
        return True
    except Exception as e:
        print(f"오류: 자산 성장 그래프 생성 중 문제가 발생했습니다.")
        print(f"오류 메시지: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="트레이더 자산 성장 그래프 생성 도구")
    parser.add_argument("-t", "--trader", help="분석할 트레이더 ID", type=str)
    parser.add_argument("-f", "--file", help="분석할 CSV 파일 경로", type=str)
    parser.add_argument("-o", "--output-dir", help="결과를 저장할 디렉토리", type=str)
    
    args = parser.parse_args()
    
    # 자산 성장 그래프 생성
    success = generate_asset_growth_charts(args.trader, args.file, args.output_dir)
    
    # 실행 결과에 따라 종료 코드 설정
    sys.exit(0 if success else 1)
