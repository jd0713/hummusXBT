#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import os
from trader_analysis_part1 import *

def plot_pnl_by_periods(df, df_period1, df_period2):
    """기간별 PnL 그래프 생성"""
    # 전체 기간 그래프
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(df['Close_Time_KST'], df['Cumulative_PnL'], marker='o', linestyle='-', color='blue')
    plt.axvline(x=convert_to_kst(PERIOD_SPLIT_DATE), color='red', linestyle='--', label='원금 변경 시점')
    plt.title('전체 기간 누적 PnL (한국 시간 기준)', fontsize=16)
    plt.ylabel('누적 PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 기간 1 그래프
    if not df_period1.empty:
        plt.subplot(3, 1, 2)
        plt.plot(df_period1['Close_Time_KST'], df_period1['Period_Cumulative_PnL'], 
                marker='o', linestyle='-', color='green')
        plt.title(f'기간 1 누적 PnL (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        plt.ylabel('누적 PnL (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    # 기간 2 그래프
    if not df_period2.empty:
        plt.subplot(3, 1, 3)
        plt.plot(df_period2['Close_Time_KST'], df_period2['Period_Cumulative_PnL'], 
                marker='o', linestyle='-', color='purple')
        plt.title(f'기간 2 누적 PnL (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        plt.ylabel('누적 PnL (USDT)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OVERALL_DIR, 'pnl_analysis_by_periods.png'), dpi=300, bbox_inches='tight')
    print(f"기간별 PnL 그래프가 {os.path.join(OVERALL_DIR, 'pnl_analysis_by_periods.png')}에 저장되었습니다.")

def plot_drawdown_by_periods(df_period1, df_period2):
    """기간별 낙폭(Drawdown) 그래프 생성"""
    plt.figure(figsize=(15, 10))
    
    # 기간 1 낙폭 그래프
    if not df_period1.empty:
        # 일별 데이터 준비
        df_period1_copy = df_period1.copy()
        df_period1_copy.loc[:, 'Close_Date'] = df_period1_copy['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
        daily_pnl1 = df_period1_copy.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
        daily_pnl1['Daily_Return'] = daily_pnl1['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD1
        
        # 자산 가치 계산 (초기 자산 + 누적 PnL)
        daily_pnl1['Cumulative_PnL'] = daily_pnl1['PnL_Numeric'].cumsum()
        daily_pnl1['Asset_Value'] = daily_pnl1['Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD1
        
        # 올바른 MDD 계산
        peak1 = daily_pnl1['Asset_Value'].cummax()
        drawdown1 = (daily_pnl1['Asset_Value'] - peak1) / peak1
        
        # 그래프 그리기
        plt.subplot(2, 1, 1)
        plt.plot(daily_pnl1['Close_Date'], drawdown1 * 100, color='red')
        plt.fill_between(daily_pnl1['Close_Date'], drawdown1 * 100, 0, color='red', alpha=0.3)
        plt.title(f'기간 1 낙폭 (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        plt.ylabel('낙폭 (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # y축 반전 (낙폭을 아래로 표시)
        plt.xticks(rotation=45)
        
        # 결과 저장 (기간 1)
        plt.savefig(os.path.join(PERIOD1_DIR, 'drawdown_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"기간 1 낙폭 그래프가 {os.path.join(PERIOD1_DIR, 'drawdown_analysis.png')}에 저장되었습니다.")
    
    # 기간 2 낙폭 그래프
    if not df_period2.empty:
        # 일별 데이터 준비
        df_period2_copy = df_period2.copy()
        df_period2_copy.loc[:, 'Close_Date'] = df_period2_copy['Close_Time_UTC'].apply(lambda x: x.date() if x is not None else None)
        daily_pnl2 = df_period2_copy.groupby('Close_Date')['PnL_Numeric'].sum().reset_index()
        daily_pnl2['Daily_Return'] = daily_pnl2['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD2
        
        # 자산 가치 계산 (초기 자산 + 누적 PnL)
        daily_pnl2['Cumulative_PnL'] = daily_pnl2['PnL_Numeric'].cumsum()
        daily_pnl2['Asset_Value'] = daily_pnl2['Cumulative_PnL'] + INITIAL_CAPITAL_PERIOD2
        
        # 올바른 MDD 계산
        peak2 = daily_pnl2['Asset_Value'].cummax()
        drawdown2 = (daily_pnl2['Asset_Value'] - peak2) / peak2
        
        # 그래프 그리기
        plt.subplot(2, 1, 2)
        plt.plot(daily_pnl2['Close_Date'], drawdown2 * 100, color='red')
        plt.fill_between(daily_pnl2['Close_Date'], drawdown2 * 100, 0, color='red', alpha=0.3)
        plt.title(f'기간 2 낙폭 (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        plt.ylabel('낙폭 (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # y축 반전 (낙폭을 아래로 표시)
        plt.xticks(rotation=45)
        
        # 결과 저장 (기간 2)
        plt.savefig(os.path.join(PERIOD2_DIR, 'drawdown_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"기간 2 낙폭 그래프가 {os.path.join(PERIOD2_DIR, 'drawdown_analysis.png')}에 저장되었습니다.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OVERALL_DIR, 'drawdown_by_periods.png'), dpi=300, bbox_inches='tight')
    print(f"전체 낙폭 그래프가 {os.path.join(OVERALL_DIR, 'drawdown_by_periods.png')}에 저장되었습니다.")

def plot_monthly_returns(df_period1, df_period2):
    """월별 수익률 히트맵 생성"""
    # 기간 1 월별 수익률
    if not df_period1.empty:
        plt.figure(figsize=(12, 8))
        # 날짜 컬럼 추가
        df_period1_copy = df_period1.copy()
        df_period1_copy.loc[:, 'Year'] = df_period1_copy['Close_Time_KST'].apply(lambda x: x.year if x is not None else None)
        df_period1_copy.loc[:, 'Month'] = df_period1_copy['Close_Time_KST'].apply(lambda x: x.month if x is not None else None)
        
        # 월별 수익률 계산
        monthly_returns1 = df_period1_copy.groupby(['Year', 'Month'])['PnL_Numeric'].sum().reset_index()
        monthly_returns1['Monthly_Return'] = monthly_returns1['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD1 * 100
        
        # 피봇 테이블 생성
        pivot1 = monthly_returns1.pivot(index='Year', columns='Month', values='Monthly_Return')
        
        # 히트맵 그리기
        sns.heatmap(pivot1, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title(f'기간 1 월별 수익률 (%) (원금: {INITIAL_CAPITAL_PERIOD1:,} USD)', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD1_DIR, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
        print(f"기간 1 월별 수익률 히트맵이 {os.path.join(PERIOD1_DIR, 'monthly_returns.png')}에 저장되었습니다.")
    
    # 기간 2 월별 수익률
    if not df_period2.empty:
        plt.figure(figsize=(12, 8))
        # 날짜 컬럼 추가
        df_period2_copy = df_period2.copy()
        df_period2_copy.loc[:, 'Year'] = df_period2_copy['Close_Time_KST'].apply(lambda x: x.year if x is not None else None)
        df_period2_copy.loc[:, 'Month'] = df_period2_copy['Close_Time_KST'].apply(lambda x: x.month if x is not None else None)
        
        # 월별 수익률 계산
        monthly_returns2 = df_period2_copy.groupby(['Year', 'Month'])['PnL_Numeric'].sum().reset_index()
        monthly_returns2['Monthly_Return'] = monthly_returns2['PnL_Numeric'] / INITIAL_CAPITAL_PERIOD2 * 100
        
        # 피봇 테이블 생성
        pivot2 = monthly_returns2.pivot(index='Year', columns='Month', values='Monthly_Return')
        
        # 히트맵 그리기
        sns.heatmap(pivot2, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title(f'기간 2 월별 수익률 (%) (원금: {INITIAL_CAPITAL_PERIOD2:,} USD)', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PERIOD2_DIR, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
        print(f"기간 2 월별 수익률 히트맵이 {os.path.join(PERIOD2_DIR, 'monthly_returns.png')}에 저장되었습니다.")

def plot_symbol_analysis(df_period, period_name, output_dir):
    """심볼별 분석 그래프 생성"""
    if df_period.empty:
        return
    
    # 심볼별 PnL 합계
    symbol_pnl = df_period.groupby('Symbol')['PnL_Numeric'].sum().sort_values(ascending=False)
    
    # 상위 10개 및 하위 10개 심볼 추출
    top_symbols = symbol_pnl.head(10)
    bottom_symbols = symbol_pnl.tail(10)
    
    # 그래프 그리기
    plt.figure(figsize=(15, 10))
    
    # 상위 10개 심볼
    plt.subplot(2, 1, 1)
    colors = ['green' if pnl >= 0 else 'red' for pnl in top_symbols]
    plt.bar(top_symbols.index, top_symbols.values, color=colors, alpha=0.7)
    plt.title(f'{period_name} 상위 10개 심볼 PnL', fontsize=16)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 하위 10개 심볼
    plt.subplot(2, 1, 2)
    colors = ['green' if pnl >= 0 else 'red' for pnl in bottom_symbols]
    plt.bar(bottom_symbols.index, bottom_symbols.values, color=colors, alpha=0.7)
    plt.title(f'{period_name} 하위 10개 심볼 PnL', fontsize=16)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'symbol_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name} 심볼 분석 그래프가 {os.path.join(output_dir, 'symbol_analysis.png')}에 저장되었습니다.")
    
    # 심볼별 거래 횟수 분석
    symbol_count = df_period.groupby('Symbol').size().sort_values(ascending=False)
    top_symbols_count = symbol_count.head(15)
    
    plt.figure(figsize=(15, 8))
    plt.bar(top_symbols_count.index, top_symbols_count.values, color='blue', alpha=0.7)
    plt.title(f'{period_name} 심볼별 거래 횟수 (상위 15개)', fontsize=16)
    plt.ylabel('거래 횟수', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'symbol_trade_count.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name} 심볼별 거래 횟수 그래프가 {os.path.join(output_dir, 'symbol_trade_count.png')}에 저장되었습니다.")
    
    # 심볼별 승률 분석
    symbol_win_rate = df_period.groupby('Symbol').apply(
        lambda x: (x['PnL_Numeric'] > 0).mean() * 100 if len(x) > 5 else np.nan
    ).dropna().sort_values(ascending=False)
    
    if not symbol_win_rate.empty:
        top_symbols_win_rate = symbol_win_rate.head(15)
        
        plt.figure(figsize=(15, 8))
        plt.bar(top_symbols_win_rate.index, top_symbols_win_rate.values, color='green', alpha=0.7)
        plt.title(f'{period_name} 심볼별 승률 (%) (상위 15개, 최소 5회 이상 거래)', fontsize=16)
        plt.ylabel('승률 (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'symbol_win_rate.png'), dpi=300, bbox_inches='tight')
        print(f"{period_name} 심볼별 승률 그래프가 {os.path.join(output_dir, 'symbol_win_rate.png')}에 저장되었습니다.")

def plot_direction_analysis(df_period, period_name, output_dir):
    """방향별(Long/Short) 분석 그래프 생성"""
    if df_period.empty:
        return
    
    # 방향별 PnL 합계
    direction_pnl = df_period.groupby('Direction')['PnL_Numeric'].sum().reset_index()
    
    # 방향별 거래 횟수
    direction_count = df_period.groupby('Direction').size().reset_index(name='Count')
    
    # 방향별 승률
    direction_win_rate = df_period.groupby('Direction').apply(
        lambda x: (x['PnL_Numeric'] > 0).mean() * 100
    ).reset_index(name='Win_Rate')
    
    # 그래프 그리기
    plt.figure(figsize=(15, 15))
    
    # 방향별 PnL
    plt.subplot(3, 1, 1)
    colors = ['green' if pnl >= 0 else 'red' for pnl in direction_pnl['PnL_Numeric']]
    plt.bar(direction_pnl['Direction'], direction_pnl['PnL_Numeric'], color=colors, alpha=0.7)
    plt.title(f'{period_name} 방향별 PnL', fontsize=16)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 방향별 거래 횟수
    plt.subplot(3, 1, 2)
    plt.bar(direction_count['Direction'], direction_count['Count'], color='blue', alpha=0.7)
    plt.title(f'{period_name} 방향별 거래 횟수', fontsize=16)
    plt.ylabel('거래 횟수', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 방향별 승률
    plt.subplot(3, 1, 3)
    plt.bar(direction_win_rate['Direction'], direction_win_rate['Win_Rate'], color='green', alpha=0.7)
    plt.title(f'{period_name} 방향별 승률 (%)', fontsize=16)
    plt.ylabel('승률 (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name} 방향별 분석 그래프가 {os.path.join(output_dir, 'direction_analysis.png')}에 저장되었습니다.")

def plot_trade_size_distribution(df_period, period_name, output_dir):
    """거래 크기 분포 분석"""
    if df_period.empty:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 거래 크기 히스토그램
    plt.subplot(2, 1, 1)
    sns.histplot(df_period['Size_USDT_Abs'], bins=30, kde=True)
    plt.title(f'{period_name} 거래 크기 분포', fontsize=16)
    plt.xlabel('거래 크기 (USDT)', fontsize=14)
    plt.ylabel('빈도', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 거래 크기 vs PnL 산점도
    plt.subplot(2, 1, 2)
    plt.scatter(df_period['Size_USDT_Abs'], df_period['PnL_Numeric'], 
               alpha=0.5, c=df_period['PnL_Numeric'], cmap='RdYlGn')
    plt.title(f'{period_name} 거래 크기 vs PnL', fontsize=16)
    plt.xlabel('거래 크기 (USDT)', fontsize=14)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='PnL (USDT)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trade_size_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name} 거래 크기 분석 그래프가 {os.path.join(output_dir, 'trade_size_analysis.png')}에 저장되었습니다.")

def plot_holding_period_analysis(df_period, period_name, output_dir):
    """보유 기간 분석"""
    if df_period.empty:
        return
    
    # 보유 기간 계산 (시간 단위)
    df_period_copy = df_period.copy()
    df_period_copy.loc[:, 'Holding_Period_Hours'] = df_period_copy.apply(
        lambda row: (row['Close_Time_UTC'] - row['Open_Time_UTC']).total_seconds() / 3600 
        if row['Close_Time_UTC'] is not None and row['Open_Time_UTC'] is not None else np.nan, 
        axis=1
    )
    
    # 이상치 제거 (너무 긴 보유 기간)
    df_filtered = df_period_copy[df_period_copy['Holding_Period_Hours'] <= 720]  # 30일 이하
    
    if df_filtered.empty:
        return
    
    plt.figure(figsize=(15, 15))
    
    # 보유 기간 히스토그램
    plt.subplot(3, 1, 1)
    sns.histplot(df_filtered['Holding_Period_Hours'], bins=30, kde=True)
    plt.title(f'{period_name} 보유 기간 분포 (시간)', fontsize=16)
    plt.xlabel('보유 기간 (시간)', fontsize=14)
    plt.ylabel('빈도', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 보유 기간 vs PnL 산점도
    plt.subplot(3, 1, 2)
    plt.scatter(df_filtered['Holding_Period_Hours'], df_filtered['PnL_Numeric'], 
               alpha=0.5, c=df_filtered['PnL_Numeric'], cmap='RdYlGn')
    plt.title(f'{period_name} 보유 기간 vs PnL', fontsize=16)
    plt.xlabel('보유 기간 (시간)', fontsize=14)
    plt.ylabel('PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='PnL (USDT)')
    
    # 보유 기간별 평균 수익률
    holding_period_bins = [0, 1, 6, 12, 24, 48, 72, 168, 336, 720]
    bin_labels = ['0-1시간', '1-6시간', '6-12시간', '12-24시간', '1-2일', '2-3일', '3-7일', '7-14일', '14-30일']
    
    df_filtered.loc[:, 'Holding_Period_Bin'] = pd.cut(
        df_filtered['Holding_Period_Hours'], 
        bins=holding_period_bins, 
        labels=bin_labels, 
        include_lowest=True
    )
    
    avg_pnl_by_period = df_filtered.groupby('Holding_Period_Bin')['PnL_Numeric'].mean().reset_index()
    
    plt.subplot(3, 1, 3)
    colors = ['green' if pnl >= 0 else 'red' for pnl in avg_pnl_by_period['PnL_Numeric']]
    plt.bar(avg_pnl_by_period['Holding_Period_Bin'], avg_pnl_by_period['PnL_Numeric'], color=colors, alpha=0.7)
    plt.title(f'{period_name} 보유 기간별 평균 PnL', fontsize=16)
    plt.xlabel('보유 기간', fontsize=14)
    plt.ylabel('평균 PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'holding_period_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name} 보유 기간 분석 그래프가 {os.path.join(output_dir, 'holding_period_analysis.png')}에 저장되었습니다.")

def generate_all_visualizations(df, df_period1, df_period2):
    """모든 시각화 생성"""
    # 기간별 PnL 그래프
    plot_pnl_by_periods(df, df_period1, df_period2)
    
    # 낙폭 그래프
    plot_drawdown_by_periods(df_period1, df_period2)
    
    # 월별 수익률 히트맵
    plot_monthly_returns(df_period1, df_period2)
    
    # 기간 1 추가 분석
    if not df_period1.empty:
        plot_symbol_analysis(df_period1, "기간 1", PERIOD1_DIR)
        plot_direction_analysis(df_period1, "기간 1", PERIOD1_DIR)
        plot_trade_size_distribution(df_period1, "기간 1", PERIOD1_DIR)
        plot_holding_period_analysis(df_period1, "기간 1", PERIOD1_DIR)
    
    # 기간 2 추가 분석
    if not df_period2.empty:
        plot_symbol_analysis(df_period2, "기간 2", PERIOD2_DIR)
        plot_direction_analysis(df_period2, "기간 2", PERIOD2_DIR)
        plot_trade_size_distribution(df_period2, "기간 2", PERIOD2_DIR)
        plot_holding_period_analysis(df_period2, "기간 2", PERIOD2_DIR)
    
    # 전체 기간 추가 분석
    plot_symbol_analysis(df, "전체 기간", OVERALL_DIR)
    plot_direction_analysis(df, "전체 기간", OVERALL_DIR)
    plot_trade_size_distribution(df, "전체 기간", OVERALL_DIR)
    plot_holding_period_analysis(df, "전체 기간", OVERALL_DIR)
