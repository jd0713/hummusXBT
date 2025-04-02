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

if __name__ == "__main__":
    # CSV 파일 경로
    csv_file = "closed_positions_20250403_073826.csv"
    
    # 폴더 생성 확인
    for directory in [RESULT_DIR, PERIOD1_DIR, PERIOD2_DIR, OVERALL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"폴더 생성: {directory}")
    
    print("\n===== 자산 성장 그래프 생성 시작 =====")
    
    # PnL 분석 (기간별)
    df, df_period1, df_period2 = analyze_pnl_by_periods(csv_file)
    
    # 자산 성장 그래프 생성
    plot_asset_growth_by_periods(df, df_period1, df_period2)
    
    # 일별 자산 성장 그래프 생성
    plot_daily_asset_growth(df_period1, df_period2)
    
    print("\n===== 자산 성장 그래프 생성 완료 =====")
