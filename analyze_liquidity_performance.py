#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_data(trader_name):
    """Load trader's trading data and liquidity analysis results"""
    # Load trading data
    trade_data_path = f"analysis_results/{trader_name}/overall/analyzed_data.csv"
    trade_df = pd.read_csv(trade_data_path)
    
    # Load liquidity analysis results
    liquidity_path = f"analysis_results/{trader_name}/liquidity/{trader_name}_liquidity_analysis.csv"
    liquidity_df = pd.read_csv(liquidity_path)
    
    return trade_df, liquidity_df

def merge_data(trade_df, liquidity_df):
    """Merge trading data with liquidity analysis results"""
    # Map liquidity information based on symbols
    symbol_liquidity_map = liquidity_df.set_index('symbol')['liquidity_status'].to_dict()
    
    # Add liquidity status to trading data
    trade_df['liquidity_status'] = trade_df['Symbol'].map(symbol_liquidity_map)
    
    # Handle missing values as 'unknown'
    trade_df['liquidity_status'] = trade_df['liquidity_status'].fillna('unknown')
    
    return trade_df

def analyze_performance_by_liquidity(trade_df):
    """유동성 상태별 성과 분석"""
    # 수익률 전처리
    trade_df['pnl_percent'] = trade_df['Realized PnL %'].str.rstrip(' %').astype(float)
    
    # 유동성 상태별 그룹화
    grouped = trade_df.groupby('liquidity_status')
    
    # 유동성 상태별 기본 통계 계산
    stats_dict = {}
    
    for status, group in grouped:
        win_rate = (group['PnL_Sign'] > 0).mean() * 100
        roi = group['PnL_Numeric'].sum() / group['Size_USDT_Abs'].sum() * 100 if group['Size_USDT_Abs'].sum() > 0 else 0
        
        stats_dict[status] = {
            '거래 횟수': len(group),
            '총 거래량(USDT)': group['Size_USDT_Abs'].sum(),
            '평균 포지션 크기(USDT)': group['Size_USDT_Abs'].mean(),
            '최대 포지션 크기(USDT)': group['Size_USDT_Abs'].max(),
            '총 수익(USDT)': group['PnL_Numeric'].sum(),
            '평균 수익률(%)': group['pnl_percent'].mean(),
            '승률(%)': win_rate,
            'ROI(%)': roi
        }
    
    # 데이터프레임으로 변환
    stats = pd.DataFrame.from_dict(stats_dict, orient='index')
    
    return stats

def analyze_position_size_distribution(trade_df):
    """유동성 상태별 포지션 크기 분포 분석"""
    # 포지션 크기 구간 설정
    bins = [0, 10000, 50000, 100000, 500000, 1000000, float('inf')]
    labels = ['~1만', '1만~5만', '5만~10만', '10만~50만', '50만~100만', '100만 이상']
    
    # 포지션 크기 구간 분류
    trade_df['position_size_category'] = pd.cut(trade_df['Size_USDT_Abs'], bins=bins, labels=labels)
    
    # 유동성 상태와 포지션 크기 구간별 거래 횟수
    position_size_dist = pd.crosstab(
        trade_df['liquidity_status'], 
        trade_df['position_size_category'],
        normalize='index'
    ) * 100  # 백분율로 변환
    
    return position_size_dist

def analyze_pnl_by_position_size(trade_df):
    """포지션 크기별 수익 분석"""
    # 수익률 전처리 (이미 처리되었을 수 있지만 확실히 하기 위해)
    if 'pnl_percent' not in trade_df.columns:
        trade_df['pnl_percent'] = trade_df['Realized PnL %'].str.rstrip(' %').astype(float)
    
    # 포지션 크기 구간 설정
    bins = [0, 10000, 50000, 100000, 500000, 1000000, float('inf')]
    labels = ['~1만', '1만~5만', '5만~10만', '10만~50만', '50만~100만', '100만 이상']
    
    # 포지션 크기 구간 분류
    trade_df['position_size_category'] = pd.cut(trade_df['Size_USDT_Abs'], bins=bins, labels=labels)
    
    # 유동성 상태와 포지션 크기별 그룹화
    grouped = trade_df.groupby(['liquidity_status', 'position_size_category'])
    
    # 결과 저장할 딕셔너리
    result_dict = {}
    
    # 각 그룹별로 통계 계산
    for (status, size_cat), group in grouped:
        roi = group['PnL_Numeric'].sum() / group['Size_USDT_Abs'].sum() * 100 if group['Size_USDT_Abs'].sum() > 0 else 0
        
        if (status, size_cat) not in result_dict:
            result_dict[(status, size_cat)] = {}
            
        result_dict[(status, size_cat)] = {
            '거래 횟수': len(group),
            '총 수익(USDT)': group['PnL_Numeric'].sum(),
            '평균 수익률(%)': group['pnl_percent'].mean(),
            'ROI(%)': roi
        }
    
    # 데이터프레임으로 변환
    pnl_by_size = pd.DataFrame.from_dict(result_dict, orient='index')
    pnl_by_size.index = pd.MultiIndex.from_tuples(pnl_by_size.index, names=['liquidity_status', 'position_size_category'])
    pnl_by_size = pnl_by_size.sort_index()
    
    return pnl_by_size

def analyze_pnl_distribution(trade_df):
    """유동성 상태별 수익률 분포 분석"""
    # 수익률 전처리 (이미 처리되었을 수 있지만 확실히 하기 위해)
    if 'pnl_percent' not in trade_df.columns:
        trade_df['pnl_percent'] = trade_df['Realized PnL %'].str.rstrip(' %').astype(float)
    
    # 수익률 구간 설정
    bins = [-float('inf'), -10, -5, 0, 5, 10, float('inf')]
    labels = ['<-10%', '-10%~-5%', '-5%~0%', '0%~5%', '5%~10%', '>10%']
    
    # 수익률 구간 분류
    trade_df['pnl_percent_category'] = pd.cut(
        trade_df['pnl_percent'], 
        bins=bins, 
        labels=labels
    )
    
    # 유동성 상태와 수익률 구간별 거래 횟수
    pnl_dist = pd.crosstab(
        trade_df['liquidity_status'], 
        trade_df['pnl_percent_category'],
        normalize='index'
    ) * 100  # 백분율로 변환
    
    return pnl_dist

def analyze_symbols_by_performance(trade_df, liquidity_df, top_n=10):
    """유동성 상태별 상위/하위 성과 심볼 분석"""
    # 수익률 전처리 (이미 처리되었을 수 있지만 확실히 하기 위해)
    if 'pnl_percent' not in trade_df.columns:
        trade_df['pnl_percent'] = trade_df['Realized PnL %'].str.rstrip(' %').astype(float)
    
    # 심볼별 성과 집계
    symbol_performance = trade_df.groupby('Symbol').agg({
        'PnL_Numeric': 'sum',
        'Size_USDT_Abs': 'sum',
        'pnl_percent': 'mean'
    }).reset_index()
    
    # ROI 계산
    symbol_performance['ROI(%)'] = symbol_performance['PnL_Numeric'] / symbol_performance['Size_USDT_Abs'] * 100
    
    # 유동성 상태 추가
    symbol_liquidity_map = liquidity_df.set_index('symbol')['liquidity_status'].to_dict()
    symbol_performance['liquidity_status'] = symbol_performance['Symbol'].map(symbol_liquidity_map)
    
    # 유동성 상태별 그룹화
    grouped = symbol_performance.groupby('liquidity_status')
    
    # 상위 성과 심볼
    top_performers = {}
    bottom_performers = {}
    
    for status, group in grouped:
        # ROI 기준 상위 심볼
        top_performers[status] = group.nlargest(top_n, 'ROI(%)')
        
        # ROI 기준 하위 심볼
        bottom_performers[status] = group.nsmallest(top_n, 'ROI(%)')
    
    return top_performers, bottom_performers

def create_visualizations(trade_df, output_dir, trader_name):
    """Visualize analysis results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Trade distribution by liquidity status
    plt.figure(figsize=(10, 6))
    trade_counts = trade_df['liquidity_status'].value_counts()
    trade_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title(f'{trader_name} - Trade Distribution by Liquidity Status')
    plt.ylabel('')
    plt.savefig(os.path.join(output_dir, f'{trader_name}_liquidity_trade_counts.png'), dpi=300, bbox_inches='tight')
    
    # Profit rate distribution by liquidity status
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='liquidity_status', y='pnl_percent', data=trade_df)
    plt.title(f'{trader_name} - Profit Rate Distribution by Liquidity Status')
    plt.xlabel('Liquidity Status')
    plt.ylabel('Profit Rate (%)')
    plt.savefig(os.path.join(output_dir, f'{trader_name}_liquidity_pnl_distribution.png'), dpi=300, bbox_inches='tight')
    
    # Position size distribution by liquidity status
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='liquidity_status', y='Size_USDT_Abs', data=trade_df)
    plt.title(f'{trader_name} - Position Size Distribution by Liquidity Status')
    plt.xlabel('Liquidity Status')
    plt.ylabel('Position Size (USDT)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, f'{trader_name}_liquidity_position_size.png'), dpi=300, bbox_inches='tight')
    
    # ROI comparison by liquidity status
    plt.figure(figsize=(10, 6))
    liquidity_groups = trade_df.groupby('liquidity_status')
    roi_by_liquidity = liquidity_groups['PnL_Numeric'].sum() / liquidity_groups['Size_USDT_Abs'].sum() * 100
    roi_by_liquidity.plot(kind='bar', color=sns.color_palette('pastel'))
    plt.title(f'{trader_name} - ROI by Liquidity Status')
    plt.xlabel('Liquidity Status')
    plt.ylabel('ROI (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'{trader_name}_liquidity_roi.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')

def analyze_trader(trader_name):
    """Analyze trader's performance by liquidity status"""
    print(f"\n===== {trader_name} Trader's Performance Analysis by Liquidity Status =====")
    
    # Load data
    trade_df, liquidity_df = load_data(trader_name)
    
    # Merge data
    merged_df = merge_data(trade_df, liquidity_df)
    
    # Analyze performance by liquidity status
    performance_stats = analyze_performance_by_liquidity(merged_df)
    print("\n[Basic Performance Statistics by Liquidity Status]")
    print(performance_stats)
    
    # Position size distribution analysis
    position_size_dist = analyze_position_size_distribution(merged_df)
    print("\n[Position Size Distribution by Liquidity Status (%)]")
    print(position_size_dist)
    
    # Profit analysis by position size
    pnl_by_size = analyze_pnl_by_position_size(merged_df)
    print("\n[Profit Analysis by Liquidity Status and Position Size]")
    print(pnl_by_size)
    
    # Profit rate distribution analysis
    pnl_dist = analyze_pnl_distribution(merged_df)
    print("\n[Profit Rate Distribution by Liquidity Status (%)]")
    print(pnl_dist)
    
    # Top/bottom performing symbols analysis
    top_performers, bottom_performers = analyze_symbols_by_performance(merged_df, liquidity_df)
    
    print("\n[Top 5 Performing Symbols with Low Liquidity]")
    if 'low' in top_performers:
        print(top_performers['low'][['Symbol', 'PnL_Numeric', 'Size_USDT_Abs', 'ROI(%)', 'pnl_percent']].head())
    else:
        print("No symbols with low liquidity.")
    
    print("\n[Bottom 5 Performing Symbols with Low Liquidity]")
    if 'low' in bottom_performers:
        print(bottom_performers['low'][['Symbol', 'PnL_Numeric', 'Size_USDT_Abs', 'ROI(%)', 'pnl_percent']].head())
    else:
        print("No symbols with low liquidity.")
    
    print("\n[Top 5 Performing Symbols with Normal Liquidity]")
    if 'normal' in top_performers:
        print(top_performers['normal'][['Symbol', 'PnL_Numeric', 'Size_USDT_Abs', 'ROI(%)', 'pnl_percent']].head())
    else:
        print("No symbols with normal liquidity.")
    
    # Save results
    output_dir = f"analysis_results/{trader_name}/liquidity/performance"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save basic statistics
    performance_stats.to_csv(os.path.join(output_dir, f"{trader_name}_liquidity_performance_stats.csv"))
    position_size_dist.to_csv(os.path.join(output_dir, f"{trader_name}_position_size_distribution.csv"))
    pnl_by_size.to_csv(os.path.join(output_dir, f"{trader_name}_pnl_by_position_size.csv"))
    pnl_dist.to_csv(os.path.join(output_dir, f"{trader_name}_pnl_distribution.csv"))
    
    # Save top/bottom performing symbols
    for status, df in top_performers.items():
        df.to_csv(os.path.join(output_dir, f"{trader_name}_top_performers_{status}.csv"), index=False)
    
    for status, df in bottom_performers.items():
        df.to_csv(os.path.join(output_dir, f"{trader_name}_bottom_performers_{status}.csv"), index=False)
    
    # Visualize
    create_visualizations(merged_df, output_dir, trader_name)
    
    print(f"\n{trader_name} trader's performance analysis by liquidity status has been saved to {output_dir}.")
    
    return merged_df, performance_stats

def compare_traders(traders):
    """Compare performance by liquidity status across multiple traders"""
    if len(traders) <= 1:
        return
    
    print("\n===== Comparison of Performance by Liquidity Status Across Traders =====")
    
    # Collect performance statistics for each trader
    all_stats = {}
    for trader in traders:
        _, stats = analyze_trader(trader)
        all_stats[trader] = stats
    
    # Create comparison table
    comparison = pd.concat(all_stats, axis=0)
    
    # Save results
    output_dir = "analysis_results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    comparison.to_csv(os.path.join(output_dir, "liquidity_performance_comparison.csv"))
    
    print(f"\nComparison of performance by liquidity status across traders has been saved to {output_dir}.")
    
    # Visualization: ROI comparison by liquidity status across traders
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    roi_data = []
    for trader, stats in all_stats.items():
        for liquidity_status, row in stats.iterrows():
            roi_data.append({
                'Trader': trader,
                'Liquidity Status': liquidity_status,
                'ROI(%)': row['ROI(%)']
            })
    
    roi_df = pd.DataFrame(roi_data)
    
    # Create graph
    sns.barplot(x='Trader', y='ROI(%)', hue='Liquidity Status', data=roi_df)
    plt.title('ROI Comparison by Liquidity Status Across Traders')
    plt.xlabel('Trader')
    plt.ylabel('ROI (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "trader_liquidity_roi_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Analyze trader performance by liquidity status')
    parser.add_argument('--traders', nargs='+', default=['hummusXBT', 'Cyborg0578', 'ONLYCANDLE'],
                        help='Trader names to analyze (default: hummusXBT, Cyborg0578, ONLYCANDLE)')
    args = parser.parse_args()
    
    # Analyze each trader
    for trader in args.traders:
        analyze_trader(trader)
    
    # Compare traders
    compare_traders(args.traders)
    
    print("\nAll analyses completed.")

if __name__ == "__main__":
    main()
