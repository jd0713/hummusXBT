#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime


def save_summary_data(summary_data, output_dir):
    """
    모델 포트폴리오 요약 데이터를 JSON 및 마크다운 파일로 저장
    
    Args:
        summary_data (dict): 저장할 요약 데이터 딕셔너리
        output_dir (str): 출력 디렉토리 경로
    """
    # JSON 파일로 저장
    summary_path = os.path.join(output_dir, 'model_portfolio_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"Summary data saved: {summary_path}")
    
    # 요약 마크다운 파일 저장
    summary_md_path = os.path.join(output_dir, 'model_portfolio_summary.md')
    with open(summary_md_path, 'w') as f:
        f.write('# 모델 포트폴리오 성과 요약\n\n')
        
        f.write('## 결과 요약\n\n')
        f.write(f"* 총 수익률: {summary_data['total_return_pct']:+.2f}%\n")
        f.write(f"* 연환산 수익률: {summary_data['annualized_return_pct']:+.2f}%\n")
        f.write(f"* 표준편차: {summary_data['std_dev_pct']:.2f}%\n")
        f.write(f"* 샤프 비율: {summary_data['sharpe_ratio']:.2f}\n")
        f.write(f"* 칼마 비율: {summary_data['calmar_ratio']:.2f}\n")
        f.write(f"* MDD: {summary_data['max_drawdown_pct']:.2f}%\n")
        
        f.write('\n## 포트폴리오 정보\n\n')
        f.write(f"* 초기 자본: {summary_data['initial_capital']:,.2f} USD\n")
        f.write(f"* 최종 자본: {summary_data['final_capital']:,.2f} USD\n")
        f.write(f"* 모니터링 기간: {summary_data['start_date']} ~ {summary_data['end_date']} ({summary_data['period_days']} 일)\n")
        f.write(f"* 총 트레이딩 일수: {summary_data['trading_days']} 일\n")
    
    print(f"Summary markdown saved: {summary_md_path}")


def save_portfolio_data(df, results_data, output_dir):
    """
    포트폴리오 자본 데이터를 CSV 및 마크다운 파일로 저장
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
        results_data (list): 구간별 결과 데이터 리스트
        output_dir (str): 출력 디렉토리 경로
    """
    # CSV 파일로 저장
    csv_path = os.path.join(output_dir, 'model_portfolio_capital.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved: {csv_path}")
    
    # 마크다운 파일 생성
    md_lines = []
    md_lines.append("# 모델 포트폴리오 자본 변화 분석\n")
    md_lines.append(f"초기 자본: {df['start_capital'].iloc[0]:,.2f} USD\n")
    md_lines.append(f"최종 자본: {df['end_capital'].iloc[-1]:,.2f} USD\n")
    md_lines.append(f"총 수익률: {((df['end_capital'].iloc[-1] / df['start_capital'].iloc[0]) - 1) * 100:+.2f}%\n")
    
    md_lines.append("## 구간별 자본 변화\n")
    md_lines.append("| 구간 | 시간 | 시작 자본(USD) | 종료 자본(USD) | 변화(USD) | 변화율(%) |")
    md_lines.append("|---|---|---|---|---|---|")
    
    for data in results_data:
        time_str = data['time'].strftime('%Y-%m-%d %H:%M:%S') if data['time'] else 'N/A'
        md_lines.append(f"| {data['period']} | {time_str} | {data['start_capital']:,.2f} | {data['end_capital']:,.2f} | {data['change']:+,.2f} | {data['change_pct']:+.2f} |")
    
    md_path = os.path.join(output_dir, 'model_portfolio_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Markdown file saved: {md_path}")


def plot_capital_graph(df, output_dir):
    """
    자본 변화 그래프를 생성하고 저장
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
        output_dir (str): 출력 디렉토리 경로
    """
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(df['time']), df['end_capital'], marker='o', linestyle='-', markersize=3)
    plt.title('Model Portfolio Capital Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital (USD)')
    plt.grid(True)
    
    # Format x-axis dates - reduce crowding by using fewer ticks and a cleaner format
    ax = plt.gca()
    # Check date range to determine appropriate tick interval
    date_range = (pd.to_datetime(df['time']).max() - pd.to_datetime(df['time']).min()).days
    
    if date_range > 180:  # For long periods (>6 months)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif date_range > 60:  # For medium periods (2-6 months)
        ax.xaxis.set_major_locator(mdates.WeekLocator(interval=2))  # Every 2 weeks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:  # For shorter periods
        ax.xaxis.set_major_locator(mdates.WeekLocator(interval=1))  # Weekly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate(rotation=45)
    
    # Add minor ticks for reference without labels
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.grid(True, which='major', axis='both')
    plt.grid(True, which='minor', axis='x', alpha=0.2)
    
    plt.tight_layout()
    graph_path = os.path.join(output_dir, 'model_portfolio_capital.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Graph saved: {graph_path}")


def plot_mdd(df, output_path):
    """
    MDD(Maximum Drawdown) 그래프를 생성하고 저장
    
    Args:
        df (DataFrame): 드로다운 분석 데이터프레임
        output_path (str): 출력 파일 경로
    """
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(pd.to_datetime(df['time']), df['end_capital'], label='Portfolio Value')
    plt.plot(pd.to_datetime(df['time']), df['cumulative_max'], 'r--', label='Peak Value')
    plt.title('Portfolio Value and Peak Value')
    plt.ylabel('USD')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(pd.to_datetime(df['time']), df['drawdown'], 'r')
    plt.fill_between(pd.to_datetime(df['time']), df['drawdown'], 0, alpha=0.3, color='red')
    plt.title('Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mdd_heatmap(df, output_path):
    """
    MDD(Maximum Drawdown) 히트맵을 생성하고 저장
    
    Args:
        df (DataFrame): 드로다운 분석 데이터프레임
        output_path (str): 출력 파일 경로
    """
    # 히트맵 데이터 준비
    df_heatmap = df.copy()
    df_heatmap['date'] = pd.to_datetime(df_heatmap['time']).dt.date
    df_heatmap['year'] = pd.to_datetime(df_heatmap['date']).dt.year
    df_heatmap['month'] = pd.to_datetime(df_heatmap['date']).dt.month
    
    # 각 월별 최저 드로다운 값 계산
    monthly_min_dd = df_heatmap.groupby(['year', 'month'])['drawdown'].min().reset_index()
    
    # 히트맵 데이터 피벗
    heatmap_data = monthly_min_dd.pivot(index='month', columns='year', values='drawdown')
    
    # 월 이름으로 인덱스 변경
    month_names = {i: month_name[i] for i in range(1, 13)}
    heatmap_data.index = heatmap_data.index.map(month_names)
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f', 
                linewidths=.5, center=0, vmin=-10, vmax=0)
    plt.title('Monthly Maximum Drawdown (%) Heatmap')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_drawdown_data(df, monthly_mdd, output_dir):
    """
    드로다운 관련 데이터를 CSV 파일로 저장
    
    Args:
        df (DataFrame): 드로다운 분석 데이터프레임
        monthly_mdd (DataFrame): 월별 MDD 데이터프레임
        output_dir (str): 출력 디렉토리 경로
    """
    # MDD 데이터 저장
    mdd_df = df[['time', 'end_capital', 'cumulative_max', 'drawdown']]
    mdd_df.to_csv(os.path.join(output_dir, 'model_portfolio_drawdown.csv'), index=False)
    print(f"Drawdown data saved: {os.path.join(output_dir, 'model_portfolio_drawdown.csv')}")
    
    # 월별 MDD 데이터 저장
    monthly_mdd.to_csv(os.path.join(output_dir, 'model_portfolio_monthly_mdd.csv'), index=False)
    print(f"Monthly MDD data saved: {os.path.join(output_dir, 'model_portfolio_monthly_mdd.csv')}")


def save_monthly_returns(df, output_dir):
    """
    월별 수익률 데이터 및 그래프를 저장
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
        output_dir (str): 출력 디렉토리 경로
    """
    # 월별 수익률 계산
    df['month'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m')
    monthly_returns = df.groupby('month').apply(
        lambda x: (x['end_capital'].iloc[-1] / x['start_capital'].iloc[0] - 1) * 100
    ).reset_index()
    monthly_returns.columns = ['month', 'monthly_return']
    
    # 그래프 생성
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_returns['month'], monthly_returns['monthly_return'])
    plt.title('Model Portfolio Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Monthly Return (%)')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    
    # 그래프 저장
    monthly_graph_path = os.path.join(output_dir, 'model_portfolio_monthly_returns.png')
    plt.tight_layout()
    plt.savefig(monthly_graph_path, dpi=300)
    plt.close()
    print(f"Monthly returns graph saved: {monthly_graph_path}")
    
    # CSV 저장
    monthly_csv_path = os.path.join(output_dir, 'model_portfolio_monthly_returns.csv')
    monthly_returns.to_csv(monthly_csv_path, index=False)
    print(f"Monthly returns CSV saved: {monthly_csv_path}")
