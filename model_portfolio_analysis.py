#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
from datetime import datetime, timedelta
from math import sqrt
from overlap_analysis.config import model_initial_capital, trader_weight

# 유틸리티 함수 import
from utils.portfolio_utils import (
    save_summary_data,
    save_portfolio_data,
    plot_capital_graph,
    plot_mdd,
    plot_mdd_heatmap,
    save_drawdown_data,
    save_monthly_returns
)
from utils.markdown_parser import parse_position_markdown
from utils.portfolio_metrics import (
    calculate_mdd,
    calculate_monthly_mdd,
    calculate_annualized_return,
    calculate_daily_returns,
    calculate_sharpe_ratio,
    calculate_calmar_ratio
)

# 커맨드 라인 인자 파싱
parser = argparse.ArgumentParser(description='모델 포트폴리오 분석에 슬리피지 적용')
parser.add_argument('--slippage', type=float, default=0.0, help='슬리피지 비율 (%, 예: 0.05는 0.05%)')
args = parser.parse_args()

# 슬리피지 비율 설정
slippage_rate = args.slippage / 100  # 퍼센트에서 소수점으로 변환 (0.05% -> 0.0005)

# 분석할 마크다운 파일 경로
md_file_path = os.path.join('overlap_analysis', 'output', 'position_overlap_analysis.md')

# Output directory for results
output_dir = os.path.join('overlap_analysis', 'output', 'model_portfolio')
if slippage_rate > 0:
    output_dir = os.path.join('overlap_analysis', 'output', f'model_portfolio_slippage_{args.slippage:.2f}')
os.makedirs(output_dir, exist_ok=True)

# 슬리피지 정보 출력
if slippage_rate > 0:
    print(f"\n적용 슬리피지: {args.slippage:.2f}% (거래량 * {slippage_rate:.6f})")

# Initial model portfolio capital
model_capital = model_initial_capital
print(f"Initial model portfolio capital: {model_capital:,.2f} USD")

# 마크다운 파일 파싱
periods = parse_position_markdown(md_file_path)

# 모델 포트폴리오 분석
print("\n모델 포트폴리오 자본 변화 분석:")
print("-" * 60)
print("| 구간 | 시작 자본 | 종료 자본 | 변화 | 변화율 |")
print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|")

print("\n| 구간 | 구간 시작 자본 | 구간 종료 자본 | 변화량 | 변화율 |")
print("|" + "-" * 6 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

# 성능 지표 계산 함수들은 utils.portfolio_metrics 모듈로 이동

# 결과 데이터 준비
results_data = []
timestamps = []

# 시간 정보와 자본 이력 저장
model_capital = model_initial_capital

# 한 번만 계산하여 결과를 저장하고 출력하기
for i, period in enumerate(periods):
    period_start_capital = model_capital
    period_contribution = 0
    period_number = i + 1
    
    # 트레이더별 기여도 계산
    contributions = {}
    trader_processed = set()  # 이미 처리한 트레이더 추적
    
    # 포지션 종료에 따른 수익/손실이 있는 경우 계산
    # 해당 구간에 종료된 포지션만 필터링
    closed_positions = []
    for pos in period['positions']:
        # 포지션 클로즈 시간 파싱
        try:
            close_time_str = pos.get('close', '')
            if not close_time_str:
                continue
                
            # 클로즈 시간이 문자열이 아닌 경우 무시
            if not isinstance(close_time_str, str):
                continue
                
            # 클로즈 시간을 datetime 객체로 변환
            close_time = datetime.strptime(close_time_str, '%Y-%m-%d %H:%M:%S')
            
            # 해당 구간의 시작과 종료 시간
            period_start = period.get('start_time')
            period_end = period.get('end_time')
            
            # 해당 구간 내에 포지션이 종료된 경우만 포함
            # 종료 시간이 구간 종료와 일치하거나 그 이전이며 구간 시작 이후인 경우
            if period_start <= close_time <= period_end:
                closed_positions.append(pos)
        except Exception as e:
            print(f"Error parsing position close time: {e}")
    
    # 각 트레이더당 해당 구간에서 종료된 포지션들의 실현 수익 계산
    trader_pnl = {}
    for pos in closed_positions:
        trader = pos['trader']
        realized_pnl = pos.get('realized_pnl', 0)
        
        # 슬리피지 적용 (volume이 있는 경우에만)
        if slippage_rate > 0 and 'volume' in pos and pos['volume'] is not None:
            volume = float(pos['volume'])
            slippage_cost = volume * slippage_rate
            realized_pnl -= slippage_cost
        
        if trader not in trader_pnl:
            trader_pnl[trader] = 0
        
        trader_pnl[trader] += realized_pnl
        
    # 트레이더별 기여도 계산
    for trader, pnl in trader_pnl.items():
        
        if trader in period['weights'] and trader in period['balance_after']:
            # 가중치
            weight = period['weights'][trader]
            
            # 트레이더 종료 후 자산
            balance_after = period['balance_after'][trader]
            
            # 해당 구간에서 실현된 수익/손실 (이미 계산된 값 사용)
            trader_realized_pnl = pnl
            
            # 트레이더에게 해당 구간에 종료된 포지션이 있을 경우에만 계산 진행
            if trader_realized_pnl != 0:
                # 포지션 종료 전 자산 계산 (종료 후 자산 - 실현 수익)
                balance_before = balance_after - trader_realized_pnl
                
                # balance_before가 0 또는 음수면 오류 발생 가능성이 있음
                if balance_before <= 0:
                    print(f"Warning: Invalid balance_before for {trader}: {balance_before}. Using balance_after instead.")
                    balance_before = balance_after
                    if balance_before <= 0:
                        print(f"Warning: Still invalid balance. Skipping {trader}.")
                        continue
                
                # 자본 할당 계산 (model_capital * weight / balance_before)
                # 모델이 트레이더 자본의 링문에 따라 수익/손실 계산 조정
                allocation = model_capital * weight / balance_before
                
                # 기여도 계산 (할당 비율 * 실현 수익)
                trader_contribution = allocation * trader_realized_pnl
                
                period_contribution += trader_contribution
                
                # 트레이더별 수익률 계산 
                trader_pnl_pct = (trader_realized_pnl / balance_before) * 100
                
                # 기여도 정보 저장
                contributions[trader] = {
                    'weight': weight * 100,  # 백분율로 표시
                    'pnl_pct': trader_pnl_pct,
                    'contribution': trader_contribution,
                    'contribution_pct': (trader_contribution / period_start_capital) * 100 if period_start_capital > 0 else 0
                }
    
    # 모델 포트폴리오 자본 업데이트
    model_capital += period_contribution
    
    # 결과 출력
    change = model_capital - period_start_capital
    change_pct = (change / period_start_capital) * 100 if period_start_capital > 0 else 0
    
    print(f"| {period_number} | {period_start_capital:,.2f} | {model_capital:,.2f} | {change:+,.2f} | {change_pct:+.2f}% |")
    
    # 트레이더별 기여도 출력
    if contributions:
        print(f"\n구간 {period_number} 트레이더별 기여도:")
        print("| 트레이더 | 가중치(%) | 수익률(%) | 기여도(USD) | 기여율(%) |")
        print("|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 15 + "|" + "-" * 12 + "|")
        
        for trader, data in contributions.items():
            print(f"| {trader} | {data['weight']:.2f} | {data['pnl_pct']:+.2f} | {data['contribution']:+,.2f} | {data['contribution_pct']:+.2f} |")
        
        print("")
    
    # 결과 데이터 저장
    # 시간 정보가 있는 경우에만 추가
    if 'end_time' in period and period['end_time'] is not None:
        results_data.append({
            'period': period_number,
            'time': period['end_time'],
            'start_capital': period_start_capital,
            'end_capital': model_capital,
            'change': change,
            'change_pct': change_pct
        })
        timestamps.append(period['end_time'])

print("-" * 60)
print(f"Final model portfolio capital: {model_capital:,.2f} USD")
print(f"Total return: {((model_capital / model_initial_capital) - 1) * 100:+.2f}%")

# 데이터프레임 생성
df = pd.DataFrame(results_data)

# 기본 성과 지표 계산
# MDD 계산
mdd, mdd_time, peak_time, peak_value, valley_value = calculate_mdd(df)

# 연환산 수익률 계산
annualized_ret = calculate_annualized_return(df)

# 일별 수익률 계산
daily_returns = calculate_daily_returns(df)

# 샤프 비율 계산
sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02)

# 칼마 비율 계산
calmar = calculate_calmar_ratio(annualized_ret, mdd)

# 결과 출력
print("-" * 60)
print("포트폴리오 성과 요약:")
print(f"총 수익률: {((model_capital / model_initial_capital) - 1) * 100:+.2f}%")
print(f"연환산 수익률: {annualized_ret:+.2f}%")
print(f"표준편차: {daily_returns['daily_return'].std() * 100:.2f}%")
print(f"샤프 비율: {sharpe:.2f}")
print(f"칼마 비율: {calmar:.2f}")

print("-" * 60)
print("Maximum Drawdown (최대 낭폭) 분석:")
print(f"MDD: {mdd:.2f}%")
print(f"피크 시점: {peak_time} (${peak_value:,.2f})")
print(f"MDD 발생 시점: {mdd_time} (${valley_value:,.2f})")
print(f"낭폭 기간: {(pd.to_datetime(mdd_time) - pd.to_datetime(peak_time)).days} 일")

# 월별 MDD 계산
monthly_mdd = calculate_monthly_mdd(df)
print("-" * 60)
print("월별 MDD:")
for _, row in monthly_mdd.iterrows():
    print(f"{row['year_month']}: {row['mdd']:.2f}%")

# 요약 파일 저장
summary_path = os.path.join(output_dir, 'model_portfolio_summary.json')
summary_data = {
    'total_return_pct': ((model_capital / model_initial_capital) - 1) * 100,
    'annualized_return_pct': annualized_ret,
    'std_dev_pct': daily_returns['daily_return'].std() * 100,
    'sharpe_ratio': sharpe,
    'calmar_ratio': calmar,
    'max_drawdown_pct': mdd,
    'max_drawdown_duration_days': (pd.to_datetime(mdd_time) - pd.to_datetime(peak_time)).days,
    'peak_value': peak_value,
    'peak_time': peak_time,
    'valley_value': valley_value,
    'valley_time': mdd_time,
    'initial_capital': model_initial_capital,
    'final_capital': model_capital,
    'trading_days': len(daily_returns),
    'start_date': df['time'].iloc[0],
    'end_date': df['time'].iloc[-1],
    'period_days': (pd.to_datetime(df['time'].iloc[-1]) - pd.to_datetime(df['time'].iloc[0])).days
}

# 모델 포트폴리오 요약 데이터 저장
summary_data['total_return_pct'] = ((model_capital / model_initial_capital) - 1) * 100
summary_data['annualized_return_pct'] = annualized_ret
summary_data['std_dev_pct'] = daily_returns['daily_return'].std() * 100
summary_data['sharpe_ratio'] = sharpe
save_summary_data(summary_data, output_dir)

# 포트폴리오 자본 데이터 저장
save_portfolio_data(df, results_data, output_dir)

# 자본 변화 그래프 생성 및 저장
plot_capital_graph(df, output_dir)

# MDD 그래프 생성 및 저장
plot_mdd(df, os.path.join(output_dir, 'model_portfolio_drawdown.png'))
print(f"Drawdown graph saved: {os.path.join(output_dir, 'model_portfolio_drawdown.png')}")

# MDD 히트맵 생성 및 저장
plot_mdd_heatmap(df, os.path.join(output_dir, 'model_portfolio_heatmap.png'))
print(f"Drawdown heatmap saved: {os.path.join(output_dir, 'model_portfolio_heatmap.png')}")

# 드로다운 데이터 저장
save_drawdown_data(df, monthly_mdd, output_dir)

# 월별 수익률 데이터 및 그래프 저장
save_monthly_returns(df, output_dir)
