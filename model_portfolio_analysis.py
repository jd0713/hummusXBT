#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from calendar import month_name
from math import sqrt
from overlap_analysis.config import model_initial_capital, trader_weight

# 분석할 마크다운 파일 경로
md_file_path = os.path.join('overlap_analysis', 'output', 'position_overlap_analysis.md')

# Output directory for results
output_dir = os.path.join('overlap_analysis', 'output', 'model_portfolio')
os.makedirs(output_dir, exist_ok=True)

# Initial model portfolio capital
model_capital = model_initial_capital
print(f"Initial model portfolio capital: {model_capital:,.2f} USD")

# 마크다운 파일 읽기
with open(md_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 구간 분석 결과 파싱
periods = []
current_period = None
current_section = None

for line in lines:
    line = line.strip()
    
    # 구간 시작 - 제목 라인 분석
    if line.startswith('## ['):
        if current_period is not None:
            periods.append(current_period)
        
        # 새로운 구간 정보 초기화
        current_period = {
            'title': line,
            'traders': {},
            'positions': [],         # 포지션 정보를 저장하는 리스트
            'weights': {},          # 트레이더별 가중치
            'balance_after': {}      # 구간 종료 후 트레이더별 자산
        }
        
        # 시작 시간과 종료 시간 추출
        try:
            time_part = line.split('] ')[1].split(' | ')[0]
            start_time_str, end_time_str = time_part.split(' ~ ')
            current_period['start_time_str'] = start_time_str
            current_period['end_time_str'] = end_time_str
            current_period['start_time'] = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
            current_period['end_time'] = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error parsing period time: {e}")
        
        current_section = None
        continue
    
    # 표 헤더 식별
    if line.startswith('| 트레이더 | 심볼 | 방향 |'):
        current_section = 'positions'
        continue
    elif line.startswith('| 트레이더 | Weight(%) | 롱 오픈'):
        current_section = 'weights'
        continue
    elif line.startswith('| 트레이더 | 구간 종료 후 자산'):
        current_section = 'balance_after'
        continue
    
    # 데이터 파싱
    if current_period and current_section == 'positions' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 8:  # 거래량(USDT) 칼럼이 추가되어 최소 8개 이상의 칼럼 필요
            trader = parts[0]
            symbol = parts[1]
            direction = parts[2]
            size_str = parts[3].replace(',', '').replace('USDT', '').strip()
            volume_str = parts[4].replace(',', '').replace('USDT', '').strip()  # 거래량(USDT) 추출
            
            # 오픈 시간과 클로즈 시간 추출
            open_time_str = parts[5].strip()  # 인덱스 변경
            close_time_str = parts[6].strip()  # 인덱스 변경
            
            realized_pnl_str = parts[7].replace(',', '').replace('USDT', '').strip()  # 인덱스 변경
            realized_pnl_pct_str = parts[8].replace('%', '').strip()  # 인덱스 변경
            
            try:
                size = float(size_str)
                volume = float(volume_str)  # 거래량 변환
                realized_pnl = float(realized_pnl_str)
                realized_pnl_pct = float(realized_pnl_pct_str) / 100.0  # 백분율을 소수로 변환
                
                # 트레이더별 실현 수익 정보 추가
                position = {
                    'trader': trader,
                    'symbol': symbol,
                    'direction': direction,
                    'size': size,
                    'volume': volume,  # 거래량 추가
                    'open': open_time_str,    # 오픈 시간 저장
                    'close': close_time_str,  # 클로즈 시간 저장
                    'realized_pnl': realized_pnl,
                    'realized_pnl_pct': realized_pnl_pct
                }
                current_period['positions'].append(position)
                
            except Exception as e:
                print(f"Error parsing position for {trader}: {e}")
    
    elif current_period and current_section == 'weights' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 8:
            trader = parts[0]
            weight_str = parts[1].replace('%', '').strip()
            
            try:
                weight = float(weight_str) / 100.0  # 백분율을 소수로 변환
                current_period['weights'][trader] = weight
            except Exception as e:
                print(f"Error parsing weight for {trader}: {e}")
    
    elif current_period and current_section == 'balance_after' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 2:
            trader = parts[0]
            balance_str = parts[1].replace(',', '').replace('USDT', '').strip()
            
            try:
                balance = float(balance_str)
                current_period['balance_after'][trader] = balance
            except Exception as e:
                print(f"Error parsing balance for {trader}: {e}")

# 마지막 구간 추가
if current_period is not None:
    periods.append(current_period)

# 모델 포트폴리오 분석
print("\n모델 포트폴리오 자본 변화 분석:")
print("-" * 60)
print("| 구간 | 시작 자본 | 종료 자본 | 변화 | 변화율 |")
print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|")

print("\n| 구간 | 구간 시작 자본 | 구간 종료 자본 | 변화량 | 변화율 |")
print("|" + "-" * 6 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

# MDD(Maximum Drawdown) 계산 함수
def calculate_mdd(df):
    # 누적 최대값 계산
    df['cumulative_max'] = df['end_capital'].cummax()
    # 드로다운 계산
    df['drawdown'] = (df['end_capital'] - df['cumulative_max']) / df['cumulative_max'] * 100
    # MDD 계산 (최저 드로다운)
    mdd = df['drawdown'].min()
    # MDD 발생 시점 찾기
    mdd_idx = df['drawdown'].idxmin()
    mdd_time = df.loc[mdd_idx, 'time']
    # 최대값 도달 시점 (MDD 발생 직전의 피크)
    peak_value = df.loc[mdd_idx, 'cumulative_max']
    peak_idx = df[df['end_capital'] == peak_value].index[0]
    peak_time = df.loc[peak_idx, 'time']
    
    return mdd, mdd_time, peak_time, peak_value, df.loc[mdd_idx, 'end_capital']

# 월별 MDD 계산 함수
def calculate_monthly_mdd(df):
    df['year_month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly_mdd = df.groupby('year_month').apply(lambda x: calculate_mdd(x)[0]).reset_index()
    monthly_mdd.columns = ['year_month', 'mdd']
    return monthly_mdd

# MDD 그래프 생성 함수
def plot_mdd(df, output_path):
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

# MDD 히트맵 생성 함수
def plot_mdd_heatmap(df, output_path):
    df['year'] = pd.to_datetime(df['time']).dt.year
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['day'] = pd.to_datetime(df['time']).dt.day
    
    monthly_data = df.groupby(['year', 'month']).agg(
        start_capital = ('start_capital', 'first'),
        end_capital = ('end_capital', 'last'),
        min_dd = ('drawdown', 'min')
    ).reset_index()
    
    # 월별 수익률 계산
    monthly_data['return_pct'] = (monthly_data['end_capital'] / monthly_data['start_capital'] - 1) * 100
    
    # 피벗 테이블 생성
    pivot_return = monthly_data.pivot(index='month', columns='year', values='return_pct').fillna(0)
    pivot_dd = monthly_data.pivot(index='month', columns='year', values='min_dd').fillna(0)
    
    # 월 이름으로 인덱스 변경
    pivot_return.index = [month_name[i] for i in pivot_return.index]
    pivot_dd.index = [month_name[i] for i in pivot_dd.index]
    
    # 히트맵 그리기
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # 수익률 히트맵
    sns.heatmap(pivot_return, annot=True, cmap='RdYlGn', fmt='.2f', ax=axes[0], cbar_kws={'label': '%'})
    axes[0].set_title('Monthly Returns (%)')
    axes[0].set_ylabel('')
    
    # 드로다운 히트맵
    sns.heatmap(pivot_dd, annot=True, cmap='RdYlBu_r', fmt='.2f', ax=axes[1], cbar_kws={'label': '%'})
    axes[1].set_title('Monthly Maximum Drawdown (%)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# 연환산 수익률 계산 함수
def calculate_annualized_return(df):
    # 처음 기록과 마지막 기록의 날짜 차이 계산 (일 단위)
    start_date = pd.to_datetime(df['time'].iloc[0])
    end_date = pd.to_datetime(df['time'].iloc[-1])
    days = (end_date - start_date).days
    
    # 리턴 계산
    total_return = df['end_capital'].iloc[-1] / df['start_capital'].iloc[0]
    
    # 연환산 수익률 계산
    if days > 0:
        annualized_return = (total_return ** (365 / days)) - 1
        return annualized_return * 100  # 퍼센트로 변환
    else:
        return 0

# 일별 수익률 계산 함수
def calculate_daily_returns(df):
    # 시간 정렬
    df = df.sort_values('time')
    
    # 시간 변환
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    # 일별 데이터로 그룹화
    daily_data = df.groupby('date').agg({'end_capital': 'last'}).reset_index()
    
    # 일별 수익률 계산
    daily_data['prev_capital'] = daily_data['end_capital'].shift(1)
    daily_data = daily_data.dropna()  # 첫번째 행(이전 값이 없는) 제거
    
    if len(daily_data) > 0:
        daily_data['daily_return'] = (daily_data['end_capital'] / daily_data['prev_capital']) - 1
        return daily_data
    else:
        return pd.DataFrame(columns=['date', 'end_capital', 'prev_capital', 'daily_return'])

# 샤프 비율(Sharpe Ratio) 계산 함수
def calculate_sharpe_ratio(daily_returns_df, risk_free_rate=0.02):
    # 일별 수익률 없으면 계산 불가
    if len(daily_returns_df) <= 1:
        return 0
    
    # 일별 수익률 평균
    avg_daily_return = daily_returns_df['daily_return'].mean()
    
    # 일별 수익률 표준편차
    std_daily_return = daily_returns_df['daily_return'].std()
    
    # 일별 무위험 이자율
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/365) - 1
    
    # 샤프 비율 계산
    if std_daily_return > 0:
        sharpe_ratio = (avg_daily_return - daily_risk_free_rate) / std_daily_return
        # 연단위 샤프 비율로 변환
        annual_sharpe = sharpe_ratio * sqrt(252)  # 거래일 252일 기준
        return annual_sharpe
    else:
        return 0

# 칼마 비율(Calmar Ratio) 계산 함수
def calculate_calmar_ratio(annualized_return, max_drawdown):
    # 최대 낙폭이 없거나 0이면 계산 불가
    if max_drawdown >= 0:
        return 0
    
    # 칼마 비율 계산 (연환산 수익률 / 절대값(MDD))
    calmar_ratio = annualized_return / abs(max_drawdown)  # 절대값 사용
    return calmar_ratio

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
                
                # 오버플로우 방지 처리
                if abs(trader_contribution) > model_capital * 0.5:  # 리미터 적용
                    print(f"Warning: Excessively large contribution from {trader}: {trader_contribution}. Capping at 10%")
                    if trader_contribution > 0:
                        trader_contribution = model_capital * 0.1  # 최대 10% 수익 제한
                    else:
                        trader_contribution = model_capital * -0.1  # 최대 10% 손실 제한
                
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

# JSON 파일로 저장
with open(summary_path, 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)
print(f"Summary data saved: {summary_path}")

# 요약 마크다운 파일 저장
summary_md_path = os.path.join(output_dir, 'model_portfolio_summary.md')
with open(summary_md_path, 'w') as f:
    f.write('# 모델 포트폴리오 성과 요약\n\n')
    
    f.write('## 결과 요약\n\n')
    f.write(f"* 총 수익률: {((model_capital / model_initial_capital) - 1) * 100:+.2f}%\n")
    f.write(f"* 연환산 수익률: {annualized_ret:+.2f}%\n")
    f.write(f"* 표준편차: {daily_returns['daily_return'].std() * 100:.2f}%\n")
    f.write(f"* 샤프 비율: {sharpe:.2f}\n")
    f.write(f"* 칼마 비율: {calmar:.2f}\n")
    f.write(f"* MDD: {mdd:.2f}%\n")
    
    f.write('\n## 포트폴리오 정보\n\n')
    f.write(f"* 초기 자본: {model_initial_capital:,.2f} USD\n")
    f.write(f"* 최종 자본: {model_capital:,.2f} USD\n")
    f.write(f"* 모니터링 기간: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} ({(pd.to_datetime(df['time'].iloc[-1]) - pd.to_datetime(df['time'].iloc[0])).days} 일)\n")
    f.write(f"* 총 차트 데이터 개수: {len(df)} 개\n")
    f.write(f"* 총 트레이딩 일수: {len(daily_returns)} 일\n")

print(f"Summary markdown saved: {summary_md_path}")


# Save to CSV file
csv_path = os.path.join(output_dir, 'model_portfolio_capital.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV file saved: {csv_path}")

# 마크다운 파일 생성
md_lines = []
md_lines.append("# 모델 포트폴리오 자본 변화 분석\n")
md_lines.append(f"초기 자본: {model_initial_capital:,.2f} USD\n")
md_lines.append(f"최종 자본: {model_capital:,.2f} USD\n")
md_lines.append(f"총 수익률: {((model_capital / model_initial_capital) - 1) * 100:+.2f}%\n")

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

# Plot capital change over time
plt.figure(figsize=(14, 7))
plt.plot(pd.to_datetime(df['time']), df['end_capital'], marker='o', linestyle='-', markersize=3)
plt.title('Model Portfolio Capital Over Time')
plt.xlabel('Date')
plt.ylabel('Capital (USD)')
plt.grid(True)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # every 7 days
plt.gcf().autofmt_xdate()  # Rotate date labels

plt.tight_layout()
graph_path = os.path.join(output_dir, 'model_portfolio_capital.png')
plt.savefig(graph_path)
plt.close()
print(f"Graph saved: {graph_path}")

# MDD 그래프 생성 및 저장
plot_mdd(df, os.path.join(output_dir, 'model_portfolio_drawdown.png'))
print(f"Drawdown graph saved: {os.path.join(output_dir, 'model_portfolio_drawdown.png')}")

# MDD 히트맵 생성 및 저장
plot_mdd_heatmap(df, os.path.join(output_dir, 'model_portfolio_heatmap.png'))
print(f"Drawdown heatmap saved: {os.path.join(output_dir, 'model_portfolio_heatmap.png')}")

# MDD 데이터 저장
mdd_df = df[['time', 'end_capital', 'cumulative_max', 'drawdown']]
mdd_df.to_csv(os.path.join(output_dir, 'model_portfolio_drawdown.csv'), index=False)
print(f"Drawdown data saved: {os.path.join(output_dir, 'model_portfolio_drawdown.csv')}")

# 월별 MDD 데이터 저장
monthly_mdd.to_csv(os.path.join(output_dir, 'model_portfolio_monthly_mdd.csv'), index=False)
print(f"Monthly MDD data saved: {os.path.join(output_dir, 'model_portfolio_monthly_mdd.csv')}")

# Monthly return graph creation and saving
df['month'] = df['time'].dt.strftime('%Y-%m')
monthly_returns = df.groupby('month').apply(lambda x: (x['end_capital'].iloc[-1] / x['start_capital'].iloc[0] - 1) * 100).reset_index()
monthly_returns.columns = ['month', 'monthly_return']

plt.figure(figsize=(12, 6))
plt.bar(monthly_returns['month'], monthly_returns['monthly_return'])
plt.title('Model Portfolio Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Monthly Return (%)')
plt.grid(axis='y')
plt.xticks(rotation=45)

# Save graph
monthly_graph_path = os.path.join(output_dir, 'model_portfolio_monthly_returns.png')
plt.tight_layout()
plt.savefig(monthly_graph_path, dpi=300)
plt.close()
print(f"Monthly returns graph saved: {monthly_graph_path}")

# Save monthly returns to CSV
monthly_csv_path = os.path.join(output_dir, 'model_portfolio_monthly_returns.csv')
monthly_returns.to_csv(monthly_csv_path, index=False)
print(f"Monthly returns CSV saved: {monthly_csv_path}")
