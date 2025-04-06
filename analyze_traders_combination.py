#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy.optimize import minimize

# 결과 저장 디렉토리 생성
RESULTS_DIR = 'analysis_results/combination'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_trader_data(trader_id, period=None):
    """트레이더 데이터 로드"""
    if period:
        file_path = f'analysis_results/{trader_id}/{period}/analyzed_data.csv'
    else:
        file_path = f'analysis_results/{trader_id}/overall/analyzed_data.csv'
    
    df = pd.read_csv(file_path)
    
    # 날짜 형식 변환
    df['Close_Time_KST'] = pd.to_datetime(df['Close_Time_KST'])
    df['Open_Time_KST'] = pd.to_datetime(df['Open_Time_KST'])
    
    # 트레이더 ID 추가
    df['Trader'] = trader_id
    
    return df

def analyze_trader_performance(df, trader_id, initial_capital=None):
    """트레이더 성과 분석"""
    # 기본 정보
    trade_count = len(df)
    start_date = df['Close_Time_KST'].min()
    end_date = df['Close_Time_KST'].max()
    total_pnl = df['PnL_Numeric'].sum()
    win_rate = (df['PnL_Numeric'] > 0).mean() * 100
    
    # 최대 드로다운 계산
    if 'Period_Cumulative_PnL' in df.columns:
        max_drawdown = df['Period_Cumulative_PnL'].min()
    else:
        max_drawdown = df['Cumulative_PnL'].min()
    
    # 수익률 계산
    if initial_capital:
        total_return = total_pnl / initial_capital * 100
        # 연간 수익률 계산 (거래일 기준)
        days = (end_date - start_date).days
        annual_return = total_return * 365 / days if days > 0 else 0
    else:
        total_return = "N/A"
        annual_return = "N/A"
    
    # 평균 수익 및 손실
    avg_win = df[df['PnL_Numeric'] > 0]['PnL_Numeric'].mean() if len(df[df['PnL_Numeric'] > 0]) > 0 else 0
    avg_loss = df[df['PnL_Numeric'] < 0]['PnL_Numeric'].mean() if len(df[df['PnL_Numeric'] < 0]) > 0 else 0
    
    # 샤프 비율 계산 (일간 수익률 기준)
    df['Date'] = df['Close_Time_KST'].dt.date
    daily_returns = df.groupby('Date')['PnL_Numeric'].sum()
    if initial_capital:
        daily_returns_pct = daily_returns / initial_capital * 100
        sharpe_ratio = daily_returns_pct.mean() / daily_returns_pct.std() * np.sqrt(365) if daily_returns_pct.std() > 0 else 0
    else:
        sharpe_ratio = "N/A"
    
    # 최대 연속 손실 및 이익 계산
    df['Win'] = df['PnL_Numeric'] > 0
    df['Lose'] = df['PnL_Numeric'] < 0
    
    # 연속 이익/손실 계산
    win_streak = 0
    lose_streak = 0
    max_win_streak = 0
    max_lose_streak = 0
    current_win_streak = 0
    current_lose_streak = 0
    
    for win, lose in zip(df['Win'], df['Lose']):
        if win:
            current_win_streak += 1
            current_lose_streak = 0
        elif lose:
            current_lose_streak += 1
            current_win_streak = 0
        
        max_win_streak = max(max_win_streak, current_win_streak)
        max_lose_streak = max(max_lose_streak, current_lose_streak)
    
    # 결과 반환
    return {
        'Trader': trader_id,
        'Trade Count': trade_count,
        'Start Date': start_date,
        'End Date': end_date,
        'Total PnL': total_pnl,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Total Return (%)': total_return,
        'Annual Return (%)': annual_return,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Sharpe Ratio': sharpe_ratio,
        'Max Win Streak': max_win_streak,
        'Max Lose Streak': max_lose_streak
    }

def calculate_daily_returns(df, initial_capital=None):
    """일간 수익률 계산"""
    # 날짜별 수익 합계
    df['Date'] = df['Close_Time_KST'].dt.date
    daily_returns = df.groupby(['Date', 'Trader'])['PnL_Numeric'].sum().reset_index()
    
    # 수익률 계산
    if initial_capital:
        daily_returns['Return_Pct'] = daily_returns['PnL_Numeric'] / initial_capital * 100
    else:
        daily_returns['Return_Pct'] = daily_returns['PnL_Numeric']
    
    return daily_returns

def simulate_portfolio(weights, daily_returns_list, initial_capitals):
    """포트폴리오 시뮬레이션"""
    # 날짜 범위 찾기
    all_dates = set()
    for returns in daily_returns_list:
        all_dates.update(returns['Date'])
    
    all_dates = sorted(all_dates)
    
    # 포트폴리오 수익 계산
    portfolio_returns = []
    
    for date in all_dates:
        portfolio_return = 0
        for i, (returns, weight, capital) in enumerate(zip(daily_returns_list, weights, initial_capitals)):
            date_returns = returns[returns['Date'] == date]
            if not date_returns.empty:
                if capital:
                    # 수익률 기반 계산
                    portfolio_return += weight * date_returns['Return_Pct'].sum()
                else:
                    # 금액 기반 계산 (가중치를 금액에 적용)
                    portfolio_return += weight * date_returns['PnL_Numeric'].sum()
        
        portfolio_returns.append(portfolio_return)
    
    # 포트폴리오 성과 지표 계산
    portfolio_return_mean = np.mean(portfolio_returns)
    portfolio_return_std = np.std(portfolio_returns)
    sharpe_ratio = portfolio_return_mean / portfolio_return_std * np.sqrt(252) if portfolio_return_std > 0 else 0
    
    # 누적 수익률 및 최대 드로다운 계산
    cumulative_returns = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = np.max(drawdown)
    
    return {
        'Returns': portfolio_returns,
        'Cumulative Returns': cumulative_returns,
        'Mean Return': portfolio_return_mean,
        'Std Return': portfolio_return_std,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Dates': all_dates
    }

def objective_function(weights, daily_returns_list, initial_capitals):
    """최적화를 위한 목적 함수"""
    portfolio = simulate_portfolio(weights, daily_returns_list, initial_capitals)
    
    # 샤프 비율 최대화 (음수 반환하여 minimize 함수로 최대화)
    return -portfolio['Sharpe Ratio']

def optimize_weights(daily_returns_list, initial_capitals, n_traders):
    """트레이더 가중치 최적화"""
    # 초기 가중치 (균등 분배)
    initial_weights = np.ones(n_traders) / n_traders
    
    # 제약 조건: 가중치 합 = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # 각 가중치는 0과 1 사이
    bounds = tuple((0, 1) for _ in range(n_traders))
    
    # 최적화
    result = minimize(
        objective_function,
        initial_weights,
        args=(daily_returns_list, initial_capitals),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

def plot_cumulative_returns(portfolio_results, trader_ids, weights, file_name='cumulative_returns.png'):
    """누적 수익률 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 누적 수익률
    for i, (trader_id, portfolio) in enumerate(zip(trader_ids, portfolio_results['Individual'])):
        plt.plot(portfolio['Dates'], portfolio['Cumulative Returns'], label=f'{trader_id} (Weight: {weights[i]:.2f})')
    
    # 포트폴리오 누적 수익률
    plt.plot(portfolio_results['Combined']['Dates'], portfolio_results['Combined']['Cumulative Returns'], 
             label='Combined Portfolio', linewidth=2, color='black')
    
    plt.title('Cumulative Returns Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_drawdowns(portfolio_results, trader_ids, weights, file_name='drawdowns.png'):
    """드로다운 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 드로다운
    for i, (trader_id, portfolio) in enumerate(zip(trader_ids, portfolio_results['Individual'])):
        cumulative_returns = portfolio['Cumulative Returns']
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max * 100 if any(running_max > 0) else np.zeros_like(cumulative_returns)
        plt.plot(portfolio['Dates'], drawdown, label=f'{trader_id} (Weight: {weights[i]:.2f})')
    
    # 포트폴리오 드로다운
    cumulative_returns = portfolio_results['Combined']['Cumulative Returns']
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max * 100 if any(running_max > 0) else np.zeros_like(cumulative_returns)
    plt.plot(portfolio_results['Combined']['Dates'], drawdown, label='Combined Portfolio', linewidth=2, color='black')
    
    plt.title('Drawdowns Comparison (%)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # 드로다운을 아래로 표시
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def main():
    # 트레이더 설정
    trader_configs = [
        {'id': 'hummusXBT', 'period': 'period2', 'initial_capital': 2000000},
        {'id': 'RebelOfBabylon', 'period': None, 'initial_capital': 1000000},
        {'id': 'Panic', 'period': 'period2', 'initial_capital': 2000000}
    ]
    
    # 데이터 로드
    trader_data = {}
    for config in trader_configs:
        trader_id = config['id']
        trader_data[trader_id] = load_trader_data(trader_id, config['period'])
    
    # 트레이더 성과 분석
    trader_performance = {}
    for config in trader_configs:
        trader_id = config['id']
        trader_performance[trader_id] = analyze_trader_performance(
            trader_data[trader_id], 
            trader_id, 
            config['initial_capital']
        )
    
    # 성과 요약 출력
    print("===== 트레이더 성과 요약 =====")
    performance_df = pd.DataFrame(trader_performance).T
    print(performance_df[['Trade Count', 'Total PnL', 'Win Rate', 'Max Drawdown', 'Total Return (%)', 'Annual Return (%)', 'Sharpe Ratio']])
    
    # 성과 요약 저장
    performance_df.to_csv(os.path.join(RESULTS_DIR, 'trader_performance_summary.csv'))
    
    # 일간 수익률 계산
    daily_returns_list = []
    initial_capitals = []
    for config in trader_configs:
        trader_id = config['id']
        daily_returns = calculate_daily_returns(trader_data[trader_id], config['initial_capital'])
        daily_returns_list.append(daily_returns)
        initial_capitals.append(config['initial_capital'])
    
    # 가중치 최적화
    optimized_weights = optimize_weights(daily_returns_list, initial_capitals, len(trader_configs))
    
    # 최적화된 가중치 출력
    print("\n===== 최적화된 가중치 =====")
    for i, config in enumerate(trader_configs):
        print(f"{config['id']}: {optimized_weights[i]:.4f}")
    
    # 다양한 가중치 조합 테스트
    weight_combinations = [
        optimized_weights,  # 최적화된 가중치
        np.ones(len(trader_configs)) / len(trader_configs),  # 균등 가중치
        [0.5, 0.25, 0.25],  # hummusXBT 중심
        [0.25, 0.5, 0.25],  # RebelOfBabylon 중심
        [0.25, 0.25, 0.5]   # Panic 중심
    ]
    
    # 각 가중치 조합에 대한 포트폴리오 시뮬레이션
    portfolio_results = {}
    
    for weights in weight_combinations:
        # 가중치 이름 생성
        weight_name = '_'.join([f"{w:.2f}" for w in weights])
        
        # 개별 트레이더 포트폴리오
        individual_portfolios = []
        for i, (returns, capital) in enumerate(zip(daily_returns_list, initial_capitals)):
            individual_weights = np.zeros(len(trader_configs))
            individual_weights[i] = 1.0
            individual_portfolio = simulate_portfolio(individual_weights, daily_returns_list, initial_capitals)
            individual_portfolios.append(individual_portfolio)
        
        # 조합 포트폴리오
        combined_portfolio = simulate_portfolio(weights, daily_returns_list, initial_capitals)
        
        portfolio_results[weight_name] = {
            'Individual': individual_portfolios,
            'Combined': combined_portfolio,
            'Weights': weights
        }
    
    # 결과 출력
    print("\n===== 포트폴리오 성과 비교 =====")
    portfolio_performance = []
    
    for weight_name, results in portfolio_results.items():
        combined = results['Combined']
        portfolio_performance.append({
            'Weights': weight_name,
            'Mean Return': combined['Mean Return'],
            'Sharpe Ratio': combined['Sharpe Ratio'],
            'Max Drawdown': combined['Max Drawdown'],
            'Final Return': combined['Cumulative Returns'][-1]
        })
    
    portfolio_df = pd.DataFrame(portfolio_performance)
    print(portfolio_df)
    
    # 결과 저장
    portfolio_df.to_csv(os.path.join(RESULTS_DIR, 'portfolio_performance_comparison.csv'))
    
    # 그래프 생성
    trader_ids = [config['id'] for config in trader_configs]
    
    # 최적화된 가중치 그래프
    plot_cumulative_returns(
        portfolio_results['_'.join([f"{w:.2f}" for w in optimized_weights])],
        trader_ids,
        optimized_weights,
        'optimized_cumulative_returns.png'
    )
    
    plot_drawdowns(
        portfolio_results['_'.join([f"{w:.2f}" for w in optimized_weights])],
        trader_ids,
        optimized_weights,
        'optimized_drawdowns.png'
    )
    
    # 균등 가중치 그래프
    equal_weights = np.ones(len(trader_configs)) / len(trader_configs)
    plot_cumulative_returns(
        portfolio_results['_'.join([f"{w:.2f}" for w in equal_weights])],
        trader_ids,
        equal_weights,
        'equal_cumulative_returns.png'
    )
    
    plot_drawdowns(
        portfolio_results['_'.join([f"{w:.2f}" for w in equal_weights])],
        trader_ids,
        equal_weights,
        'equal_drawdowns.png'
    )
    
    print("\n===== 분석 완료 =====")
    print(f"결과가 {RESULTS_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
