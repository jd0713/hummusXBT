#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy.optimize import minimize
from scipy import stats

# 결과 저장 디렉토리 생성
RESULTS_DIR = 'analysis_results/hummus_panic'
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

def create_timeline_data(df, initial_capital, freq='1H', apply_slippage=False, slippage_pct=0.05):
    """시간별 포트폴리오 가치 데이터 생성"""
    # 전체 기간 설정
    start_date = df['Open_Time_KST'].min()
    end_date = df['Close_Time_KST'].max()
    
    # 시간별 타임라인 생성
    timeline = pd.date_range(start=start_date, end=end_date, freq=freq)
    timeline_df = pd.DataFrame(index=timeline)
    timeline_df['Portfolio_Value'] = initial_capital
    
    # 각 거래의 영향 계산
    for _, trade in df.iterrows():
        open_time = trade['Open_Time_KST']
        close_time = trade['Close_Time_KST']
        pnl = trade['PnL_Numeric']
        
        # 슬리피지 적용 (투자금액에 대한 비율로 계산)
        if apply_slippage:
            # 슬리피지는 거래 금액에 비례하여 발생
            trade_size = abs(trade.get('Size_USDT_Abs', 0))
            if trade_size == 0 and 'Entry_Price' in trade and 'Qty' in trade:
                trade_size = abs(trade['Entry_Price'] * trade['Qty'])
            
            slippage_amount = trade_size * (slippage_pct / 100) if trade_size > 0 else 0
            pnl -= slippage_amount  # 슬리피지를 PnL에서 차감
        
        # 거래 종료 시점에 PnL 반영
        close_idx = timeline_df.index.searchsorted(close_time)
        if close_idx < len(timeline_df):
            timeline_df.iloc[close_idx:, 0] += pnl
    
    # 누적 수익률 계산
    timeline_df['Return_Pct'] = (timeline_df['Portfolio_Value'] / initial_capital - 1) * 100
    
    # 최대 드로다운 계산
    timeline_df['Peak'] = timeline_df['Portfolio_Value'].cummax()
    timeline_df['Drawdown'] = (timeline_df['Peak'] - timeline_df['Portfolio_Value']) / timeline_df['Peak'] * 100
    
    # 일별 수익률 계산 (성과 지표 계산용)
    timeline_df['Daily_Return'] = timeline_df['Portfolio_Value'].pct_change()
    
    return timeline_df

def simulate_combined_portfolio(trader_timelines, weights, fixed_initial_capital=1000000):
    """여러 트레이더의 타임라인을 결합하여 포트폴리오 시뮤레이션"""
    # 모든 타임라인의 인덱스 통합
    all_indices = set()
    for timeline_df in trader_timelines:
        all_indices.update(timeline_df.index)
    
    all_indices = sorted(all_indices)
    combined_df = pd.DataFrame(index=all_indices)
    
    # 고정된 초기 포트폴리오 가치 설정
    initial_portfolio_value = fixed_initial_capital
    combined_df['Portfolio_Value'] = initial_portfolio_value
    
    # 각 트레이더의 상대적 수익률 계산 및 가중 포트폴리오 가치 계산
    for i, (timeline_df, weight) in enumerate(zip(trader_timelines, weights)):
        if not timeline_df.empty:
            # 각 트레이더의 상대적 수익률 계산
            trader_returns = timeline_df['Return_Pct'].reindex(combined_df.index, method='ffill') / 100
            
            # 가중 포트폴리오 가치에 반영
            combined_df['Portfolio_Value'] = combined_df['Portfolio_Value'] * (1 + trader_returns * weight)
    
    # 포트폴리오 수익률 및 드로다운 계산
    combined_df['Return_Pct'] = (combined_df['Portfolio_Value'] / initial_portfolio_value - 1) * 100
    combined_df['Peak'] = combined_df['Portfolio_Value'].cummax()
    combined_df['Drawdown'] = (combined_df['Peak'] - combined_df['Portfolio_Value']) / combined_df['Peak'] * 100
    
    # 일별 수익률 계산 (성과 지표 계산용)
    combined_df['Daily_Return'] = combined_df['Portfolio_Value'].pct_change()
    
    return combined_df

def calculate_concurrent_trades(trader_data, trader_ids):
    """동시 진행 중인 거래 수와 사용 자본 계산"""
    # 전체 기간 설정
    start_date = min([df['Open_Time_KST'].min() for df in trader_data.values()])
    end_date = max([df['Close_Time_KST'].max() for df in trader_data.values()])
    
    # 시간별 타임라인 생성
    timeline = pd.date_range(start=start_date, end=end_date, freq='1H')
    concurrent_trades = pd.DataFrame(index=timeline)
    concurrent_trades['Total_Trades'] = 0
    concurrent_trades['Total_Capital_Used'] = 0
    
    for trader_id in trader_ids:
        concurrent_trades[f'{trader_id}_Trades'] = 0
        concurrent_trades[f'{trader_id}_Capital'] = 0
        
        df = trader_data[trader_id]
        
        for _, trade in df.iterrows():
            open_time = trade['Open_Time_KST']
            close_time = trade['Close_Time_KST']
            
            # 거래 크기 (USDT)
            size = trade.get('Size_USDT_Abs', 0)
            if size == 0 and 'Max_Size_USDT' in trade.columns:
                size_str = trade['Max_Size_USDT']
                if isinstance(size_str, str) and 'USDT' in size_str:
                    size = float(size_str.replace('USDT', '').replace(',', '').strip())
            
            # 거래 기간 동안 동시 거래 수와 사용 자본 증가
            mask = (concurrent_trades.index >= open_time) & (concurrent_trades.index <= close_time)
            concurrent_trades.loc[mask, 'Total_Trades'] += 1
            concurrent_trades.loc[mask, 'Total_Capital_Used'] += size
            concurrent_trades.loc[mask, f'{trader_id}_Trades'] += 1
            concurrent_trades.loc[mask, f'{trader_id}_Capital'] += size
    
    return concurrent_trades

def calculate_performance_metrics(timeline_df):
    """포트폴리오 성과 지표 계산"""
    # 기본 지표
    total_return = timeline_df['Return_Pct'].iloc[-1]
    max_drawdown = timeline_df['Drawdown'].max()
    
    # 거래 기간 계산
    start_date = timeline_df.index[0]
    end_date = timeline_df.index[-1]
    days = (end_date - start_date).days
    
    # 연율화된 수익률 계산 (CAGR)
    final_return = total_return / 100  # 백분율에서 소수점으로 변환
    annualized_return = ((1 + final_return) ** (365 / days) - 1) * 100 if days > 0 else 0
    
    # 일별 수익률로 변환하여 변동성 계산
    daily_returns = timeline_df['Portfolio_Value'].resample('D').last().pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100  # 연율화된 표준편차
    
    # 샤프 비율 계산 (무위험 이자율 2% 가정)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else 0
    
    # 칼마 비율 계산 (Max Drawdown 기반 위험 조정 수익률)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    # 승률 계산 (일별 수익률 기준)
    win_rate = (daily_returns > 0).mean() * 100
    
    # 수익/손실 비율 계산
    profit_days = daily_returns[daily_returns > 0]
    loss_days = daily_returns[daily_returns < 0]
    profit_loss_ratio = abs(profit_days.mean() / loss_days.mean()) if len(loss_days) > 0 and loss_days.mean() != 0 else float('inf')
    
    # 최대 연속 수익/손실일 계산
    daily_returns_binary = (daily_returns > 0).astype(int)
    max_consecutive_wins = max([len(list(g)) for k, g in itertools.groupby(daily_returns_binary) if k == 1], default=0)
    max_consecutive_losses = max([len(list(g)) for k, g in itertools.groupby(daily_returns_binary) if k == 0], default=0)
    
    # 결과 반환
    return {
        'Total Return (%)': total_return,
        'Annualized Return (%)': annualized_return,
        'Max Drawdown (%)': max_drawdown,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Win Rate (%)': win_rate,
        'Profit/Loss Ratio': profit_loss_ratio,
        'Max Consecutive Wins': max_consecutive_wins,
        'Max Consecutive Losses': max_consecutive_losses,
        'Trading Days': days
    }

def optimize_portfolio(trader_timelines, target='sharpe', initial_capital=1000000):
    """포트폴리오 최적화 (샤프 비율 또는 칼마 비율 기준)"""
    def objective_function(weights):
        # 가중치 합이 1이 되도록 정규화
        weights = np.array(weights) / np.sum(weights)
        
        # 포트폴리오 시뮬레이션
        portfolio = simulate_combined_portfolio(trader_timelines, weights, initial_capital)
        
        # 성과 지표 계산
        metrics = calculate_performance_metrics(portfolio)
        
        # 목표에 따라 최적화 기준 선택
        if target == 'sharpe':
            # 샤프 비율 최대화 (음수로 반환하여 minimize 함수 사용)
            return -metrics['Sharpe Ratio']
        elif target == 'calmar':
            # 칼마 비율 최대화 (음수로 반환하여 minimize 함수 사용)
            return -metrics['Calmar Ratio']
        elif target == 'return':
            # 수익률 최대화 (음수로 반환하여 minimize 함수 사용)
            return -metrics['Annualized Return (%)']
        else:
            # 기본값은 샤프 비율
            return -metrics['Sharpe Ratio']
    
    # 초기 가중치 (균등 배분)
    n_traders = len(trader_timelines)
    initial_weights = [1/n_traders] * n_traders
    
    # 제약 조건: 가중치 합 = 1, 각 가중치 >= 0
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_traders))
    
    # 최적화 실행
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # 최적 가중치 정규화
    optimal_weights = result['x'] / np.sum(result['x'])
    
    return optimal_weights

def plot_portfolio_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, file_name):
    """포트폴리오 비교 그래프 생성"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 수익률 그래프
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, timeline_df['Return_Pct'], label=f'{trader_id}')
    
    # 포트폴리오 수익률 그래프
    line_styles = ['-', '--', '-.', ':']
    for i, (timeline_df, portfolio_name) in enumerate(zip(portfolio_timelines, portfolio_names)):
        plt.plot(timeline_df.index, timeline_df['Return_Pct'], 
                 label=f'{portfolio_name}', 
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2.5)
    
    plt.title('Portfolio Performance Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_drawdowns_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, file_name):
    """드로다운 비교 그래프 생성"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 드로다운 그래프
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, -timeline_df['Drawdown'], label=f'{trader_id}')
    
    # 포트폴리오 드로다운 그래프
    line_styles = ['-', '--', '-.', ':']
    for i, (timeline_df, portfolio_name) in enumerate(zip(portfolio_timelines, portfolio_names)):
        plt.plot(timeline_df.index, -timeline_df['Drawdown'], 
                 label=f'{portfolio_name}', 
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2.5)
    
    plt.title('Drawdown Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_weight_sensitivity(trader_timelines, trader_ids, metric='sharpe', initial_capital=1000000):
    """가중치 민감도 분석 그래프 생성"""
    # hummusXBT의 가중치를 0부터 1까지 변화시키면서 성과 지표 계산
    weights_range = np.linspace(0, 1, 21)  # 0, 0.05, 0.1, ..., 0.95, 1
    metrics = []
    
    for w1 in weights_range:
        w2 = 1 - w1  # Panic의 가중치
        weights = [w1, w2]
        
        # 포트폴리오 시뮬레이션
        portfolio = simulate_combined_portfolio(trader_timelines, weights, initial_capital)
        
        # 성과 지표 계산
        portfolio_metrics = calculate_performance_metrics(portfolio)
        metrics.append(portfolio_metrics)
    
    # 그래프 그리기
    plt.figure(figsize=(16, 12))
    
    # 수익률, 변동성, 드로다운 그래프
    plt.subplot(2, 2, 1)
    plt.plot(weights_range * 100, [m['Annualized Return (%)'] for m in metrics], 'b-', linewidth=2, label='Annualized Return')
    plt.title('Annualized Return vs. Weight of ' + trader_ids[0], fontsize=14)
    plt.xlabel(f'Weight of {trader_ids[0]} (%)', fontsize=12)
    plt.ylabel('Annualized Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(weights_range * 100, [m['Volatility (%)'] for m in metrics], 'r-', linewidth=2, label='Volatility')
    plt.title('Volatility vs. Weight of ' + trader_ids[0], fontsize=14)
    plt.xlabel(f'Weight of {trader_ids[0]} (%)', fontsize=12)
    plt.ylabel('Volatility (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(weights_range * 100, [m['Max Drawdown (%)'] for m in metrics], 'g-', linewidth=2, label='Max Drawdown')
    plt.title('Max Drawdown vs. Weight of ' + trader_ids[0], fontsize=14)
    plt.xlabel(f'Weight of {trader_ids[0]} (%)', fontsize=12)
    plt.ylabel('Max Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 샤프 비율과 칼마 비율 그래프
    plt.subplot(2, 2, 4)
    plt.plot(weights_range * 100, [m['Sharpe Ratio'] for m in metrics], 'b-', linewidth=2, label='Sharpe Ratio')
    plt.plot(weights_range * 100, [m['Calmar Ratio'] for m in metrics], 'r--', linewidth=2, label='Calmar Ratio')
    plt.title('Risk-Adjusted Returns vs. Weight of ' + trader_ids[0], fontsize=14)
    plt.xlabel(f'Weight of {trader_ids[0]} (%)', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'weight_sensitivity_analysis.png'), dpi=300)
    plt.close()
    
    # 최적 가중치 찾기
    if metric == 'sharpe':
        best_idx = np.argmax([m['Sharpe Ratio'] for m in metrics])
    elif metric == 'calmar':
        best_idx = np.argmax([m['Calmar Ratio'] for m in metrics])
    elif metric == 'return':
        best_idx = np.argmax([m['Annualized Return (%)'] for m in metrics])
    else:
        best_idx = np.argmax([m['Sharpe Ratio'] for m in metrics])
    
    best_weight = weights_range[best_idx]
    best_metrics = metrics[best_idx]
    
    return best_weight, best_metrics

def analyze_correlation(trader_timelines, trader_ids):
    """트레이더 간 상관관계 분석"""
    # 모든 타임라인의 인덱스 통합
    all_indices = set()
    for timeline_df in trader_timelines:
        all_indices.update(timeline_df.index)
    
    all_indices = sorted(all_indices)
    
    # 일별 수익률 데이터프레임 생성
    daily_returns = pd.DataFrame(index=all_indices)
    
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        # 일별 수익률 계산
        returns = timeline_df['Portfolio_Value'].pct_change()
        daily_returns[trader_id] = returns.reindex(daily_returns.index, method='ffill')
    
    # 일별 데이터로 리샘플링
    daily_returns = daily_returns.resample('D').last().dropna()
    
    # 상관관계 계산
    correlation = daily_returns.corr()
    
    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation between Traders', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    return correlation

import itertools

def main():
    # 트레이더 설정
    trader_configs = [
        {'id': 'hummusXBT', 'period': 'period2', 'initial_capital': 2000000, 'apply_slippage': False},
        {'id': 'Panic', 'period': 'period2', 'initial_capital': 1000000, 'apply_slippage': True}
    ]
    
    # 데이터 로드
    trader_data = {}
    trader_ids = []
    initial_capitals = []
    
    for config in trader_configs:
        trader_id = config['id']
        trader_ids.append(trader_id)
        initial_capitals.append(config['initial_capital'])
        trader_data[trader_id] = load_trader_data(trader_id, config['period'])
    
    # 시간별 포트폴리오 가치 계산
    trader_timelines = []
    for config in trader_configs:
        trader_id = config['id']
        timeline_df = create_timeline_data(
            trader_data[trader_id], 
            config['initial_capital'],
            apply_slippage=config['apply_slippage']
        )
        trader_timelines.append(timeline_df)
    
    # 동시 진행 중인 거래 및 사용 자본 계산
    concurrent_trades = calculate_concurrent_trades(trader_data, trader_ids)
    
    # 상관관계 분석
    correlation = analyze_correlation(trader_timelines, trader_ids)
    
    # 다양한 포트폴리오 구성 테스트
    portfolio_timelines = []
    portfolio_names = []
    portfolio_weights = []
    
    # 1. 균등 가중치 포트폴리오 (50:50)
    equal_weights = [0.5, 0.5]
    equal_combined = simulate_combined_portfolio(trader_timelines, equal_weights)
    portfolio_timelines.append(equal_combined)
    portfolio_names.append('Equal Portfolio (50:50)')
    portfolio_weights.append(equal_weights)
    
    # 2. hummusXBT 중심 포트폴리오 (70:30)
    hummus_focused_weights = [0.7, 0.3]
    hummus_focused_combined = simulate_combined_portfolio(trader_timelines, hummus_focused_weights)
    portfolio_timelines.append(hummus_focused_combined)
    portfolio_names.append('Hummus Focused (70:30)')
    portfolio_weights.append(hummus_focused_weights)
    
    # 3. Panic 중심 포트폴리오 (30:70)
    panic_focused_weights = [0.3, 0.7]
    panic_focused_combined = simulate_combined_portfolio(trader_timelines, panic_focused_weights)
    portfolio_timelines.append(panic_focused_combined)
    portfolio_names.append('Panic Focused (30:70)')
    portfolio_weights.append(panic_focused_weights)
    
    # 4. 샤프 비율 최적화 포트폴리오
    sharpe_optimal_weights = optimize_portfolio(trader_timelines, target='sharpe')
    sharpe_optimal_combined = simulate_combined_portfolio(trader_timelines, sharpe_optimal_weights)
    portfolio_timelines.append(sharpe_optimal_combined)
    portfolio_names.append(f'Sharpe Optimal ({sharpe_optimal_weights[0]:.2f}:{sharpe_optimal_weights[1]:.2f})')
    portfolio_weights.append(sharpe_optimal_weights)
    
    # 5. 칼마 비율 최적화 포트폴리오 (드로다운 대비 수익률 최적화)
    calmar_optimal_weights = optimize_portfolio(trader_timelines, target='calmar')
    calmar_optimal_combined = simulate_combined_portfolio(trader_timelines, calmar_optimal_weights)
    portfolio_timelines.append(calmar_optimal_combined)
    portfolio_names.append(f'Calmar Optimal ({calmar_optimal_weights[0]:.2f}:{calmar_optimal_weights[1]:.2f})')
    portfolio_weights.append(calmar_optimal_weights)
    
    # 6. 수익률 최적화 포트폴리오
    return_optimal_weights = optimize_portfolio(trader_timelines, target='return')
    return_optimal_combined = simulate_combined_portfolio(trader_timelines, return_optimal_weights)
    portfolio_timelines.append(return_optimal_combined)
    portfolio_names.append(f'Return Optimal ({return_optimal_weights[0]:.2f}:{return_optimal_weights[1]:.2f})')
    portfolio_weights.append(return_optimal_weights)
    
    # 가중치 민감도 분석
    best_weight, best_metrics = plot_weight_sensitivity(trader_timelines, trader_ids)
    
    # 그래프 생성
    plot_portfolio_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, 'portfolio_comparison.png')
    plot_drawdowns_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, 'drawdowns_comparison.png')
    
    # 성과 지표 계산
    performance_summary = []
    
    # 개별 트레이더 성과 지표
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        metrics = calculate_performance_metrics(timeline_df)
        metrics['Portfolio'] = trader_id
        metrics['Weights'] = f'{trader_id} Only'
        performance_summary.append(metrics)
    
    # 포트폴리오 성과 지표
    for i, (timeline_df, portfolio_name, weights) in enumerate(zip(portfolio_timelines, portfolio_names, portfolio_weights)):
        metrics = calculate_performance_metrics(timeline_df)
        metrics['Portfolio'] = portfolio_name
        metrics['Weights'] = f'{weights[0]:.2f}:{weights[1]:.2f}'
        performance_summary.append(metrics)
    
    # 성과 요약 데이터프레임 생성
    summary_df = pd.DataFrame(performance_summary)
    
    # 결과 출력
    print("===== 트레이더 간 상관관계 =====")
    print(correlation)
    
    print("\n===== 포트폴리오 성과 요약 =====")
    print(summary_df[['Portfolio', 'Weights', 'Total Return (%)', 'Annualized Return (%)', 
                     'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Calmar Ratio', 
                     'Win Rate (%)', 'Trading Days']].to_string(index=False))
    
    print("\n===== 가중치 민감도 분석 결과 =====")
    print(f"최적 가중치 (샤프 비율 기준): {best_weight:.2f} ({trader_ids[0]}) : {1-best_weight:.2f} ({trader_ids[1]})")
    print(f"샤프 비율: {best_metrics['Sharpe Ratio']:.2f}")
    print(f"칼마 비율: {best_metrics['Calmar Ratio']:.2f}")
    print(f"연율화 수익률: {best_metrics['Annualized Return (%)']:.2f}%")
    print(f"최대 드로다운: {best_metrics['Max Drawdown (%)']:.2f}%")
    
    # 결과 저장
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'performance_summary.csv'), index=False)
    
    print("\n===== 분석 완료 =====")
    print(f"결과가 {RESULTS_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
