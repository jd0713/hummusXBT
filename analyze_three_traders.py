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
import itertools

# 결과 저장 디렉토리 생성
RESULTS_DIR = 'analysis_results/three_traders'
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
        # 일별 수익률 추출 및 통합 인덱스에 맞춤
        daily_returns[trader_id] = timeline_df['Daily_Return'].reindex(all_indices, method='ffill')
    
    # 일별 데이터로 리샘플링
    daily_returns = daily_returns.resample('D').last().dropna()
    
    # 상관관계 계산
    correlation = daily_returns.corr()
    
    # 상관관계 히트맵 그래프 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Daily Return Correlation Between Traders')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    return correlation

def plot_portfolio_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, file_name):
    """포트폴리오 비교 그래프 생성"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 수익률 그래프
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, timeline_df['Return_Pct'], label=f'{trader_id}')
    
    # 포트폴리오 수익률 그래프
    for i, (timeline_df, portfolio_name) in enumerate(zip(portfolio_timelines, portfolio_names)):
        plt.plot(timeline_df.index, timeline_df['Return_Pct'], label=portfolio_name, linestyle='--')
    
    plt.title('Trader and Portfolio Return Comparison')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name))
    plt.close()

def plot_drawdowns_comparison(trader_timelines, portfolio_timelines, trader_ids, portfolio_names, file_name):
    """드로다운 비교 그래프 생성"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 드로다운 그래프
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, -timeline_df['Drawdown'], label=f'{trader_id}')
    
    # 포트폴리오 드로다운 그래프
    for i, (timeline_df, portfolio_name) in enumerate(zip(portfolio_timelines, portfolio_names)):
        plt.plot(timeline_df.index, -timeline_df['Drawdown'], label=portfolio_name, linestyle='--')
    
    plt.title('Trader and Portfolio Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name))
    plt.close()

def main():
    # 트레이더 설정
    trader_configs = [
        {'id': 'hummusXBT', 'period': 'period2', 'initial_capital': 2000000, 'apply_slippage': False},
        {'id': 'Panic', 'period': 'period2', 'initial_capital': 1000000, 'apply_slippage': True},
        {'id': 'Cyborg0578', 'period': None, 'initial_capital': 1000000, 'apply_slippage': True}
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
    
    # 상관관계 분석
    correlation = analyze_correlation(trader_timelines, trader_ids)
    
    # 다양한 포트폴리오 구성 테스트
    portfolio_timelines = []
    portfolio_names = []
    portfolio_weights = []
    
    # 1. 균등 가중치 포트폴리오 (33:33:33)
    equal_weights = [1/3, 1/3, 1/3]
    equal_combined = simulate_combined_portfolio(trader_timelines, equal_weights)
    portfolio_timelines.append(equal_combined)
    portfolio_names.append('Equal Portfolio (33:33:33)')
    portfolio_weights.append(equal_weights)
    
    # 2. hummusXBT 중심 포트폴리오 (60:20:20)
    hummus_focused_weights = [0.6, 0.2, 0.2]
    hummus_focused_combined = simulate_combined_portfolio(trader_timelines, hummus_focused_weights)
    portfolio_timelines.append(hummus_focused_combined)
    portfolio_names.append('Hummus Focused (60:20:20)')
    portfolio_weights.append(hummus_focused_weights)
    
    # 3. Panic 중심 포트폴리오 (20:60:20)
    panic_focused_weights = [0.2, 0.6, 0.2]
    panic_focused_combined = simulate_combined_portfolio(trader_timelines, panic_focused_weights)
    portfolio_timelines.append(panic_focused_combined)
    portfolio_names.append('Panic Focused (20:60:20)')
    portfolio_weights.append(panic_focused_weights)
    
    # 4. Cyborg 중심 포트폴리오 (20:20:60)
    cyborg_focused_weights = [0.2, 0.2, 0.6]
    cyborg_focused_combined = simulate_combined_portfolio(trader_timelines, cyborg_focused_weights)
    portfolio_timelines.append(cyborg_focused_combined)
    portfolio_names.append('Cyborg Focused (20:20:60)')
    portfolio_weights.append(cyborg_focused_weights)
    
    # 5. hummusXBT & Cyborg 중심 포트폴리오 (40:40:20)
    hummus_cyborg_weights = [0.4, 0.2, 0.4]
    hummus_cyborg_combined = simulate_combined_portfolio(trader_timelines, hummus_cyborg_weights)
    portfolio_timelines.append(hummus_cyborg_combined)
    portfolio_names.append('Hummus-Cyborg (40:20:40)')
    portfolio_weights.append(hummus_cyborg_weights)
    
    # 6. hummusXBT 70%, Cyborg0578 20%, Panic 10% 포트폴리오
    hummus_heavy_weights = [0.7, 0.1, 0.2]
    hummus_heavy_combined = simulate_combined_portfolio(trader_timelines, hummus_heavy_weights)
    portfolio_timelines.append(hummus_heavy_combined)
    portfolio_names.append('Hummus-Heavy (70:10:20)')
    portfolio_weights.append(hummus_heavy_weights)
    
    # 7. 현금 50% + 트레이더 50% (hummus 60%, cyborg 25%, panic 15%)
    # 현금 50%는 각 트레이더의 가중치를 절반으로 줄여서 구현
    cash_hedge_weights = [0.3, 0.075, 0.125]  # 50% 현금 + 50% 트레이더 (60:15:25)
    cash_hedge_combined = simulate_combined_portfolio(trader_timelines, cash_hedge_weights)
    portfolio_timelines.append(cash_hedge_combined)
    portfolio_names.append('Cash-Hedge (50% Cash + 30:7.5:12.5)')
    portfolio_weights.append(cash_hedge_weights)
    
    # 8. hummusXBT 50%, Cyborg0578 50%, Panic 0%
    hummus_cyborg_only_equal_weights = [0.5, 0.0, 0.5]
    hummus_cyborg_only_equal = simulate_combined_portfolio(trader_timelines, hummus_cyborg_only_equal_weights)
    portfolio_timelines.append(hummus_cyborg_only_equal)
    portfolio_names.append('Hummus-Cyborg Only (50:0:50)')
    portfolio_weights.append(hummus_cyborg_only_equal_weights)
    
    # 9. hummusXBT 70%, Cyborg0578 30%, Panic 0%
    hummus_cyborg_only_hummus_heavy_weights = [0.7, 0.0, 0.3]
    hummus_cyborg_only_hummus_heavy = simulate_combined_portfolio(trader_timelines, hummus_cyborg_only_hummus_heavy_weights)
    portfolio_timelines.append(hummus_cyborg_only_hummus_heavy)
    portfolio_names.append('Hummus-Cyborg Only (70:0:30)')
    portfolio_weights.append(hummus_cyborg_only_hummus_heavy_weights)
    
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
        weights_str = f'{weights[0]:.2f}:{weights[1]:.2f}:{weights[2]:.2f}'
        metrics['Weights'] = weights_str
        performance_summary.append(metrics)
    
    # 성과 요약 데이터프레임 생성
    summary_df = pd.DataFrame(performance_summary)
    
    # 결과 출력
    print("===== Trader Correlation =====")
    print(correlation)
    
    print("\n===== Portfolio Performance Summary =====")
    print(summary_df[['Portfolio', 'Weights', 'Total Return (%)', 'Annualized Return (%)', 
                     'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Calmar Ratio', 
                     'Win Rate (%)', 'Trading Days']].to_string(index=False))
    
    # 결과 저장
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'performance_summary.csv'), index=False)
    
    print("\n===== Analysis Complete =====")
    print(f"Results saved to {RESULTS_DIR} folder.")

if __name__ == "__main__":
    main()
