#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# 결과 저장 디렉토리 생성
RESULTS_DIR = 'analysis_results/timeline'
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
            trade_size = abs(trade.get('Size_USD', 0))
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
            trader_initial = timeline_df['Portfolio_Value'].iloc[0]
            resampled_df = timeline_df.reindex(all_indices, method='ffill')
            relative_return = resampled_df['Portfolio_Value'] / trader_initial - 1
            
            # 초기 포트폴리오 가치에 가중치와 상대적 수익률 적용
            # 초기값을 유지하기 위해 처음에는 초기 포트폴리오 가치를 비백업
            if i == 0:
                combined_df['Portfolio_Value_Backup'] = combined_df['Portfolio_Value'].copy()
                combined_df['Portfolio_Value'] = 0
            
            # 각 트레이더의 기여도 계산
            trader_contribution = initial_portfolio_value * weight * (1 + relative_return)
            combined_df['Portfolio_Value'] += trader_contribution
    
    # 수익률 및 드로다운 계산
    combined_df['Return_Pct'] = (combined_df['Portfolio_Value'] / initial_portfolio_value - 1) * 100
    combined_df['Peak'] = combined_df['Portfolio_Value'].cummax()
    combined_df['Drawdown'] = (combined_df['Peak'] - combined_df['Portfolio_Value']) / combined_df['Peak'] * 100
    
    # 임시 컬럼 삭제
    if 'Portfolio_Value_Backup' in combined_df.columns:
        combined_df.drop('Portfolio_Value_Backup', axis=1, inplace=True)
    
    return combined_df

def calculate_concurrent_trades(trader_data_list, trader_ids):
    """동시 진행 중인 거래 분석"""
    # 모든 거래의 시작/종료 시간 추출
    all_trades = []
    
    for df, trader_id in zip(trader_data_list, trader_ids):
        for _, trade in df.iterrows():
            all_trades.append({
                'Trader': trader_id,
                'Open_Time': trade['Open_Time_KST'],
                'Close_Time': trade['Close_Time_KST'],
                'Symbol': trade['Symbol'],
                'Direction': trade['Direction'],
                'Size_USDT': trade['Size_USDT_Abs'],
                'PnL': trade['PnL_Numeric']
            })
    
    trades_df = pd.DataFrame(all_trades)
    
    # 시간별 진행 중인 거래 수 계산
    start_date = trades_df['Open_Time'].min()
    end_date = trades_df['Close_Time'].max()
    timeline = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    concurrent_trades = pd.DataFrame(index=timeline)
    concurrent_trades['Total_Trades'] = 0
    concurrent_trades['Total_Capital_Used'] = 0
    
    for trader_id in trader_ids:
        concurrent_trades[f'{trader_id}_Trades'] = 0
        concurrent_trades[f'{trader_id}_Capital'] = 0
    
    # 각 시간대별 진행 중인 거래 계산
    for _, trade in trades_df.iterrows():
        open_time = trade['Open_Time']
        close_time = trade['Close_Time']
        trader = trade['Trader']
        size = trade['Size_USDT']
        
        # 해당 거래가 활성화된 시간대 찾기
        mask = (concurrent_trades.index >= open_time) & (concurrent_trades.index <= close_time)
        
        # 거래 수 및 사용 자본 업데이트
        concurrent_trades.loc[mask, 'Total_Trades'] += 1
        concurrent_trades.loc[mask, 'Total_Capital_Used'] += size
        concurrent_trades.loc[mask, f'{trader}_Trades'] += 1
        concurrent_trades.loc[mask, f'{trader}_Capital'] += size
    
    return concurrent_trades

def plot_portfolio_values(trader_timelines, combined_timeline, trader_ids, weights, file_name='portfolio_values.png'):
    """포트폴리오 가치 변화 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 포트폴리오 가치
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, timeline_df['Portfolio_Value'], 
                 label=f'{trader_id} (Weight: {weights[i]:.2f})', alpha=0.7)
    
    # 결합 포트폴리오 가치
    plt.plot(combined_timeline.index, combined_timeline['Portfolio_Value'], 
             label='Combined Portfolio', linewidth=2, color='black')
    
    plt.title('Portfolio Value Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value (USD)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_drawdowns_timeline(trader_timelines, combined_timeline, trader_ids, weights, file_name='drawdowns_timeline.png'):
    """시간별 드로다운 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 드로다운
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        plt.plot(timeline_df.index, timeline_df['Drawdown'], 
                 label=f'{trader_id} (Weight: {weights[i]:.2f})', alpha=0.7)
    
    # 결합 포트폴리오 드로다운
    plt.plot(combined_timeline.index, combined_timeline['Drawdown'], 
             label='Combined Portfolio', linewidth=2, color='black')
    
    plt.title('Drawdown Over Time (%)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # 드로다운을 아래로 표시
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_concurrent_trades(concurrent_trades, trader_ids, file_name='concurrent_trades.png'):
    """동시 진행 중인 거래 수 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 거래 수
    for trader_id in trader_ids:
        plt.plot(concurrent_trades.index, concurrent_trades[f'{trader_id}_Trades'], 
                 label=f'{trader_id} Trades', alpha=0.7)
    
    # 전체 거래 수
    plt.plot(concurrent_trades.index, concurrent_trades['Total_Trades'], 
             label='Total Concurrent Trades', linewidth=2, color='black')
    
    plt.title('Number of Concurrent Trades Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Trades', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def plot_capital_usage(concurrent_trades, trader_ids, file_name='capital_usage.png'):
    """자본 사용량 그래프"""
    plt.figure(figsize=(14, 8))
    
    # 개별 트레이더 자본 사용량
    for trader_id in trader_ids:
        plt.plot(concurrent_trades.index, concurrent_trades[f'{trader_id}_Capital'], 
                 label=f'{trader_id} Capital', alpha=0.7)
    
    # 전체 자본 사용량
    plt.plot(concurrent_trades.index, concurrent_trades['Total_Capital_Used'], 
             label='Total Capital Used', linewidth=2, color='black')
    
    plt.title('Capital Usage Over Time (USD)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Capital Used (USD)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()

def analyze_correlation(trader_timelines, trader_ids, file_name='return_correlation.png'):
    """트레이더 간 수익률 상관관계 분석"""
    # 모든 타임라인의 인덱스 통합
    all_indices = set()
    for timeline_df in trader_timelines:
        all_indices.update(timeline_df.index)
    
    all_indices = sorted(all_indices)
    
    # 일간 수익률 데이터프레임 생성
    returns_df = pd.DataFrame(index=all_indices)
    
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        resampled_df = timeline_df.reindex(all_indices, method='ffill')
        # 일간 변화율 계산
        returns_df[trader_id] = resampled_df['Portfolio_Value'].pct_change() * 100
    
    # NaN 제거
    returns_df = returns_df.dropna()
    
    # 상관관계 계산
    correlation = returns_df.corr()
    
    # 상관관계 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
    plt.title('Correlation of Daily Returns Between Traders', fontsize=16)
    
    # 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_name), dpi=300)
    plt.close()
    
    return correlation

def main():
    # 트레이더 설정
    trader_configs = [
        {'id': 'hummusXBT', 'period': 'period2', 'initial_capital': 2000000, 'apply_slippage': True},
        {'id': 'RebelOfBabylon', 'period': None, 'initial_capital': 1000000, 'apply_slippage': True},
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
        # 슬리피지 적용 여부 확인
        apply_slippage = config.get('apply_slippage', False)
        timeline_df = create_timeline_data(trader_data[trader_id], config['initial_capital'], apply_slippage=apply_slippage, slippage_pct=0.05)
        trader_timelines.append(timeline_df)
    
    # 최적화된 가중치 (이전 분석 결과에서 가져옴)
    optimized_weights = [0.2532, 0.3232, 0.4236]  # hummusXBT, RebelOfBabylon, Panic
    
    # 균등 가중치
    equal_weights = [1/3, 1/3, 1/3]
    
    # hummusXBT 중심 가중치 (0.5, 0.25, 0.25)
    hummus_focused_weights = [0.5, 0.25, 0.25]
    
    # 결합 포트폴리오 시뮬레이션 (고정된 초기 자본 1,000,000 달러 사용)
    optimized_combined = simulate_combined_portfolio(trader_timelines, optimized_weights, fixed_initial_capital=1000000)
    equal_combined = simulate_combined_portfolio(trader_timelines, equal_weights, fixed_initial_capital=1000000)
    hummus_focused_combined = simulate_combined_portfolio(trader_timelines, hummus_focused_weights, fixed_initial_capital=1000000)
    
    # 동시 진행 중인 거래 분석
    concurrent_trades = calculate_concurrent_trades(
        [trader_data[trader_id] for trader_id in trader_ids], 
        trader_ids
    )
    
    # 그래프 생성
    plot_portfolio_values(trader_timelines, optimized_combined, trader_ids, optimized_weights, 'optimized_portfolio_values.png')
    plot_portfolio_values(trader_timelines, equal_combined, trader_ids, equal_weights, 'equal_portfolio_values.png')
    plot_portfolio_values(trader_timelines, hummus_focused_combined, trader_ids, hummus_focused_weights, 'hummus_focused_portfolio_values.png')
    
    plot_drawdowns_timeline(trader_timelines, optimized_combined, trader_ids, optimized_weights, 'optimized_drawdowns_timeline.png')
    plot_drawdowns_timeline(trader_timelines, equal_combined, trader_ids, equal_weights, 'equal_drawdowns_timeline.png')
    plot_drawdowns_timeline(trader_timelines, hummus_focused_combined, trader_ids, hummus_focused_weights, 'hummus_focused_drawdowns_timeline.png')
    
    plot_concurrent_trades(concurrent_trades, trader_ids)
    plot_capital_usage(concurrent_trades, trader_ids)
    
    # 상관관계 분석
    correlation = analyze_correlation(trader_timelines, trader_ids)
    
    # 결과 요약
    print("===== 트레이더 간 상관관계 =====")
    print(correlation)
    
    print("\n===== 최대 드로다운 비교 =====")
    for i, (timeline_df, trader_id) in enumerate(zip(trader_timelines, trader_ids)):
        max_dd = timeline_df['Drawdown'].max()
        print(f"{trader_id}: {max_dd:.2f}%")
    
    print(f"최적화 가중치 포트폴리오: {optimized_combined['Drawdown'].max():.2f}%")
    print(f"균등 가중치 포트폴리오: {equal_combined['Drawdown'].max():.2f}%")
    print(f"hummusXBT 중심 포트폴리오: {hummus_focused_combined['Drawdown'].max():.2f}%")
    
    print("\n===== 자본 사용 요약 =====")
    max_capital = concurrent_trades['Total_Capital_Used'].max()
    avg_capital = concurrent_trades['Total_Capital_Used'].mean()
    print(f"최대 동시 사용 자본: ${max_capital:,.2f}")
    print(f"평균 사용 자본: ${avg_capital:,.2f}")
    
    for trader_id in trader_ids:
        max_trader_capital = concurrent_trades[f'{trader_id}_Capital'].max()
        print(f"{trader_id} 최대 사용 자본: ${max_trader_capital:,.2f}")
    
    # 연율화된 수익률 계산
    annualized_returns = []
    trading_days = []
    
    # 각 트레이더의 거래 기간 계산
    for i, timeline_df in enumerate(trader_timelines):
        start_date = timeline_df.index[0]
        end_date = timeline_df.index[-1]
        days = (end_date - start_date).days
        trading_days.append(days)
        
        # 연율화된 수익률 계산 (CAGR: Compound Annual Growth Rate)
        final_return = timeline_df['Return_Pct'].iloc[-1] / 100  # 백분율에서 소수점으로 변환
        annualized_return = ((1 + final_return) ** (365 / days) - 1) * 100 if days > 0 else 0
        annualized_returns.append(annualized_return)
    
    # 포트폴리오의 거래 기간은 가장 긴 트레이더의 기간을 사용
    max_days = max(trading_days)
    
    # 포트폴리오의 연율화된 수익률 계산
    optimized_return = optimized_combined['Return_Pct'].iloc[-1] / 100
    equal_return = equal_combined['Return_Pct'].iloc[-1] / 100
    hummus_focused_return = hummus_focused_combined['Return_Pct'].iloc[-1] / 100
    
    optimized_annualized = ((1 + optimized_return) ** (365 / max_days) - 1) * 100 if max_days > 0 else 0
    equal_annualized = ((1 + equal_return) ** (365 / max_days) - 1) * 100 if max_days > 0 else 0
    hummus_focused_annualized = ((1 + hummus_focused_return) ** (365 / max_days) - 1) * 100 if max_days > 0 else 0
    
    annualized_returns.extend([optimized_annualized, equal_annualized, hummus_focused_annualized])
    
    # 결과 저장
    summary = {
        'Trader': trader_ids + ['Optimized Portfolio', 'Equal Portfolio', 'Hummus Focused Portfolio'],
        'Max Drawdown (%)': [timeline_df['Drawdown'].max() for timeline_df in trader_timelines] + 
                          [optimized_combined['Drawdown'].max(), equal_combined['Drawdown'].max(), hummus_focused_combined['Drawdown'].max()],
        'Final Return (%)': [timeline_df['Return_Pct'].iloc[-1] for timeline_df in trader_timelines] + 
                          [optimized_combined['Return_Pct'].iloc[-1], equal_combined['Return_Pct'].iloc[-1], hummus_focused_combined['Return_Pct'].iloc[-1]],
        'Annualized Return (%)': annualized_returns,
        'Trading Days': trading_days + [max_days, max_days, max_days],
        'Max Capital Used': [concurrent_trades[f'{trader_id}_Capital'].max() for trader_id in trader_ids] + 
                          [max_capital, max_capital, max_capital]
    }
    
    # 최종 수익률 출력
    print("\n===== 최종 수익률 비교 =====")
    for i, trader_id in enumerate(trader_ids):
        print(f"{trader_id}: {summary['Final Return (%)'][i]:.2f}% (연율화: {summary['Annualized Return (%)'][i]:.2f}%)")
    print(f"최적화 가중치 포트폴리오: {optimized_combined['Return_Pct'].iloc[-1]:.2f}% (연율화: {optimized_annualized:.2f}%)")
    print(f"균등 가중치 포트폴리오: {equal_combined['Return_Pct'].iloc[-1]:.2f}% (연율화: {equal_annualized:.2f}%)")
    print(f"hummusXBT 중심 포트폴리오: {hummus_focused_combined['Return_Pct'].iloc[-1]:.2f}% (연율화: {hummus_focused_annualized:.2f}%)")
    
    print("\n===== 거래 기간 =====")
    for i, trader_id in enumerate(trader_ids):
        print(f"{trader_id}: {summary['Trading Days'][i]} 일")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'timeline_analysis_summary.csv'), index=False)
    
    print("\n===== 분석 완료 =====")
    print(f"결과가 {RESULTS_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
