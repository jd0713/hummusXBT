import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from scipy import stats
import itertools

# 트레이더 데이터 로드 함수
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

# 타임라인 데이터 생성 함수
def create_timeline_data(df, initial_capital, freq='1h', apply_slippage=False, slippage_pct=0.05):
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
            timeline_df.iloc[close_idx:, 0] = timeline_df.iloc[close_idx:, 0].astype(float) + pnl
    
    # 누적 수익률 계산
    timeline_df['Return_Pct'] = (timeline_df['Portfolio_Value'] / initial_capital - 1) * 100
    
    # 최대 드로다운 계산
    timeline_df['Peak'] = timeline_df['Portfolio_Value'].cummax()
    timeline_df['Drawdown'] = (timeline_df['Peak'] - timeline_df['Portfolio_Value']) / timeline_df['Peak'] * 100
    
    # 일별 수익률 계산 (성과 지표 계산용)
    timeline_df['Daily_Return'] = timeline_df['Portfolio_Value'].pct_change()
    
    return timeline_df

# 포트폴리오 시뮬레이션 함수
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
    
    # 수익률 계산
    combined_df['Return_Pct'] = (combined_df['Portfolio_Value'] / initial_portfolio_value - 1) * 100
    
    # 최대 드로다운 계산
    combined_df['Peak'] = combined_df['Portfolio_Value'].cummax()
    combined_df['Drawdown'] = (combined_df['Peak'] - combined_df['Portfolio_Value']) / combined_df['Peak'] * 100
    
    return combined_df

# 성과 지표 계산 함수
def calculate_performance_metrics(timeline_df, trading_days=None):
    # 마지막 포트폴리오 가치와 초기 가치 가져오기
    final_value = timeline_df['Portfolio_Value'].iloc[-1]
    initial_value = timeline_df['Portfolio_Value'].iloc[0]
    
    # 총 수익률
    total_return_pct = (final_value / initial_value - 1) * 100
    
    # 거래일 계산
    if trading_days is None:
        # 거래일 수 계산 (주말 제외)
        trading_days = len(timeline_df.index.normalize().unique())
    
    # 연간 수익률
    annualized_return = ((1 + total_return_pct/100) ** (252/trading_days) - 1) * 100
    
    # 최대 낙폭
    max_drawdown = abs(timeline_df['Drawdown'].max())
    
    # 일별 수익률 계산
    daily_returns = timeline_df['Portfolio_Value'].resample('D').last().pct_change().dropna()
    
    # 변동성 (연간화)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # 샤프 비율 (무위험 이자율 0% 가정)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    # 칼마 비율
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
    
    # 승률 계산
    win_rate = (daily_returns > 0).mean() * 100
    
    return {
        'Total Return (%)': total_return_pct,
        'Annualized Return (%)': annualized_return,
        'Max Drawdown (%)': max_drawdown,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Win Rate (%)': win_rate,
        'Trading Days': trading_days
    }

# 최적화를 위한 목적 함수 (샤프 비율 최대화)
def negative_sharpe_ratio(weights, trader_timelines):
    # 가중치 정규화
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # 포트폴리오 시뮬레이션
    portfolio = simulate_combined_portfolio(trader_timelines, weights)
    
    # 성과 지표 계산
    metrics = calculate_performance_metrics(portfolio)
    
    # 샤프 비율의 음수 반환 (최소화 문제로 변환)
    return -metrics['Sharpe Ratio']

# 최적화를 위한 목적 함수 (칼마 비율 최대화)
def negative_calmar_ratio(weights, trader_timelines):
    # 가중치 정규화
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # 포트폴리오 시뮬레이션
    portfolio = simulate_combined_portfolio(trader_timelines, weights)
    
    # 성과 지표 계산
    metrics = calculate_performance_metrics(portfolio)
    
    # 칼마 비율의 음수 반환 (최소화 문제로 변환)
    return -metrics['Calmar Ratio']

# 메인 함수
def main():
    # 트레이더 ID 설정
    trader_ids = ['hummusXBT', 'Panic', 'Cyborg0578']
    
    # 초기 자본금 설정
    initial_capitals = [2000000, 1000000, 1000000]
    
    # 결과 저장 디렉토리 생성
    results_dir = 'analysis_results/optimized_portfolio'
    os.makedirs(results_dir, exist_ok=True)
    
    # 트레이더 데이터 로드 및 타임라인 생성
    trader_timelines = []
    for trader_id, initial_capital in zip(trader_ids, initial_capitals):
        try:
            df = load_trader_data(trader_id)
            timeline_df = create_timeline_data(df, initial_capital)
            trader_timelines.append(timeline_df)
            print(f"Loaded data for {trader_id}")
        except Exception as e:
            print(f"Error loading data for {trader_id}: {e}")
            return
    
    # 상관관계 계산
    correlation_matrix = pd.DataFrame(index=trader_ids, columns=trader_ids)
    for i, trader1 in enumerate(trader_ids):
        for j, trader2 in enumerate(trader_ids):
            # 공통 날짜 찾기
            common_dates = trader_timelines[i].index.intersection(trader_timelines[j].index)
            
            # 상관관계 계산
            if len(common_dates) > 0:
                corr = trader_timelines[i].loc[common_dates, 'Return_Pct'].corr(
                    trader_timelines[j].loc[common_dates, 'Return_Pct']
                )
                correlation_matrix.loc[trader1, trader2] = corr
            else:
                correlation_matrix.loc[trader1, trader2] = np.nan
    
    print("===== Trader Correlation =====")
    print(correlation_matrix)
    
    # 최적화 제약 조건 설정
    n_traders = len(trader_ids)
    bounds = tuple((0.0, 1.0) for _ in range(n_traders))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 초기 가중치 (동일 가중치)
    initial_weights = np.array([1/n_traders] * n_traders)
    
    # 샤프 비율 최적화
    sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(trader_timelines,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    sharpe_optimal_weights = sharpe_result['x']
    sharpe_optimal_weights = sharpe_optimal_weights / np.sum(sharpe_optimal_weights)
    
    # 칼마 비율 최적화
    calmar_result = minimize(
        negative_calmar_ratio,
        initial_weights,
        args=(trader_timelines,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    calmar_optimal_weights = calmar_result['x']
    calmar_optimal_weights = calmar_optimal_weights / np.sum(calmar_optimal_weights)
    
    # 최적 포트폴리오 시뮬레이션
    sharpe_optimal_portfolio = simulate_combined_portfolio(trader_timelines, sharpe_optimal_weights)
    calmar_optimal_portfolio = simulate_combined_portfolio(trader_timelines, calmar_optimal_weights)
    
    # 성과 지표 계산
    sharpe_metrics = calculate_performance_metrics(sharpe_optimal_portfolio)
    calmar_metrics = calculate_performance_metrics(calmar_optimal_portfolio)
    
    # 개별 트레이더 성과 지표 계산
    trader_metrics = []
    for i, trader_id in enumerate(trader_ids):
        metrics = calculate_performance_metrics(trader_timelines[i])
        metrics['Portfolio'] = trader_id
        metrics['Weights'] = f"{trader_id} Only"
        trader_metrics.append(metrics)
    
    # 최적 포트폴리오 성과 지표 추가
    sharpe_metrics['Portfolio'] = 'Sharpe Optimal Portfolio'
    sharpe_metrics['Weights'] = ':'.join([f"{w:.2f}" for w in sharpe_optimal_weights])
    
    calmar_metrics['Portfolio'] = 'Calmar Optimal Portfolio'
    calmar_metrics['Weights'] = ':'.join([f"{w:.2f}" for w in calmar_optimal_weights])
    
    # 결과 데이터프레임 생성
    all_metrics = trader_metrics + [sharpe_metrics, calmar_metrics]
    results_df = pd.DataFrame(all_metrics)
    
    # 결과 출력
    print("\n===== Portfolio Performance Summary =====")
    print(results_df[['Portfolio', 'Weights', 'Total Return (%)', 'Annualized Return (%)', 
                      'Max Drawdown (%)', 'Volatility (%)', 'Sharpe Ratio', 'Calmar Ratio', 
                      'Win Rate (%)', 'Trading Days']])
    
    # 최적 가중치 출력
    print("\n===== Optimal Weights =====")
    print("Sharpe Ratio Optimal Weights:")
    for trader_id, weight in zip(trader_ids, sharpe_optimal_weights):
        print(f"{trader_id}: {weight:.4f} ({weight*100:.2f}%)")
    
    print("\nCalmar Ratio Optimal Weights:")
    for trader_id, weight in zip(trader_ids, calmar_optimal_weights):
        print(f"{trader_id}: {weight:.4f} ({weight*100:.2f}%)")
    
    # 결과 저장
    results_df.to_csv(f"{results_dir}/performance_summary.csv", index=False)
    
    # 포트폴리오 수익률 그래프 생성
    plt.figure(figsize=(12, 6))
    plt.plot(sharpe_optimal_portfolio.index, sharpe_optimal_portfolio['Return_Pct'], label='Sharpe Optimal')
    plt.plot(calmar_optimal_portfolio.index, calmar_optimal_portfolio['Return_Pct'], label='Calmar Optimal')
    
    for i, trader_id in enumerate(trader_ids):
        plt.plot(trader_timelines[i].index, trader_timelines[i]['Return_Pct'], label=trader_id)
    
    plt.title('Portfolio Return Comparison')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/optimal_portfolio_comparison.png", dpi=300, bbox_inches='tight')
    
    # 포트폴리오 낙폭 그래프 생성
    plt.figure(figsize=(12, 6))
    plt.plot(sharpe_optimal_portfolio.index, sharpe_optimal_portfolio['Drawdown'], label='Sharpe Optimal')
    plt.plot(calmar_optimal_portfolio.index, calmar_optimal_portfolio['Drawdown'], label='Calmar Optimal')
    
    for i, trader_id in enumerate(trader_ids):
        plt.plot(trader_timelines[i].index, trader_timelines[i]['Drawdown'], label=trader_id)
    
    plt.title('Portfolio Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/optimal_drawdowns_comparison.png", dpi=300, bbox_inches='tight')
    
    # 가중치 시각화
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(trader_ids))
    width = 0.35
    
    plt.bar(x - width/2, sharpe_optimal_weights, width, label='Sharpe Optimal')
    plt.bar(x + width/2, calmar_optimal_weights, width, label='Calmar Optimal')
    
    plt.xlabel('Trader')
    plt.ylabel('Weight')
    plt.title('Optimal Portfolio Weights')
    plt.xticks(x, trader_ids)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(f"{results_dir}/optimal_weights.png", dpi=300, bbox_inches='tight')
    
    print(f"\n===== Analysis Complete =====")
    print(f"Results saved to {results_dir} folder.")

if __name__ == "__main__":
    main()
