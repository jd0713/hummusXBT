#!/usr/bin/env python3
"""
슬리피지 영향 분석 스크립트
다양한 슬리피지 비율(0.05%, 0.1%, 0.15%)이 거래 성과에 미치는 영향을 분석합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# 상수 정의
INITIAL_CAPITAL_PERIOD1 = 100000  # 기간 1 초기 자본
INITIAL_CAPITAL_PERIOD2 = 2000000  # 기간 2 초기 자본
SLIPPAGE_RATES = [0.0005, 0.001, 0.0015]  # 0.05%, 0.1%, 0.15%

# 결과 디렉토리 설정
RESULTS_DIR = 'analysis_results/slippage_analysis'
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_date(date_str):
    """날짜 문자열을 datetime 객체로 변환"""
    return datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M %Z')

def parse_number(num_str):
    """쉼표와 통화 기호가 포함된 숫자 문자열을 float로 변환"""
    if isinstance(num_str, str):
        return float(num_str.replace(',', '').replace(' USDT', ''))
    return num_str

def apply_slippage(df, slippage_rate):
    """
    거래 데이터에 슬리피지를 적용
    
    슬리피지는 진입가와 청산가 모두에 적용됩니다:
    - Long 포지션: 진입가는 더 높게, 청산가는 더 낮게 조정
    - Short 포지션: 진입가는 더 낮게, 청산가는 더 높게 조정
    
    이로 인해 실현 PnL이 감소하게 됩니다.
    """
    df_copy = df.copy()
    
    # 슬리피지 적용된 PnL 계산
    for idx, row in df_copy.iterrows():
        direction = row['Direction']
        entry_price = row['Entry Price']
        max_size = row['Max Size']
        realized_pnl = row['Realized PnL']
        
        # 슬리피지 금액 계산 (진입과 청산 모두에 적용)
        slippage_amount = 2 * entry_price * max_size * slippage_rate
        
        # PnL에서 슬리피지 금액 차감
        df_copy.at[idx, 'Slippage_Adjusted_PnL'] = realized_pnl - slippage_amount
        
        # 슬리피지 조정된 수익률 계산
        max_size_usdt = abs(row['Max Size USDT'])
        df_copy.at[idx, 'Slippage_Adjusted_PnL_Pct'] = (df_copy.at[idx, 'Slippage_Adjusted_PnL'] / max_size_usdt) * 100
    
    return df_copy

def calculate_performance_metrics(df, initial_capital, is_original=False):
    """슬리피지가 적용된 데이터로 성과 지표 계산"""
    # 날짜 기준 정렬
    df = df.sort_values('Close Time')
    
    # 원본 데이터와 슬리피지 적용 데이터 구분
    if is_original:
        pnl_column = 'Realized PnL'
    else:
        pnl_column = 'Slippage_Adjusted_PnL'
    
    # 누적 PnL 계산
    df['Cumulative_PnL'] = df[pnl_column].cumsum()
    
    # 자산 가치 계산
    df['Asset_Value'] = initial_capital + df['Cumulative_PnL']
    
    # 최대 낙폭(MDD) 계산
    df['Peak'] = df['Asset_Value'].cummax()
    df['Drawdown'] = (df['Asset_Value'] - df['Peak']) / df['Peak']
    mdd = df['Drawdown'].min() * 100
    
    # 총 수익 및 수익률 계산
    total_pnl = df[pnl_column].sum()
    total_return_pct = (total_pnl / initial_capital) * 100
    
    # 거래 일수 계산
    start_date = df['Close Time'].min()
    end_date = df['Close Time'].max()
    trading_days = (end_date - start_date).days
    
    # 연율화 수익률 계산
    annualized_return = ((1 + total_return_pct/100) ** (365/trading_days) - 1) * 100
    
    # 승률 계산
    win_count = len(df[df[pnl_column] > 0])
    win_rate = (win_count / len(df)) * 100
    
    # 일평균 수익 및 수익률
    daily_return = total_pnl / trading_days
    daily_return_pct = total_return_pct / trading_days
    
    # 샴프 비율 계산 (간소화된 버전)
    if len(df) > 1:
        # 일별 수익률 계산을 위한 리샘플링
        df['Date'] = df['Close Time'].dt.date
        daily_returns = df.groupby('Date')[pnl_column].sum() / initial_capital
        
        # 샴프 비율 계산 (무위험 수익률 = 0 가정)
        sharpe_ratio = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'Initial Capital': initial_capital,
        'Total PnL': total_pnl,
        'Total Return (%)': total_return_pct,
        'Annualized Return (%)': annualized_return,
        'MDD (%)': mdd,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate (%)': win_rate,
        'Trading Days': trading_days,
        'Daily Return': daily_return,
        'Daily Return (%)': daily_return_pct,
        'Asset Value Series': df[['Close Time', 'Asset_Value']].copy()
    }

def plot_asset_growth_comparison(metrics_list, period_name, slippage_rates):
    """Plot asset growth comparison with different slippage rates"""
    plt.figure(figsize=(14, 8))
    
    colors = ['green', 'blue', 'orange', 'red']
    labels = ['Original (No Slippage)'] + [f'Slippage {rate*100:.2f}%' for rate in slippage_rates]
    
    # Convert Korean period name to English
    if period_name == "기간 1":
        period_name_en = "Period 1"
    elif period_name == "기간 2":
        period_name_en = "Period 2"
    else:
        period_name_en = period_name
    
    for i, metrics in enumerate(metrics_list):
        asset_series = metrics['Asset Value Series']
        plt.plot(asset_series['Close Time'], asset_series['Asset_Value'], 
                 marker='', linestyle='-', color=colors[i], label=labels[i])
    
    plt.title(f'{period_name_en} Slippage Impact Analysis: Asset Growth Comparison', fontsize=16)
    plt.ylabel('Asset Value (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    
    # Display return percentages
    for i, metrics in enumerate(metrics_list):
        total_return = metrics['Total Return (%)']
        plt.annotate(f'{labels[i]}: {total_return:.2f}%', 
                    xy=(0.02, 0.95 - i*0.05), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{period_name_en.lower().replace(" ", "_")}_asset_growth_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name_en} slippage impact analysis graph has been saved.")

def create_performance_comparison_table(metrics_list, period_name, slippage_rates):
    """Create performance comparison table for different slippage rates"""
    headers = ['Metric', 'Original', 'Slippage 0.05%', 'Slippage 0.1%', 'Slippage 0.15%']
    
    # Convert Korean period name to English
    if period_name == "기간 1":
        period_name_en = "Period 1"
    elif period_name == "기간 2":
        period_name_en = "Period 2"
    else:
        period_name_en = period_name
    
    rows = [
        ['Total PnL (USD)', f"{metrics_list[0]['Total PnL']:.2f}"] + [f"{metrics['Total PnL']:.2f}" for metrics in metrics_list[1:]],
        ['Total Return (%)', f"{metrics_list[0]['Total Return (%)']:.2f}"] + [f"{metrics['Total Return (%)']:.2f}" for metrics in metrics_list[1:]],
        ['Annualized Return (%)', f"{metrics_list[0]['Annualized Return (%)']:.2f}"] + [f"{metrics['Annualized Return (%)']:.2f}" for metrics in metrics_list[1:]],
        ['MDD (%)', f"{metrics_list[0]['MDD (%)']:.2f}"] + [f"{metrics['MDD (%)']:.2f}" for metrics in metrics_list[1:]],
        ['Sharpe Ratio', f"{metrics_list[0]['Sharpe Ratio']:.2f}"] + [f"{metrics['Sharpe Ratio']:.2f}" for metrics in metrics_list[1:]],
        ['Win Rate (%)', f"{metrics_list[0]['Win Rate (%)']:.2f}"] + [f"{metrics['Win Rate (%)']:.2f}" for metrics in metrics_list[1:]]
    ]
    
    # Save table to file
    with open(os.path.join(RESULTS_DIR, f'{period_name_en.lower().replace(" ", "_")}_performance_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write(f"===== {period_name_en} Slippage Impact Analysis =====\n\n")
        
        # Write header
        header_format = "{:<20} {:<15} {:<15} {:<15} {:<15}\n"
        f.write(header_format.format(*headers))
        
        # Write data rows
        row_format = "{:<20} {:<15} {:<15} {:<15} {:<15}\n"
        for row in rows:
            f.write(row_format.format(*row))
    
    print(f"{period_name_en} slippage impact analysis table has been saved.")

def analyze_slippage_impact():
    """슬리피지 영향 분석 메인 함수"""
    # CSV 파일 로드
    csv_file = "closed_positions_20250403_073826.csv"
    df = pd.read_csv(csv_file)
    
    # 데이터 전처리
    df['Entry Price'] = df['Entry Price'].apply(parse_number)
    df['Max Size'] = df['Max Size'].apply(parse_number)
    df['Max Size USDT'] = df['Max Size USDT'].apply(parse_number)
    df['Realized PnL'] = df['Realized PnL'].apply(parse_number)
    df['Open Time'] = df['Open Time'].apply(parse_date)
    df['Close Time'] = df['Close Time'].apply(parse_date)
    
    # 기간 분리
    period1_start = datetime.datetime(2023, 10, 11)
    period1_end = datetime.datetime(2024, 5, 21)
    period2_start = datetime.datetime(2024, 7, 24)
    period2_end = datetime.datetime(2025, 3, 26)
    
    df_period1 = df[(df['Close Time'] >= period1_start) & (df['Close Time'] <= period1_end)]
    df_period2 = df[(df['Close Time'] >= period2_start) & (df['Close Time'] <= period2_end)]
    
    # 원본 데이터로 성과 지표 계산
    original_metrics_period1 = calculate_performance_metrics(df_period1.copy(), INITIAL_CAPITAL_PERIOD1, is_original=True)
    original_metrics_period2 = calculate_performance_metrics(df_period2.copy(), INITIAL_CAPITAL_PERIOD2, is_original=True)
    
    # 각 슬리피지 비율에 대한 성과 지표 계산
    period1_metrics = [original_metrics_period1]
    period2_metrics = [original_metrics_period2]
    
    for rate in SLIPPAGE_RATES:
        # 기간 1 분석
        df_period1_with_slippage = apply_slippage(df_period1, rate)
        metrics_period1 = calculate_performance_metrics(df_period1_with_slippage, INITIAL_CAPITAL_PERIOD1)
        period1_metrics.append(metrics_period1)
        
        # 기간 2 분석
        df_period2_with_slippage = apply_slippage(df_period2, rate)
        metrics_period2 = calculate_performance_metrics(df_period2_with_slippage, INITIAL_CAPITAL_PERIOD2)
        period2_metrics.append(metrics_period2)
    
    # 결과 시각화 및 저장
    plot_asset_growth_comparison(period1_metrics, "기간 1", SLIPPAGE_RATES)
    plot_asset_growth_comparison(period2_metrics, "기간 2", SLIPPAGE_RATES)
    
    # 성과 비교 표 생성
    create_performance_comparison_table(period1_metrics, "기간 1", SLIPPAGE_RATES)
    create_performance_comparison_table(period2_metrics, "기간 2", SLIPPAGE_RATES)
    
    # Summary of slippage impact analysis
    print("\n===== Slippage Impact Analysis Summary =====")
    print(f"Period 1 Original Return: {original_metrics_period1['Total Return (%)']:.2f}%")
    for i, rate in enumerate(SLIPPAGE_RATES):
        print(f"Period 1 with {rate*100:.2f}% Slippage: {period1_metrics[i+1]['Total Return (%)']:.2f}% (Difference: {period1_metrics[i+1]['Total Return (%)'] - original_metrics_period1['Total Return (%)']:.2f}%)")
    
    print(f"\nPeriod 2 Original Return: {original_metrics_period2['Total Return (%)']:.2f}%")
    for i, rate in enumerate(SLIPPAGE_RATES):
        print(f"Period 2 with {rate*100:.2f}% Slippage: {period2_metrics[i+1]['Total Return (%)']:.2f}% (Difference: {period2_metrics[i+1]['Total Return (%)'] - original_metrics_period2['Total Return (%)']:.2f}%)")

if __name__ == "__main__":
    analyze_slippage_impact()
