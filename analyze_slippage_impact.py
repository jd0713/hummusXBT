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
SLIPPAGE_RATES = [0.0005, 0.001, 0.0015]  # 0.05%, 0.1%, 0.15%

# 결과 디렉토리 설정 (트레이더별 디렉토리 사용)
def get_results_dir(trader_id=None):
    """트레이더 ID에 따른 결과 디렉토리 경로 반환"""
    if trader_id:
        return os.path.join('analysis_results', trader_id, 'slippage_analysis')
    else:
        return 'analysis_results/slippage_analysis'

# 기본 결과 디렉토리 (하위 호환성 유지)
RESULTS_DIR = 'analysis_results/slippage_analysis'

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

def calculate_performance_metrics_without_capital(df, period_name="", is_original=False):
    """원금 정보 없이 슬리피지가 적용된 데이터로 성과 지표 계산"""
    # 날짜 기준 정렬
    df = df.sort_values('Close Time')
    
    # 원본 데이터와 슬리피지 적용 데이터 구분
    if is_original:
        pnl_column = 'Realized PnL'
    else:
        pnl_column = 'Slippage_Adjusted_PnL'
    
    # 누적 PnL 계산
    df['Cumulative_PnL'] = df[pnl_column].cumsum()
    
    # 총 수익 계산
    total_pnl = df[pnl_column].sum()
    
    # 거래 일수 계산
    start_date = df['Close Time'].min()
    end_date = df['Close Time'].max()
    trading_days = (end_date - start_date).days if start_date and end_date else 0
    
    # 승률 계산
    win_count = len(df[df[pnl_column] > 0])
    win_rate = (win_count / len(df)) * 100 if len(df) > 0 else 0
    
    # 일평균 수익
    daily_return = total_pnl / trading_days if trading_days > 0 else 0
    
    # 평균 수익/손실
    win_trades = df[df[pnl_column] > 0]
    loss_trades = df[df[pnl_column] < 0]
    avg_win = win_trades[pnl_column].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades[pnl_column].mean() if len(loss_trades) > 0 else 0
    
    return {
        'Period': period_name,
        'Is Original': is_original,
        'Total PnL': total_pnl,
        'Win Rate (%)': win_rate,
        'Trading Days': trading_days,
        'Daily Return': daily_return,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Start Date': start_date,
        'End Date': end_date,
        'PnL Series': df[['Close Time', pnl_column]].copy(),
        'Cumulative PnL Series': df[['Close Time', 'Cumulative_PnL']].copy()
    }

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

def plot_pnl_growth_comparison(metrics_list, period_name, slippage_rates):
    """Plot PnL growth comparison with different slippage rates for traders without initial capital"""
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
        pnl_series = metrics['Cumulative PnL Series']
        plt.plot(pnl_series['Close Time'], pnl_series['Cumulative_PnL'], 
                 marker='', linestyle='-', color=colors[i], label=labels[i])
    
    plt.title(f'{period_name_en} Slippage Impact Analysis: PnL Growth Comparison', fontsize=16)
    plt.ylabel('Cumulative PnL (USDT)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    
    # Display total PnL
    for i, metrics in enumerate(metrics_list):
        total_pnl = metrics['Total PnL']
        plt.annotate(f'{labels[i]}: ${total_pnl:,.2f}', 
                    xy=(0.02, 0.95 - i*0.05), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{period_name_en.lower().replace(" ", "_")}_pnl_growth_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"{period_name_en} slippage impact PnL analysis graph has been saved.")

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
    
    # 원금 정보가 있는지 확인 (어떤 지표를 표시할지 결정)
    has_capital_info = 'Total Return (%)' in metrics_list[0]
    
    if has_capital_info:
        # 원금 정보가 있는 트레이더를 위한 행
        rows = [
            ['Total PnL (USD)', f"{metrics_list[0]['Total PnL']:.2f}"] + [f"{metrics['Total PnL']:.2f}" for metrics in metrics_list[1:]],
            ['Total Return (%)', f"{metrics_list[0]['Total Return (%)']:.2f}"] + [f"{metrics['Total Return (%)']:.2f}" for metrics in metrics_list[1:]],
            ['Annualized Return (%)', f"{metrics_list[0]['Annualized Return (%)']:.2f}"] + [f"{metrics['Annualized Return (%)']:.2f}" for metrics in metrics_list[1:]],
            ['MDD (%)', f"{metrics_list[0]['MDD (%)']:.2f}"] + [f"{metrics['MDD (%)']:.2f}" for metrics in metrics_list[1:]],
            ['Sharpe Ratio', f"{metrics_list[0]['Sharpe Ratio']:.2f}"] + [f"{metrics['Sharpe Ratio']:.2f}" for metrics in metrics_list[1:]],
            ['Win Rate (%)', f"{metrics_list[0]['Win Rate (%)']:.2f}"] + [f"{metrics['Win Rate (%)']:.2f}" for metrics in metrics_list[1:]]
        ]
    else:
        # 원금 정보가 없는 트레이더를 위한 행
        rows = [
            ['Total PnL (USD)', f"${metrics_list[0]['Total PnL']:,.2f}"] + [f"${metrics['Total PnL']:,.2f}" for metrics in metrics_list[1:]],
            ['Daily Return (USD)', f"${metrics_list[0]['Daily Return']:,.2f}"] + [f"${metrics['Daily Return']:,.2f}" for metrics in metrics_list[1:]],
            ['Win Rate (%)', f"{metrics_list[0]['Win Rate (%)']:.2f}%"] + [f"{metrics['Win Rate (%)']:.2f}%" for metrics in metrics_list[1:]],
            ['Avg Win (USD)', f"${metrics_list[0]['Avg Win']:,.2f}"] + [f"${metrics['Avg Win']:,.2f}" for metrics in metrics_list[1:]],
            ['Avg Loss (USD)', f"${metrics_list[0]['Avg Loss']:,.2f}"] + [f"${metrics['Avg Loss']:,.2f}" for metrics in metrics_list[1:]],
            ['Trading Days', f"{metrics_list[0]['Trading Days']}"] + [f"{metrics['Trading Days']}" for metrics in metrics_list[1:]]
        ]
    
    # Save table to file
    # 결과 디렉토리가 존재하는지 확인
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
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

def analyze_slippage_impact(trader_id=None, csv_file=None):
    """슬리피지 영향 분석 메인 함수"""
    from trader_config import get_trader, get_all_trader_ids
    from trader_analysis_part1 import analyze_pnl_by_periods, analyze_pnl_without_periods
    
    # 트레이더 선택 또는 확인
    if trader_id is None:
        # 기본 트레이더 사용
        trader_id = "hummusXBT"
    
    # 트레이더 정보 가져오기
    trader = get_trader(trader_id)
    trader_name = trader['name']
    
    # 트레이더 설정에 따라 분석 방식 결정
    use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
    
    # 결과 폴더 설정
    global RESULTS_DIR
    RESULTS_DIR = f"analysis_results/{trader_id}/slippage_analysis"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # CSV 파일 경로 가져오기
    if csv_file is None:
        # 최신 CSV 파일 찾기
        trader_data_dir = f"trader_data/{trader_id}"
        if not os.path.exists(trader_data_dir):
            print(f"오류: {trader_id} 트레이더의 데이터 디렉토리를 찾을 수 없습니다.")
            return False
        
        # 디렉토리에서 CSV 파일 찾기
        csv_files = [f for f in os.listdir(trader_data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"오류: {trader_id} 트레이더의 CSV 파일을 찾을 수 없습니다.")
            return False
        
        # 가장 최근 파일 선택
        csv_files.sort(reverse=True)
        csv_file = os.path.join(trader_data_dir, csv_files[0])
    
    print(f"\n===== {trader_name} 트레이더 슬리피지 영향 분석 시작 =====")
    print(f"분석할 파일: {csv_file}")
    
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_file)
        
        # 데이터 전처리
        df['Entry Price'] = df['Entry Price'].apply(parse_number)
        df['Max Size'] = df['Max Size'].apply(parse_number)
        df['Max Size USDT'] = df['Max Size USDT'].apply(parse_number)
        df['Realized PnL'] = df['Realized PnL'].apply(parse_number)
        df['Open Time'] = df['Open Time'].apply(parse_date)
        df['Close Time'] = df['Close Time'].apply(parse_date)
        
        # 트레이더 설정에 따라 분석 방식 이미 결정됨
        
        if use_periods:
            # 기간 구분이 있는 경우 (hummusXBT 같은 트레이더)
            initial_capital_period1 = trader.get('initial_capital_period1')
            if initial_capital_period1 is None:
                print(f"오류: {trader_id} 트레이더의 기간 1 원금이 설정되지 않았습니다.")
                return None
                
            initial_capital_period2 = trader.get('initial_capital_period2')
            if initial_capital_period2 is None:
                print(f"오류: {trader_id} 트레이더의 기간 2 원금이 설정되지 않았습니다.")
                return None
            period_split_date_str = trader.get('period_split_date')
            if period_split_date_str is None:
                print(f"오류: {trader_id} 트레이더의 기간 구분 날짜가 설정되지 않았습니다.")
                return None
            
            # 기간 구분 날짜 설정
            period_split_date = datetime.datetime.strptime(period_split_date_str, '%Y-%m-%d')
            
            # 기간 분리
            df_period1 = df[df['Close Time'] < period_split_date]
            df_period2 = df[df['Close Time'] >= period_split_date]
            
            if df_period1.empty or df_period2.empty:
                print("경고: 일부 기간에 데이터가 없습니다. 기간 구분 날짜를 확인하세요.")
                if df_period1.empty:
                    print(f"기간 1 ({period_split_date} 이전) 데이터가 없습니다.")
                if df_period2.empty:
                    print(f"기간 2 ({period_split_date} 이후) 데이터가 없습니다.")
                return False
            
            # 원본 데이터로 성과 지표 계산
            original_metrics_period1 = calculate_performance_metrics(df_period1.copy(), initial_capital_period1, is_original=True)
            original_metrics_period2 = calculate_performance_metrics(df_period2.copy(), initial_capital_period2, is_original=True)
            
            # 각 슬리피지 비율에 대한 성과 지표 계산
            period1_metrics = [original_metrics_period1]
            period2_metrics = [original_metrics_period2]
            
            for rate in SLIPPAGE_RATES:
                # 기간 1 분석
                df_period1_with_slippage = apply_slippage(df_period1, rate)
                metrics_period1 = calculate_performance_metrics(df_period1_with_slippage, initial_capital_period1)
                period1_metrics.append(metrics_period1)
                
                # 기간 2 분석
                df_period2_with_slippage = apply_slippage(df_period2, rate)
                metrics_period2 = calculate_performance_metrics(df_period2_with_slippage, initial_capital_period2)
                period2_metrics.append(metrics_period2)
            
            # 결과 시각화 및 저장
            plot_asset_growth_comparison(period1_metrics, "기간 1", SLIPPAGE_RATES)
            plot_asset_growth_comparison(period2_metrics, "기간 2", SLIPPAGE_RATES)
            
            # 성과 비교 표 생성
            create_performance_comparison_table(period1_metrics, "기간 1", SLIPPAGE_RATES)
            create_performance_comparison_table(period2_metrics, "기간 2", SLIPPAGE_RATES)
            
            # 슬리피지 영향 분석 요약
            print("\n===== 슬리피지 영향 분석 요약 =====")
            print(f"기간 1 원본 수익률: {original_metrics_period1['Total Return (%)']:.2f}%")
            for i, rate in enumerate(SLIPPAGE_RATES):
                print(f"기간 1 {rate*100:.2f}% 슬리피지 적용: {period1_metrics[i+1]['Total Return (%)']:.2f}% (차이: {period1_metrics[i+1]['Total Return (%)'] - original_metrics_period1['Total Return (%)']:.2f}%)")
            
            print(f"\n기간 2 원본 수익률: {original_metrics_period2['Total Return (%)']:.2f}%")
            for i, rate in enumerate(SLIPPAGE_RATES):
                print(f"기간 2 {rate*100:.2f}% 슬리피지 적용: {period2_metrics[i+1]['Total Return (%)']:.2f}% (차이: {period2_metrics[i+1]['Total Return (%)'] - original_metrics_period2['Total Return (%)']:.2f}%)")
        else:
            # 기간 구분이 없는 경우 (TRADERT22 같은 트레이더)
            initial_capital = trader.get('initial_capital', 100000)  # 기본 원금 설정
            
            # 원금 정보 확인
            if initial_capital is None:
                # 원금 정보가 없는 경우
                # 원본 데이터로 성과 지표 계산 (원금 없이)
                original_metrics = calculate_performance_metrics_without_capital(df.copy(), '전체 기간', is_original=True)
                
                # 각 슬리피지 비율에 대한 성과 지표 계산
                metrics_list = [original_metrics]
                
                for rate in SLIPPAGE_RATES:
                    # 슬리피지 적용 분석
                    df_with_slippage = apply_slippage(df, rate)
                    metrics = calculate_performance_metrics_without_capital(df_with_slippage, '전체 기간')
                    metrics_list.append(metrics)
            else:
                # 원금 정보가 있는 경우
                # 원본 데이터로 성과 지표 계산
                original_metrics = calculate_performance_metrics(df.copy(), initial_capital, is_original=True)
                
                # 각 슬리피지 비율에 대한 성과 지표 계산
                metrics_list = [original_metrics]
                
                for rate in SLIPPAGE_RATES:
                    # 슬리피지 적용 분석
                    df_with_slippage = apply_slippage(df, rate)
                    metrics = calculate_performance_metrics(df_with_slippage, initial_capital)
                    metrics_list.append(metrics)
            
            # 결과 시각화 및 저장
            if initial_capital is None:
                # 원금 정보가 없는 경우 PnL 성장 그래프 사용
                plot_pnl_growth_comparison(metrics_list, "전체 기간", SLIPPAGE_RATES)
            else:
                # 원금 정보가 있는 경우 자산 성장 그래프 사용
                plot_asset_growth_comparison(metrics_list, "전체 기간", SLIPPAGE_RATES)
            
            # 성과 비교 표 생성
            create_performance_comparison_table(metrics_list, "전체 기간", SLIPPAGE_RATES)
            
            # 슬리피지 영향 분석 요약
            print("\n===== 슬리피지 영향 분석 요약 =====")
            if initial_capital is None:
                print("원금 정보가 없어 수익률을 계산할 수 없습니다.")
                print(f"원본 수익(PnL): ${original_metrics['Total PnL']:,.2f}")
                for i, rate in enumerate(SLIPPAGE_RATES):
                    pnl_diff = metrics_list[i+1]['Total PnL'] - original_metrics['Total PnL']
                    print(f"{rate*100:.2f}% 슬리피지 적용: ${metrics_list[i+1]['Total PnL']:,.2f} (차이: ${pnl_diff:,.2f})")
            else:
                print(f"원본 수익률: {original_metrics['Total Return (%)']:.2f}%")
                for i, rate in enumerate(SLIPPAGE_RATES):
                    return_diff = metrics_list[i+1]['Total Return (%)'] - original_metrics['Total Return (%)']
                    print(f"{rate*100:.2f}% 슬리피지 적용: {metrics_list[i+1]['Total Return (%)']:.2f}% (차이: {return_diff:.2f}%)")
        
        print("\n===== 슬리피지 분석 완료 =====")
        return True
    except Exception as e:
        print(f"오류: 슬리피지 영향 분석 중 문제가 발생했습니다.")
        print(f"오류 메시지: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="트레이더 슬리피지 영향 분석 도구")
    parser.add_argument("-t", "--trader", help="분석할 트레이더 ID", type=str)
    parser.add_argument("-f", "--file", help="분석할 CSV 파일 경로", type=str)
    
    args = parser.parse_args()
    
    # 슬리피지 영향 분석 실행
    analyze_slippage_impact(args.trader, args.file)
