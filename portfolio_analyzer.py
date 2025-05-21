#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Set plot style and font settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_context("talk")

# Disable warning
pd.options.mode.chained_assignment = None

def extract_capital_data(input_path, output_path):
    """
    모델 포트폴리오 CSV 파일에서 자본금 시계열 데이터를 추출
    
    Args:
        input_path (str): 입력 CSV 파일 경로
        output_path (str): 출력 CSV 파일 경로
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter data from September 2024 onwards
    sept_2024 = pd.Timestamp('2024-09-01')
    filtered_df = df[df['time'] >= sept_2024].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    
    if filtered_df.empty:
        print("No data found: Unable to find data after September 2024.")
        return None
    
    # Create result dataframe with time and capital columns
    result = []
    
    # First row of September data: Use the start_capital
    result.append({
        'time': filtered_df.iloc[0]['time'],
        'capital': filtered_df.iloc[0]['start_capital']
    })
    
    # For all subsequent rows: Use the end_capital
    for i in range(1, len(filtered_df)):  # Start from the second row (index 1)
        result.append({
            'time': filtered_df.iloc[i]['time'],
            'capital': filtered_df.iloc[i]['end_capital']
        })
    
    # Convert to dataframe
    result_df = pd.DataFrame(result)
    
    # Remove any duplicates based on time
    result_df = result_df.drop_duplicates(subset=['time'])
    
    # Sort by time
    result_df = result_df.sort_values('time')
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Data extraction completed: {output_path} (Total rows: {len(result_df)})")
    
    return result_df

def load_capital_data(file_path):
    """
    자본금 시계열 데이터 CSV 파일 로드
    
    Args:
        file_path (str): 로드할 CSV 파일 경로
        
    Returns:
        DataFrame: 로드된 자본금 데이터
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df = df.reset_index(drop=True)
    # Fill any missing dates with forward fill
    df = df.set_index('time')
    df = df.resample('D').last().ffill()
    return df.reset_index()

def calculate_returns(df):
    """
    일별 수익률 및 누적 수익률 계산
    
    Args:
        df (DataFrame): 자본금 데이터
        
    Returns:
        DataFrame: 수익률이 계산된 데이터
    """
    # Calculate daily returns
    df['daily_return'] = df['capital'].pct_change()
    
    # Calculate cumulative returns
    initial_capital = df['capital'].iloc[0]
    df['cumulative_return'] = (df['capital'] / initial_capital) - 1
    
    # Calculate log returns for Sharpe ratio
    df['log_return'] = np.log(df['capital'] / df['capital'].shift(1))
    
    return df

def calculate_drawdown(df):
    """
    드로다운(낙폭) 시리즈 계산
    
    Args:
        df (DataFrame): 자본금 데이터
        
    Returns:
        DataFrame: 드로다운이 계산된 데이터
    """
    # Calculate rolling maximum
    df['rolling_max'] = df['capital'].cummax()
    # Calculate drawdown
    df['drawdown'] = (df['capital'] / df['rolling_max']) - 1
    
    return df

def calculate_performance_metrics(df):
    """
    다양한 성능 지표 계산
    
    Args:
        df (DataFrame): 자본금 데이터
        
    Returns:
        dict: 계산된 성능 지표
    """
    # Time period in years
    start_date = df['time'].iloc[0]
    end_date = df['time'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    
    # Total return
    total_return = df['cumulative_return'].iloc[-1]
    
    # Annualized return
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Maximum drawdown
    max_drawdown = df['drawdown'].min()
    
    # Annualized volatility (using log returns)
    daily_vol = df['log_return'].std()
    annualized_vol = daily_vol * np.sqrt(365)  # Using 365 days for crypto markets (24/7 trading)
    
    # Sharpe ratio (assuming risk-free rate of 0% for simplicity)
    sharpe_ratio = (annualized_return) / annualized_vol
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Maximum Drawdown': max_drawdown,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Start Date': start_date,
        'End Date': end_date,
        'Period (Years)': years
    }

def create_monthly_return_heatmap(df, output_dir, slippage_str=""):
    """
    월별 수익률 히트맵 생성
    
    Args:
        df (DataFrame): 자본금 데이터
        output_dir (str): 결과 저장 디렉토리
        slippage_str (str): 슬리피지 정보 문자열
    """
    # Calculate monthly returns
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    # Group by year and month and calculate returns
    monthly_grouped = df.groupby(['year', 'month'])
    monthly_returns = monthly_grouped.apply(
        lambda x: (x['capital'].iloc[-1] / x['capital'].iloc[0]) - 1,
        include_groups=False
    ).unstack()
    
    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the heatmap
    sns.heatmap(monthly_returns, annot=True, cmap='RdYlGn', center=0, fmt='.1%',
                cbar_kws={'label': 'Monthly Return'}, ax=ax)
    
    # Set labels
    ax.set_yticklabels(monthly_returns.index, rotation=0)
    
    # 월 이름 매핑 (월 번호에 해당하는 약자 사용)
    month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    # 실제 데이터에 있는 월 번호만 라벨로 사용
    month_labels = [month_mapping.get(col, str(col)) for col in monthly_returns.columns]
    ax.set_xticklabels(month_labels, rotation=0)
    title = 'Monthly Return Heatmap'
    if slippage_str:
        title += f' {slippage_str}'
    ax.set_title(title, fontsize=16)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'monthly_return_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Monthly return heatmap saved: {output_path}")

def plot_performance(df, metrics, output_dir, slippage_str=""):
    """
    Generate comprehensive performance chart
    
    Args:
        df (DataFrame): Capital data
        metrics (dict): Performance metrics
        output_dir (str): Output directory
        slippage_str (str): Slippage info string
    """
    # Create figure with 2 subplots (1x2 grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Title for the figure
    title = 'Portfolio Performance Analysis'
    if slippage_str:
        title += f' {slippage_str}'
    fig.suptitle(title, fontsize=24, y=0.98)
    
    # Set datetime column as index for proper date plotting
    df_plot = df.copy()
    df_plot.set_index('time', inplace=True)
    
    # Equity curve (ax1)
    df_plot['capital'].plot(ax=ax1, color='#1f77b4', linewidth=2)
    ax1.set_title('Equity Curve', fontsize=16)
    ax1.set_ylabel('Capital (USDT)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis with commas
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    
    # Format x-axis with dates at appropriate intervals
    import matplotlib.dates as mdates
    
    # Determine date locator based on the date range
    date_range = (df['time'].max() - df['time'].min()).days
    
    if date_range <= 60:  # For short periods (< 2 months)
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    elif date_range <= 180:  # For medium periods (2-6 months)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:  # For longer periods
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Format the date labels with year included
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    
    # Rotate date labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Performance metrics (ax2)
    # Clear the axis and just use it as a text box
    ax2.axis('off')
    
    # Create performance metrics string
    metrics_str = '\n'.join([
        f"Starting Capital: {df['capital'].iloc[0]:,.2f} USDT",
        f"Ending Capital: {df['capital'].iloc[-1]:,.2f} USDT",
        f"Period: {metrics['Start Date'].strftime('%b %d, %Y')} - {metrics['End Date'].strftime('%b %d, %Y')} ({metrics['Period (Years)']:.2f} years)",
        f"Total Return: {metrics['Total Return']:.2%}",
        f"Annualized Return: {metrics['Annualized Return']:.2%}",
        f"Maximum Drawdown: {metrics['Maximum Drawdown']:.2%}",
        f"Annualized Volatility: {metrics['Annualized Volatility']:.2%}",
        f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}",
        f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}"
    ])
    
    # Add text box with performance metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, metrics_str, transform=ax2.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax2.set_title('Performance Metrics', fontsize=16)
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Performance summary graph saved: {output_path}")
    
    # Save metrics to markdown
    md_path = os.path.join(output_dir, 'performance_metrics.md')
    with open(md_path, 'w') as f:
        title = '# Portfolio Performance Metrics'
        if slippage_str:
            title += f' {slippage_str}'
        f.write(f"{title}\n\n")
        f.write(f"## Basic Information\n\n")
        f.write(f"* Starting Capital: {df['capital'].iloc[0]:,.2f} USDT\n")
        f.write(f"* Ending Capital: {df['capital'].iloc[-1]:,.2f} USDT\n")
        f.write(f"* Analysis Period: {metrics['Start Date'].strftime('%b %d, %Y')} - {metrics['End Date'].strftime('%b %d, %Y')} ({metrics['Period (Years)']:.2f} years)\n\n")
        
        f.write(f"## Return Metrics\n\n")
        f.write(f"* Total Return: {metrics['Total Return']:.2%}\n")
        f.write(f"* Annualized Return: {metrics['Annualized Return']:.2%}\n")
        f.write(f"* Annualized Volatility: {metrics['Annualized Volatility']:.2%}\n\n")
        
        f.write(f"## Risk Metrics\n\n")
        f.write(f"* Maximum Drawdown (MDD): {metrics['Maximum Drawdown']:.2%}\n")
        f.write(f"* Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n")
        f.write(f"* Calmar Ratio: {metrics['Calmar Ratio']:.2f}\n")
    
    print(f"Performance metrics markdown saved: {md_path}")

def run_analysis(input_file, output_dir, slippage=0.0):
    """
    포트폴리오 성능 분석 실행
    
    Args:
        input_file (str): 입력 CSV 파일 경로
        output_dir (str): 결과 저장 디렉토리
        slippage (float): 슬리피지 비율(%)
    """
    # 슬리피지 정보 문자열 생성
    slippage_str = f"(Slippage: {slippage:.2f}%)" if slippage > 0 else ""
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 자본금 시계열 데이터 추출
    capital_time_series_path = os.path.join(output_dir, 'capital_time_series.csv')
    
    # 1. 데이터 추출
    df = extract_capital_data(input_file, capital_time_series_path)
    if df is None:
        return
    
    # 2. 계산 및 분석
    df = calculate_returns(df)
    df = calculate_drawdown(df)
    metrics = calculate_performance_metrics(df)
    
    # 3. Print performance metrics
    print("\n===== Performance Metrics =====")
    print(f"Starting Capital: {df['capital'].iloc[0]:,.2f} USDT")
    print(f"Ending Capital: {df['capital'].iloc[-1]:,.2f} USDT")
    print(f"Period: {metrics['Start Date'].strftime('%b %d, %Y')} - {metrics['End Date'].strftime('%b %d, %Y')} ({metrics['Period (Years)']:.2f} years)")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Annualized Return: {metrics['Annualized Return']:.2%}")
    print(f"Maximum Drawdown: {metrics['Maximum Drawdown']:.2%}")
    print(f"Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
    
    # Create monthly return heatmap and plot performance
    create_monthly_return_heatmap(df, output_dir, slippage_str)
    plot_performance(df, metrics, output_dir, slippage_str)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Model Portfolio Performance Analysis')
    parser.add_argument('--input', type=str, 
                        default=None,
                        help='Model portfolio CSV file path (if not specified, it will be auto-determined based on slippage)')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/jdpark/Downloads/binance_leaderboard_analysis/overlap_analysis/output/portfolio_analysis',
                        help='Output directory for results')
    parser.add_argument('--slippage', type=float, default=0.0,
                        help='Slippage rate (%, e.g., 0.05 means 0.05%)')
    
    args = parser.parse_args()
    
    # Determine input file path based on slippage
    input_file = args.input
    if input_file is None:
        # Base directory for model portfolio results
        base_dir = '/Users/jdpark/Downloads/binance_leaderboard_analysis/overlap_analysis/output'
        
        # Format slippage with 2 decimal places for directory naming (e.g., 0.10 instead of 0.1)
        if args.slippage > 0:
            # Always format with 2 decimal places: 0.1 -> 0.10
            slippage_formatted = f"{args.slippage:.2f}"
            input_dir = f"model_portfolio_slippage_{slippage_formatted}"
        else:
            input_dir = "model_portfolio"
            
        input_file = os.path.join(base_dir, input_dir, "model_portfolio_capital.csv")
        print(f"Auto-determined input file: {input_file}")
    
    # Determine output directory based on slippage
    output_dir = args.output_dir
    if args.slippage > 0:
        # Always format with 2 decimal places: 0.1 -> 0.10
        slippage_formatted = f"{args.slippage:.2f}"
        output_dir = f"{args.output_dir}_slippage_{slippage_formatted}"
    
    # Run analysis
    run_analysis(input_file, output_dir, args.slippage)

if __name__ == "__main__":
    main()
