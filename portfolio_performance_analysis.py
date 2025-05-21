import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_capital_data(file_path):
    """Load capital time series data from CSV file"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df = df.reset_index(drop=True)
    # Fill any missing dates with forward fill
    df = df.set_index('time')
    df = df.resample('D').last().ffill()
    return df.reset_index()

def calculate_returns(df):
    """Calculate daily returns and cumulative returns"""
    # Calculate daily returns
    df['daily_return'] = df['capital'].pct_change()
    
    # Calculate cumulative returns
    initial_capital = df['capital'].iloc[0]
    df['cumulative_return'] = (df['capital'] / initial_capital) - 1
    
    # Calculate log returns for Sharpe ratio
    df['log_return'] = np.log(df['capital'] / df['capital'].shift(1))
    
    return df

def calculate_drawdown(df):
    """Calculate drawdown series"""
    # Calculate rolling maximum
    df['rolling_max'] = df['capital'].cummax()
    # Calculate drawdown
    df['drawdown'] = (df['capital'] / df['rolling_max']) - 1
    
    return df

def calculate_performance_metrics(df):
    """Calculate various performance metrics"""
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
    annualized_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days in a year
    
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

def create_monthly_return_heatmap(df):
    """Create a heatmap of monthly returns"""
    # Calculate monthly returns
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    # Group by year and month and calculate returns
    monthly_grouped = df.groupby(['year', 'month'])
    monthly_returns = monthly_grouped.apply(
        lambda x: (x['capital'].iloc[-1] / x['capital'].iloc[0]) - 1,
        include_groups=False
    ).unstack()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get the actual months we have data for
    available_months = sorted(monthly_returns.columns)
    
    # Create heatmap with customized colormap
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    ax = sns.heatmap(monthly_returns, 
                     cmap=cmap, 
                     annot=True, 
                     fmt='.1%', 
                     center=0,
                     cbar_kws={'label': 'Monthly Return (%)'})
    
    # Set labels
    ax.set_title('Monthly Returns Heatmap', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Year', fontsize=14)
    
    # Set month names as x-tick labels using only available months
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    tick_labels = [month_names[m-1] for m in available_months]
    ax.set_xticklabels(tick_labels)
    
    plt.tight_layout()
    plt.savefig('monthly_returns_heatmap.png', dpi=300)
    
def plot_performance(df, metrics):
    """Create comprehensive performance plots"""
    # Create a 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Format percentage display function
    def percentage_formatter(x, pos):
        return f'{x:.1%}'
    
    percent_formatter = FuncFormatter(percentage_formatter)
    
    # Plot 1: Equity Curve
    ax1 = axs[0, 0]
    ax1.plot(df['time'], df['capital'], linewidth=2)
    ax1.set_title('Equity Curve', fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Capital (USDT)', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Cumulative Returns
    ax2 = axs[0, 1]
    ax2.plot(df['time'], df['cumulative_return'], linewidth=2, color='green')
    ax2.set_title('Cumulative Returns', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.yaxis.set_major_formatter(percent_formatter)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Drawdown
    ax3 = axs[1, 0]
    ax3.fill_between(df['time'], df['drawdown'], 0, color='red', alpha=0.3)
    ax3.plot(df['time'], df['drawdown'], linewidth=1, color='red')
    ax3.set_title('Drawdown', fontsize=16)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.yaxis.set_major_formatter(percent_formatter)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Performance metrics as text
    ax4 = axs[1, 1]
    ax4.axis('off')
    
    # Format the performance metrics
    metrics_str = "\n".join([
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
    ax4.text(0.05, 0.95, metrics_str, transform=ax4.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax4.set_title('Performance Metrics', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300)

def main():
    # File path
    file_path = '/Users/jdpark/Downloads/binance_leaderboard_analysis/overlap_analysis/output/model_portfolio/capital_time_series.csv'
    
    # Load data
    df = load_capital_data(file_path)
    
    # Calculate returns and drawdowns
    df = calculate_returns(df)
    df = calculate_drawdown(df)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(df)
    
    # Print performance metrics
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
    
    # Create plots
    plot_performance(df, metrics)
    create_monthly_return_heatmap(df)
    
    print("\nAnalysis complete. Generated files:")
    print("1. performance_summary.png - Overall performance metrics and visualizations")
    print("2. monthly_returns_heatmap.png - Monthly returns heatmap")

if __name__ == "__main__":
    main()
