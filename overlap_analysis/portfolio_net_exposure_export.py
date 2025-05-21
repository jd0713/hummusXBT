import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from config import trader_weight

def export_portfolio_net_exposure(periods, out_dir=None):
    """
    periods: list of dict (output of analyze_all_periods)
    out_dir: directory to save csv/png (default: script dir)
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    times = []
    net_exposures = []
    for period in periods:
        # 순노출(%) 계산 (legacy와 동일)
        net = 0.0
        for trader, summary in period.get('trader_summaries', {}).items():
            weight = trader_weight.get(trader, 0.0)
            net_ratio = summary.get('net_ratio', 0)
            # 만약 net_ratio가 1 미만(즉, 0~1 사이)이면 %로 변환 (이중 변환 방지)
            if abs(net_ratio) < 1.0 and abs(net_ratio) > 0:
                net_ratio = net_ratio * 100
            net += weight * net_ratio
        times.append(period['end'])
        net_exposures.append(net)
    # CSV 저장
    df = pd.DataFrame({'time': times, 'net_exposure': net_exposures})
    csv_path = os.path.join(out_dir, 'portfolio_net_exposure.csv')
    df.to_csv(csv_path, index=False)
    # 그래프 저장
    plt.figure(figsize=(14,5))
    plt.plot(times, net_exposures, label='Net Exposure (%)', color='royalblue')
    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.title('Portfolio Net Exposure Over Time')
    plt.xlabel('Time')
    plt.ylabel('Net Exposure (%)')
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'portfolio_net_exposure.png')
    plt.savefig(png_path)
    plt.close()
    print(f'Portfolio net exposure csv saved as {csv_path}')
    print(f'Portfolio net exposure graph saved as {png_path}')

# 사용 예시 (main.py에서)
# from portfolio_net_exposure_export import export_portfolio_net_exposure
# export_portfolio_net_exposure(periods)
