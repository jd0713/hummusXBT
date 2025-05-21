# portfolio_metrics.py
# 포트폴리오 및 트레이더별 주요 지표 계산 함수 모듈

def calc_trader_metrics(active_positions, current_balance):
    """
    트레이더별 롱/숏/순 포지션, 비율 등 계산
    """
    long_open = {trader: 0.0 for trader in current_balance}
    short_open = {trader: 0.0 for trader in current_balance}
    for p in active_positions:
        trader = p['trader']
        try:
            size = float(str(p['size']).replace('USDT','').replace(',','').strip())
            size = abs(size)
        except:
            size = 0.0
        if trader in long_open and str(p['direction']).lower().startswith('long'):
            long_open[trader] += size
        if trader in short_open and str(p['direction']).lower().startswith('short'):
            short_open[trader] += size
    return long_open, short_open


def calc_period_summary(long_open, short_open, current_balance):
    summary = {}
    for trader in current_balance:
        bal = current_balance[trader]
        net_amt = long_open[trader] - short_open[trader]
        long_ratio = (long_open[trader] / bal * 100) if bal > 0 else 0.0
        short_ratio = (short_open[trader] / bal * 100) if bal > 0 else 0.0
        net_ratio = (net_amt / bal * 100) if bal > 0 else 0.0
        summary[trader] = {
            'long_amt': long_open[trader],
            'short_amt': short_open[trader],
            'net_amt': net_amt,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'net_ratio': net_ratio
        }
    return summary
