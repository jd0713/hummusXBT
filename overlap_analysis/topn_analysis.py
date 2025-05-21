# topn_analysis.py
"""
Top N 구간 추출 유틸리티 (최대 순노출, 최대 동시 포지션 등)
"""

def top_n_by_net_exposure(periods, n=5):
    # 순노출(USDT) 절대값 기준 상위 N
    sorted_periods = sorted(periods, key=lambda p: max(abs(s['net_amt']) for s in p.get('trader_summaries', {}).values()), reverse=True)
    return sorted_periods[:n]

def top_n_by_concurrent_positions(periods, n=5):
    # 동시 포지션 수 기준 상위 N
    sorted_periods = sorted(periods, key=lambda p: len(p['positions']), reverse=True)
    return sorted_periods[:n]

def top_n_by_long_ratio(periods, n=5):
    # 롱 비율(%) 최대 기준 상위 N
    sorted_periods = sorted(periods, key=lambda p: max((s['long_ratio'] for s in p.get('trader_summaries', {}).values()), default=0), reverse=True)
    return sorted_periods[:n]

def top_n_by_short_ratio(periods, n=5):
    # 숏 비율(%) 최대 기준 상위 N
    sorted_periods = sorted(periods, key=lambda p: max((s['short_ratio'] for s in p.get('trader_summaries', {}).values()), default=0), reverse=True)
    return sorted_periods[:n]

def top_n_by_net_ratio(periods, n=5, direction='long'):
    # 순비율(%) 최대(롱) 또는 최소(숏) 기준 상위 N
    if direction == 'long':
        sorted_periods = sorted(periods, key=lambda p: max((s['net_ratio'] for s in p.get('trader_summaries', {}).values()), default=0), reverse=True)
    else:
        sorted_periods = sorted(periods, key=lambda p: min((s['net_ratio'] for s in p.get('trader_summaries', {}).values()), default=0))
    return sorted_periods[:n]
