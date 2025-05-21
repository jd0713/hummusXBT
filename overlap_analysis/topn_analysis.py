# topn_analysis.py
"""
Top N 구간 추출 유틸리티 (최대 동시 포지션, 순비율 등)
"""

def top_n_by_concurrent_positions(periods, n=5):
    # 동시 포지션 수 기준 상위 N
    sorted_periods = sorted(periods, key=lambda p: len(p['positions']), reverse=True)
    return sorted_periods[:n]

# 단순 롱/숏 비율 관련 함수 제거 - 순노출 기준만 유지

def top_n_by_net_ratio(periods, n=5, direction='long'):
    """순비율(%) 기준 상위 N개 구간 추출
    포트폴리오 가중 순비율 기준으로 정렬 (트레이더 가중치 필요)
    """
    from config import trader_weight  # 가중치 가져오기
    
    # 각 구간별 포트폴리오 가중 순비율 계산
    weighted_periods = []
    for period in periods:
        trader_summaries = period.get('trader_summaries', {})
        weighted_net_ratio = 0.0
        
        for trader, summary in trader_summaries.items():
            weight = trader_weight.get(trader, 0.0)  # 가중치 (0~1 사이)
            net_ratio = summary.get('net_ratio', 0.0)
            weighted_net_ratio += net_ratio * weight
        
        # 순비율과 가중 순비율 정보 추가
        period_with_weight = period.copy()
        period_with_weight['_weighted_net_ratio'] = weighted_net_ratio
        weighted_periods.append(period_with_weight)
    
    # 가중 순비율 기준으로 정렬
    if direction == 'long':
        # 롱 순비율은 높은 숬서(내림차순)
        sorted_periods = sorted(weighted_periods, key=lambda p: p.get('_weighted_net_ratio', 0.0), reverse=True)
    else:
        # 숏 순비율은 낮은 숬서(오름차순)
        sorted_periods = sorted(weighted_periods, key=lambda p: p.get('_weighted_net_ratio', 0.0))
    
    return sorted_periods[:n]
