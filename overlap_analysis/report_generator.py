# report_generator.py
# 중복 함수 제거 및 코드 재구성 버전

from portfolio_metrics import calc_trader_metrics, calc_period_summary

# ===== 공통 헬퍼 함수 =====

def generate_position_table(positions):
    """포지션 테이블을 생성하는 공통 함수"""
    md_lines = []
    md_lines.append("| 트레이더 | 심볼 | 방향 | 규모(USDT) | 거래량(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in positions:
        # 규모(size)와 거래량(volume) 모두 표시
        size = p.get('size', 0)
        volume = p.get('volume', 0)
        realized_pnl = p.get('realized_pnl', '')
        realized_pnl_pct = p.get('realized_pnl_pct', '')
        
        # size가 숫자인 경우 포맷팅
        if isinstance(size, (int, float)):
            size = f"{size:.2f} USDT"
            
        md_lines.append(f"| {p['trader']} | {p.get('symbol', '')} | {p.get('direction', '')} | {size} | {volume} | {p.get('open', '')} | {p.get('close', '')} | {realized_pnl} | {realized_pnl_pct} |")
    md_lines.append("")
    return md_lines

def generate_trader_summary_table(trader_summaries, trader_weight):
    """트레이더 요약 테이블을 생성하는 공통 함수"""
    md_lines = []
    md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    
    # 포트폴리오 전체 순노출 계산을 위한 변수
    portfolio_weighted_net_ratio = 0.0
    portfolio_long_amt = 0.0
    portfolio_short_amt = 0.0
    
    for trader, s in trader_summaries.items():
        weight_pct = trader_weight.get(trader, 0.0) * 100
        weight = trader_weight.get(trader, 0.0)  # 실제 가중치 (0~1 사이 값)
        
        # 포트폴리오 가중 순비율 계산
        portfolio_weighted_net_ratio += s['net_ratio'] * weight
        portfolio_long_amt += s['long_amt']
        portfolio_short_amt += s['short_amt']
        
        md_lines.append(f"| {trader} | {weight_pct:.2f} | {s['long_amt']:.2f} | {s['long_ratio']:.2f} | {s['short_amt']:.2f} | {s['short_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
    
    # 포트폴리오 순방향 결정 - 가중 순비율과 순노출 모두 고려
    portfolio_net_amt = portfolio_long_amt - portfolio_short_amt
    
    # 가중치가 모두 0인 경우, 단순 순노출 값으로 판단
    if sum(trader_weight.values()) < 0.0001:  # 가중치 총합이 거의 0일 경우
        if portfolio_net_amt > 0:
            portfolio_net_dir = '롱'
        elif portfolio_net_amt < 0:
            portfolio_net_dir = '숏'
        else:
            portfolio_net_dir = '중립'
    else:  # 가중치가 존재하는 경우, 가중 순비율으로 판단
        if portfolio_weighted_net_ratio > 0:
            portfolio_net_dir = '롱'
        elif portfolio_weighted_net_ratio < 0:
            portfolio_net_dir = '숏'
        else:
            portfolio_net_dir = '중립'
    
    # 포트폴리오 요약 테이블 추가
    md_lines.append("")
    md_lines.append("| **포트폴리오 전체** | 롱 오픈(USDT) | 숏 오픈(USDT) | 순 오픈(USDT) | 순방향 | **가중 순비율(%)** |")
    md_lines.append("|---|---|---|---|---|---|")
    md_lines.append(f"| 요약 | {portfolio_long_amt:.2f} | {portfolio_short_amt:.2f} | {portfolio_net_amt:.2f} | {portfolio_net_dir} | **{portfolio_weighted_net_ratio:.2f}** |")
    md_lines.append("")
    
    return md_lines

def generate_balance_table(balance_after):
    """자산 테이블을 생성하는 공통 함수"""
    md_lines = []
    md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
    md_lines.append("|---|---|")
    for trader, bal in balance_after.items():
        md_lines.append(f"| {trader} | {bal:,.2f} |")
    md_lines.append("")
    return md_lines

def generate_period_header(idx, period):
    """구간 헤더를 생성하는 공통 함수"""
    return [f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분\n"]

def generate_markdown(periods, current_balance, trader_weight):
    md_lines = ["# 트레이더 포지션 구간별 롱/숏/넷 분석 결과\n"]
    for idx, period in enumerate(periods, 1):
        # 구간 헤더 추가
        md_lines.extend(generate_period_header(idx, period))
        
        # 필터링된 포지션 가져오기
        active_positions = period['positions']
        filtered_positions = []
        
        for p in active_positions:
            open_t = p.get('open')
            close_t = p.get('close')
            start = period['start']
            end = period['end']
            
            # 트레이더 요약에 포지션이 있는지 확인
            trader = p.get('trader', '')
            trader_summary = period.get('trader_summaries', {}).get(trader, {})
            long_amt = trader_summary.get('long_amt', 0)
            short_amt = trader_summary.get('short_amt', 0)
            has_position_in_summary = (long_amt > 0 or short_amt > 0)
            
            # 경계와 포지션 시간 비교
            opens_during_period = (start < open_t < end)
            closes_during_period = (start < close_t <= end)
            active_during_period = (open_t < start and close_t > end)
            opens_at_start = (open_t == start and close_t > end)
            closes_at_end = (open_t < start and close_t == end)
            boundary_case = (open_t == end)
            
            # 구간에 포함되어야 하는 경우만 필터링
            if ((opens_during_period or closes_during_period or active_during_period or 
                 opens_at_start or closes_at_end or has_position_in_summary) and 
                not boundary_case):
                filtered_positions.append(p)
        
        # 포지션 테이블 추가
        if filtered_positions:
            md_lines.extend(generate_position_table(filtered_positions))
        
        # 트레이더 요약 테이블 추가
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.extend(generate_trader_summary_table(trader_summaries, trader_weight))
        
        # 자산 테이블 추가
        balance_after = period.get('balance_after', None)
        if balance_after:
            md_lines.extend(generate_balance_table(balance_after))
    
    return md_lines


# ===== TopN 보고서 생성 함수 =====

def generate_topn_report(periods, trader_weight, title, n=5, include_full_tables=True):
    """공통 TopN 보고서 생성 함수"""
    md_lines = [f"# Top {n} {title}"]
    
    # 가장 극단적인 경우가 먼저 나오도록 정렬
    # (이미 topn_analysis.py에서 정렬된 결과를 그대로 사용)
    for i, period in enumerate(periods):
        idx = i + 1  # 1부터 시작하는 인덱스
        
        # 구간 헤더 추가 (순위 표시)
        md_lines.append(f"## {idx}위: {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append("")
        
        # 포지션 테이블 추가
        if include_full_tables and 'positions' in period and period['positions']:
            md_lines.extend(generate_position_table(period['positions']))
        
        # 트레이더 요약 테이블 추가
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.extend(generate_trader_summary_table(trader_summaries, trader_weight))
        
        # 자산 테이블 추가 (옵션)
        if include_full_tables and 'balance_after' in period and period['balance_after']:
            md_lines.extend(generate_balance_table(period['balance_after']))
    
    return md_lines


def generate_topn_concurrent_positions_md(periods, trader_weight, n=5):
    """동시 포지션 수 기준 최대 Top N 구간 보고서"""
    md_lines = [f"# Top {n} 최대 동시 포지션 구간"]
    
    for idx, period in enumerate(periods, 1):
        # 구간 헤더 추가
        md_lines.extend(generate_period_header(idx, period))
        md_lines.append(f"- 동시 포지션 수: {len(period['positions'])}")
        md_lines.append("")
        
        # 포지션 테이블 추가
        if 'positions' in period and period['positions']:
            md_lines.extend(generate_position_table(period['positions']))
        
        # 트레이더 요약 테이블 추가
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.extend(generate_trader_summary_table(trader_summaries, trader_weight))
        
        # 자산 테이블 추가
        if 'balance_after' in period and period['balance_after']:
            md_lines.extend(generate_balance_table(period['balance_after']))
    
    return md_lines

# 단순 롱/숏 비율 관련 함수 제거 - 순노출 기준 보고서만 유지


def generate_topn_net_ratio_md(periods, trader_weight, n=5, direction='long'):
    """롱/숏 순비율(%) 기준 최대 Top N 구간 보고서"""
    title = "롱 순비율(%) 최대 구간" if direction=='long' else "숏 순비율(%) 최대 구간"
    
    # 기본 보고서 템플릿 사용
    return generate_topn_report(periods, trader_weight, title, n, include_full_tables=True)
