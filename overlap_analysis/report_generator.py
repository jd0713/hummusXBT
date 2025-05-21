# report_generator.py
# (초기 버전: legacy 코드의 마크다운 생성 로직만 분리)

from portfolio_metrics import calc_trader_metrics, calc_period_summary

def generate_markdown(periods, current_balance, trader_weight):
    md_lines = ["# 트레이더 포지션 구간별 롱/숏/넷 분석 결과\n"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분\n")
        active_positions = period['positions']
        # === 구간 내에서 활성화되어 있는 포지션을 출력 ===
        filtered_positions = []
        for p in active_positions:
            open_t = p.get('open')
            close_t = p.get('close')
            start = period['start']
            end = period['end']
            
            # 구간 강화 및 필터링 로직 전면 재개선
            # 경계에서의 문제 해결
            
            # 트레이더 요약에 포지션이 있는지 확인
            trader = p.get('trader', '')
            trader_summary = period.get('trader_summaries', {}).get(trader, {})
            long_amt = trader_summary.get('long_amt', 0)
            short_amt = trader_summary.get('short_amt', 0)
            has_position_in_summary = (long_amt > 0 or short_amt > 0)
            
            # 경계와 포지션 시간 비교를 위한 업데이트된 로직:
            # 1. 포지션이 구간 내에서 열리는 경우 (변경: 시작 경계의 경우 '다음' 구간에 포함)
            opens_during_period = (start < open_t < end)
            
            # 2. 포지션이 구간 내에서 닫히는 경우 (변경 없음: 종료 경계 포함)
            closes_during_period = (start < close_t <= end)
            
            # 3. 포지션이 구간 내내 활성 상태인 경우 (변경: 열린 시간이 구간 밖에 있고 닫힌 시간이 구간 이후)
            active_during_period = (open_t < start and close_t > end)
            
            # 4. 특별 경우: 포지션이 구간의 정확한 시작에 열리는 경우 (이 구간에 포함)
            opens_at_start = (open_t == start and close_t > end)
            
            # 5. 특별 경우: 포지션이 이전 구간에서 열리고 이 구간의 정확한 종료에 닫히는 경우 (이 구간에 포함)
            closes_at_end = (open_t < start and close_t == end)
            
            # 6. 특별 경우: 포지션이 경계 시점에 열리고 닫히는 경우
            # 이 경우 포지션은 '다음' 구간에 포함되어야 함
            boundary_case = (open_t == end)
            
            # 구간에 포함되어야 하는 경우만 필터링 (시간 경계에 있는 경우는 하나의 구간에만 표시되도록 조정)
            if ((opens_during_period or closes_during_period or active_during_period or 
                 opens_at_start or closes_at_end or has_position_in_summary) and 
                not boundary_case):
                filtered_positions.append(p)
        # 상세 포지션 테이블
        md_lines.append("| 트레이더 | 심볼 | 방향 | 거래량(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for p in filtered_positions:
            volume = p.get('volume', 0)
            realized_pnl = p.get('realized_pnl', '')
            realized_pnl_pct = p.get('realized_pnl_pct', '')
            md_lines.append(f"| {p['trader']} | {p.get('symbol', '')} | {p.get('direction', '')} | {volume} | {p.get('open', '')} | {p.get('close', '')} | {realized_pnl} | {realized_pnl_pct} |")
        md_lines.append("")
        # 트레이더별 롱/숏/순 요약 테이블
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for trader, s in trader_summaries.items():
                weight_pct = trader_weight.get(trader, 0.0) * 100
                md_lines.append(f"| {trader} | {weight_pct:.2f} | {s['long_amt']:.2f} | {s['long_ratio']:.2f} | {s['short_amt']:.2f} | {s['short_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
            md_lines.append("")
        # 구간 종료 후 자산 테이블
        balance_after = period.get('balance_after', None)
        if balance_after:
            md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
            md_lines.append("|---|---|")
            for trader, bal in balance_after.items():
                md_lines.append(f"| {trader} | {bal:,.2f} |")
            md_lines.append("")
    return md_lines


def generate_topn_net_exposure_md(periods, trader_weight, n=5):
    md_lines = [f"# Top {n} 최대 순노출(USDT) 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        trader_summaries = period.get('trader_summaries', {})
        md_lines.append("| 트레이더 | 순 오픈(USDT) | 순방향 | 순비율(%) | Weight(%) |")
        md_lines.append("|---|---|---|---|---|")
        for trader, s in trader_summaries.items():
            weight_pct = trader_weight.get(trader, 0.0) * 100
            md_lines.append(f"| {trader} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} | {weight_pct:.2f} |")
        md_lines.append("")
    return md_lines

def generate_topn_concurrent_positions_md(periods, trader_weight, n=5):
    md_lines = [f"# Top {n} 최대 동시 포지션 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append(f"- 동시 포지션 수: {len(period['positions'])}")
        # 상세 포지션 테이블
        active_positions = period['positions']
        md_lines.append("| 트레이더 | 심볼 | 방향 | 거래량(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for p in active_positions:
            volume = p.get('volume', 0)
            realized_pnl = p.get('realized_pnl', '')
            realized_pnl_pct = p.get('realized_pnl_pct', '')
            md_lines.append(f"| {p['trader']} | {p.get('symbol', '')} | {p.get('direction', '')} | {volume} | {p.get('open', '')} | {p.get('close', '')} | {realized_pnl} | {realized_pnl_pct} |")
        md_lines.append("")
        # 트레이더별 롱/숏/순 요약 테이블
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for trader, s in trader_summaries.items():
                weight_pct = trader_weight.get(trader, 0.0) * 100
                md_lines.append(f"| {trader} | {weight_pct:.2f} | {s['long_amt']:.2f} | {s['long_ratio']:.2f} | {s['short_amt']:.2f} | {s['short_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
            md_lines.append("")
        # 구간 종료 후 자산 테이블
        balance_after = period.get('balance_after', None)
        if balance_after:
            md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
            md_lines.append("|---|---|")
            for trader, bal in balance_after.items():
                md_lines.append(f"| {trader} | {bal:,.2f} |")
            md_lines.append("")
    return md_lines

def generate_topn_net_exposure_md(periods, trader_weight, n=5):
    md_lines = [f"# Top {n} 최대 순노출(USDT) 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        trader_summaries = period.get('trader_summaries', {})
        md_lines.append("| 트레이더 | 순 오픈(USDT) | 순방향 | 순비율(%) | Weight(%) |")
        md_lines.append("|---|---|---|---|---|")
        for trader, s in trader_summaries.items():
            weight_pct = trader_weight.get(trader, 0.0) * 100
            md_lines.append(f"| {trader} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} | {weight_pct:.2f} |")
        md_lines.append("")
    return md_lines

def generate_topn_long_ratio_md(periods, n=5):
    md_lines = [f"# Top {n} 롱 비율(%) 최대 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append("| 트레이더 | 롱 비율(%) | 롱 오픈(USDT) | 순방향 | 순비율(%) |")
        md_lines.append("|---|---|---|---|---|")
        for trader, s in period.get('trader_summaries', {}).items():
            md_lines.append(f"| {trader} | {s['long_ratio']:.2f} | {s['long_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
        md_lines.append("")
    return md_lines

def generate_topn_concurrent_positions_md(periods, trader_weight, n=5):
    md_lines = [f"# Top {n} 최대 동시 포지션 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append(f"- 동시 포지션 수: {len(period['positions'])}")
        # 상세 포지션 테이블
        active_positions = period['positions']
        md_lines.append("| 트레이더 | 심볼 | 방향 | 거래량(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for p in active_positions:
            volume = p.get('volume', 0)
            realized_pnl = p.get('realized_pnl', '')
            realized_pnl_pct = p.get('realized_pnl_pct', '')
            md_lines.append(f"| {p['trader']} | {p.get('symbol', '')} | {p.get('direction', '')} | {volume} | {p.get('open', '')} | {p.get('close', '')} | {realized_pnl} | {realized_pnl_pct} |")
        md_lines.append("")
        # 트레이더별 롱/숏/순 요약 테이블
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for trader, s in trader_summaries.items():
                weight_pct = trader_weight.get(trader, 0.0) * 100
                md_lines.append(f"| {trader} | {weight_pct:.2f} | {s['long_amt']:.2f} | {s['long_ratio']:.2f} | {s['short_amt']:.2f} | {s['short_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
            md_lines.append("")
        # 구간 종료 후 자산 테이블
        balance_after = period.get('balance_after', None)
        if balance_after:
            md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
            md_lines.append("|---|---|")
            for trader, bal in balance_after.items():
                md_lines.append(f"| {trader} | {bal:,.2f} |")
            md_lines.append("")
    return md_lines


def generate_topn_short_ratio_md(periods, n=5):
    md_lines = [f"# Top {n} 숏 비율(%) 최대 구간"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append("| 트레이더 | 숏 비율(%) | 숏 오픈(USDT) | 순방향 | 순비율(%) |")
        md_lines.append("|---|---|---|---|---|")
        for trader, s in period.get('trader_summaries', {}).items():
            md_lines.append(f"| {trader} | {s['short_ratio']:.2f} | {s['short_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
        md_lines.append("")
    return md_lines

def generate_topn_net_ratio_md(periods, trader_weight, n=5, direction='long'):
    title = "롱 순비율(%) 최대 구간" if direction=='long' else "숏 순비율(%) 최대 구간"
    md_lines = [f"# Top {n} {title}"]
    for idx, period in enumerate(periods, 1):
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분")
        md_lines.append("| 트레이더 | 순비율(%) | 순 오픈(USDT) | 순방향 | 롱 오픈(USDT) | 숏 오픈(USDT) | Weight(%) |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for trader, s in period.get('trader_summaries', {}).items():
            weight_pct = trader_weight.get(trader, 0.0) * 100
            md_lines.append(f"| {trader} | {s['net_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['long_amt']:.2f} | {s['short_amt']:.2f} | {weight_pct:.2f} |")
        md_lines.append("")
        trader_summaries = period.get('trader_summaries', {})
        if trader_summaries:
            md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for trader, s in trader_summaries.items():
                weight_pct = trader_weight.get(trader, 0.0) * 100
                md_lines.append(f"| {trader} | {weight_pct:.2f} | {s['long_amt']:.2f} | {s['long_ratio']:.2f} | {s['short_amt']:.2f} | {s['short_ratio']:.2f} | {s['net_amt']:.2f} | {s['net_dir']} | {s['net_ratio']:.2f} |")
            md_lines.append("")
        # 구간 종료 후 자산 테이블
        balance_after = period.get('balance_after', None)
        if balance_after:
            md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
            md_lines.append("|---|---|")
            for trader, bal in balance_after.items():
                md_lines.append(f"| {trader} | {bal:,.2f} |")
            md_lines.append("")
    return md_lines
