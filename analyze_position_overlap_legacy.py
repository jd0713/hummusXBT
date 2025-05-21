import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

# 트레이더별 파일 경로와 이름 매핑
target_files = {
    'Chatosil': 'analysis_results/Chatosil/overall/analyzed_data.csv',
    'hummusXBT': 'analysis_results/hummusXBT/period2/analyzed_data.csv',
    'ONLYCANDLE': 'analysis_results/ONLYCANDLE/overall/analyzed_data.csv',
    'RebelOfBabylon': 'analysis_results/RebelOfBabylon/overall/analyzed_data.csv',
    'Ohtanishohei': 'analysis_results/Ohtanishohei/overall/analyzed_data.csv'
}

trader_weight = {
    'hummusXBT': 1/3,
    'ONLYCANDLE': 1/4,
    'RebelOfBabylon': 0,
    'Chatosil': 1/5,
    'Ohtanishohei':  1/5
}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 시간 컬럼 자동 감지 함수 (KST 우선)
def detect_time_columns(df):
    # KST(UTC+9) 컬럼 우선 사용, 없으면 UTC 컬럼 사용
    kst_candidates = [
        ('Open_Time_KST', 'Close_Time_KST'),
        ('Open_Time_KST_Str', 'Close_Time_KST_Str'),
    ]
    utc_candidates = [
        ('Open_Time_UTC', 'Close_Time_UTC'),
    ]
    for open_col, close_col in kst_candidates:
        if open_col in df.columns and close_col in df.columns:
            return open_col, close_col, 'KST'
    for open_col, close_col in utc_candidates:
        if open_col in df.columns and close_col in df.columns:
            return open_col, close_col, 'UTC'
    raise ValueError(f"Cannot find open/close time columns in {df.columns}")

# 시간 파싱 함수 (KST면 그대로, UTC면 +9h)
def parse_and_convert(time_str, mode):
    # KST: 그대로 파싱, UTC: +9h
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M'):
        try:
            dt = datetime.strptime(time_str, fmt)
            if mode == 'KST':
                return dt
            else:
                return dt + timedelta(hours=9)
        except Exception:
            continue
    # UNIX timestamp (초 단위)
    try:
        ts = float(time_str)
        if ts > 1e12:  # ms 단위
            ts = ts / 1000
        dt = datetime.utcfromtimestamp(ts)
        if mode == 'KST':
            return dt
        else:
            return dt + timedelta(hours=9)
    except Exception:
        pass
    raise ValueError(f"Unknown datetime format: {time_str}")

# 모든 포지션을 하나의 리스트로 통합
def load_all_positions():
    all_positions = []
    for trader, rel_path in target_files.items():
        abs_path = os.path.join(ROOT_DIR, rel_path)
        df = pd.read_csv(abs_path)
        open_col, close_col, mode = detect_time_columns(df)
        for _, row in df.iterrows():
            try:
                open_t = parse_and_convert(str(row[open_col]), mode)
                close_t = parse_and_convert(str(row[close_col]), mode)
                all_positions.append({
                    'trader': trader,
                    'open': open_t,
                    'close': close_t,
                })
            except Exception as e:
                print(f"[WARN] {trader} row skipped ({e})")
    return all_positions

# 모든 포지션의 상세 정보도 함께 관리
# 겹치는 구간(트레이더 1명만 있는 구간 포함) 분석 및 포지션 상세 추출
def analyze_all_periods(all_positions, all_positions_raw):
    # 이벤트: (시점, open/close, 포지션 인덱스)
    events = []
    for i, pos in enumerate(all_positions):
        events.append((pos['open'], 'open', i))
        events.append((pos['close'], 'close', i))
    events.sort()

    active_pos_idx = set()
    last_time = None
    periods = []
    for t, ev, idx in events:
        if last_time is not None and len(active_pos_idx) > 0:
            # 해당 구간에서 활성화된 포지션 상세 추출
            active_positions = [all_positions_raw[i] for i in active_pos_idx]
            periods.append({
                'start': last_time,
                'end': t,
                'positions': active_positions,
                'duration_min': (t - last_time).total_seconds() / 60,
            })
        if ev == 'open':
            active_pos_idx.add(idx)
        else:
            active_pos_idx.discard(idx)
        last_time = t
    return [p for p in periods if p['duration_min'] > 0.01]

# 순노출(%) 저장용 리스트
global_portfolio_net_exposures = []
global_portfolio_times = []

def analyze_all_periods_with_exposure(all_positions, all_positions_raw):
    # 이벤트: (시점, open/close, 포지션 인덱스)
    events = []
    for i, pos in enumerate(all_positions):
        events.append((pos['open'], 'open', i))
        events.append((pos['close'], 'close', i))
    events.sort()

    active_pos_idx = set()
    last_time = None
    periods = []
    for t, ev, idx in events:
        if last_time is not None and len(active_pos_idx) > 0:
            # 해당 구간에서 활성화된 포지션 상세 추출
            active_positions = [all_positions_raw[i] for i in active_pos_idx]
            periods.append({
                'start': last_time,
                'end': t,
                'positions': active_positions,
                'duration_min': (t - last_time).total_seconds() / 60,
            })
        if ev == 'open':
            active_pos_idx.add(idx)
        else:
            active_pos_idx.discard(idx)
        last_time = t
    return [p for p in periods if p['duration_min'] > 0.01]

if __name__ == "__main__":
    # 포지션 상세 정보도 함께 로드
    all_positions = []
    all_positions_raw = []
    for trader, rel_path in target_files.items():
        abs_path = os.path.join(ROOT_DIR, rel_path)
        df = pd.read_csv(abs_path)
        open_col, close_col, mode = detect_time_columns(df)
        # 각 포지션별 상세 정보 저장
        for _, row in df.iterrows():
            try:
                open_t = parse_and_convert(str(row[open_col]), mode)
                close_t = parse_and_convert(str(row[close_col]), mode)
                # Chatosil의 롱 거래, 또는 BTC/ETH/XRP/BNB 거래(Chatosil만)는 분석에서 제외
                if trader == 'Chatosil':
                    direction = str(row.get('Direction', '')).lower()
                    symbol = str(row.get('Symbol', '')).upper()
                    if direction.startswith('long'):
                        continue  # 롱 거래는 제외
                    if symbol in {'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BNBUSDT'}:
                        continue  # BTC/ETH/XRP/BNB 거래는 제외
                all_positions.append({
                    'trader': trader,
                    'open': open_t,
                    'close': close_t,
                })
                all_positions_raw.append({
                    'trader': trader,
                    'symbol': row.get('Symbol', ''),
                    'direction': row.get('Direction', ''),
                    'size': row.get('Max Size USDT', row.get('Size_USDT_Numeric', 0)),
                    'open': open_t,
                    'close': close_t,
                    'realized_pnl': row.get('Realized PnL', ''),
                    'realized_pnl_pct': row.get('Realized PnL %', ''),
                })
            except Exception as e:
                print(f"[WARN] {trader} row skipped ({e})")
    print(f"총 {len(all_positions)}개 포지션 로드 완료.")

    periods = analyze_all_periods(all_positions, all_positions_raw)
    print(f"\n=== 모든 구간 분석 결과 (UTC+9, 한국시간) ===")
    print(f"총 {len(periods)}개 구간이 탐지되었습니다.")

    # Markdown 파일 생성
    md_lines = ["# 트레이더 포지션 구간별 롱/숏/넷 분석 결과\n"]
    # 트레이더별 초기 원금 세팅
    initial_balance = {
        'hummusXBT': 2000000,
        'Chatosil': 2000000,
        'RebelOfBabylon': 1000000,
        'ONLYCANDLE': 1000000,
        'Ohtanishohei': 1500000
    }
    current_balance = initial_balance.copy()

    # === Top 5 트레이더 동시 포지션 구간 저장용 리스트 ===
    active_trader_periods = []
    # === Top N 저장용 리스트 ===
    abs_net_ratio_periods = []
    net_exposure_periods = []
    period_md_blocks = []
    for idx, period in enumerate(periods, 1):
        # 순노출(%) 계산용 임시 변수
        portfolio_net = 0.0

        long_open = {trader: 0.0 for trader in current_balance}
        short_open = {trader: 0.0 for trader in current_balance}
        for p in all_positions_raw:
            # 구간 내에 열려있는 포지션: open < period.end AND close > period.start
            if p['open'] < period['end'] and p['close'] > period['start']:
                # 롱/숏 구분
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
        # 각 트레이더별 순비율 계산 및 포트폴리오 net exposure 계산
        for trader in current_balance:
            bal = current_balance[trader]
            net_amt = long_open[trader] - short_open[trader]
            net_ratio = (net_amt / bal * 100) if bal > 0 else 0.0
            weight = trader_weight.get(trader, 0.0)
            portfolio_net += weight * net_ratio
        global_portfolio_net_exposures.append(portfolio_net)
        global_portfolio_times.append(period['end'])
        tmp_md_block = []
        tmp_md_block.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분\n")
        md_lines.append(f"## [{idx}] {period['start']} ~ {period['end']} | {period['duration_min']:.1f}분\n")
        # 실제 해당 구간에 열려있는 포지션만 추출
        active_positions = []
        for p in all_positions_raw:
            # 구간 내에 열려있는 포지션: open < period.end AND close > period.start
            if p['open'] < period['end'] and p['close'] > period['start']:
                active_positions.append(p)
        # === Top 5 추출용: 이 구간에 포지션이 열린 트레이더 수 ===
        active_trader_set = set([p['trader'] for p in active_positions])
        active_trader_periods.append({
            'idx': idx,
            'start': period['start'],
            'end': period['end'],
            'duration_min': period['duration_min'],
            'num_active_traders': len(active_trader_set),
            'traders': list(active_trader_set),
            'active_positions': active_positions.copy(),
            # 아래 값들은 구간별 계산 후 할당
        })
        tmp_md_block.append("| 트레이더 | 심볼 | 방향 | 규모(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
        tmp_md_block.append("|---|---|---|---|---|---|---|---|")
        md_lines.append("| 트레이더 | 심볼 | 방향 | 규모(USDT) | 오픈 | 클로즈 | Realized PnL | Realized PnL % |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        long_sum = 0.0
        short_sum = 0.0
        for p in active_positions:
            try:
                size = float(str(p['size']).replace('USDT','').replace(',','').strip())
                size = abs(size)  # 항상 절대값으로 처리
            except:
                size = 0.0
            # Realized PnL 및 Realized PnL % 추출 (없으면 빈 문자열)
            realized_pnl = p.get('realized_pnl', '') if isinstance(p, dict) else ''
            realized_pnl_pct = p.get('realized_pnl_pct', '') if isinstance(p, dict) else ''
            if str(p['direction']).lower().startswith('long'):
                long_sum += size
            elif str(p['direction']).lower().startswith('short'):
                short_sum += size
            tmp_md_block.append(f"| {p['trader']} | {p['symbol']} | {p['direction']} | {size:.2f} | {p['open']} | {p['close']} | {realized_pnl} | {realized_pnl_pct} |")
            md_lines.append(f"| {p['trader']} | {p['symbol']} | {p['direction']} | {size:.2f} | {p['open']} | {p['close']} | {realized_pnl} | {realized_pnl_pct} |")
        net = long_sum - short_sum
        net_dir = '롱' if net > 0 else ('숏' if net < 0 else '중립')
        tmp_md_block.append("")
        tmp_md_block.append(f"- 롱 포지션 합: **{long_sum:,.2f} USDT**")
        tmp_md_block.append(f"- 숏 포지션 합: **{short_sum:,.2f} USDT**")
        tmp_md_block.append(f"- Net 포지션: **{net_dir} ({abs(net):,.2f} USDT)**\n")
        md_lines.append("")
        md_lines.append(f"- 롱 포지션 합: **{long_sum:,.2f} USDT**")
        md_lines.append(f"- 숏 포지션 합: **{short_sum:,.2f} USDT**")
        md_lines.append(f"- Net 포지션: **{net_dir} ({abs(net):,.2f} USDT)**\n")

        # ====== 트레이더별 롱/숏 오픈 포지션 합계 및 자산 대비 비율, 순방향 ======
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
        tmp_md_block.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
        tmp_md_block.append("|---|---|---|---|---|---|---|---|---|")
        md_lines.append("| 트레이더 | Weight(%) | 롱 오픈(USDT) | 롱 비율(%) | 숏 오픈(USDT) | 숏 비율(%) | 순 오픈(USDT) | 순방향 | 순비율(%) |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        # 트레이더별 weight 설정

        # weight 적용 포트폴리오 순노출/방향/레버리지 계산용 변수
        portfolio_net = 0.0
        portfolio_leverage = 0.0
        abs_net_ratio_sum = 0.0
        net_exposure_val = 0.0
        for trader in sorted(current_balance.keys()):
            bal = current_balance[trader]
            long_amt = long_open[trader]
            short_amt = short_open[trader]
            net_amt = long_amt - short_amt
            long_ratio = (long_amt / bal * 100) if bal > 0 else 0.0
            short_ratio = (short_amt / bal * 100) if bal > 0 else 0.0
            net_ratio = (abs(net_amt) / bal * 100) if bal > 0 else 0.0
            abs_net_ratio_sum += weight * abs(net_ratio)
            if net_amt > 0:
                net_dir = '롱'
            elif net_amt < 0:
                net_dir = '숏'
            else:
                net_dir = '중립'
            weight_pct = trader_weight.get(trader, 0.0) * 100
            tmp_md_block.append(f"| {trader} | {weight_pct:.2f} | {long_amt:,.2f} | {long_ratio:.2f} | {short_amt:,.2f} | {short_ratio:.2f} | {net_amt:,.2f} | {net_dir} | {net_ratio:.2f} |")
            md_lines.append(f"| {trader} | {weight_pct:.2f} | {long_amt:,.2f} | {long_ratio:.2f} | {short_amt:,.2f} | {short_ratio:.2f} | {net_amt:,.2f} | {net_dir} | {net_ratio:.2f} |")
            # 포트폴리오 계산(롱은 +, 숏은 -)
            weight = trader_weight.get(trader, 0.0)
            if net_amt > 0:
                portfolio_net += weight * net_ratio
                net_exposure_val += weight * net_ratio
            elif net_amt < 0:
                portfolio_net -= weight * net_ratio
                net_exposure_val -= weight * net_ratio
            # 레버리지는 롱+숏 모두 합산
            portfolio_leverage += weight * (long_ratio + short_ratio)
        tmp_md_block.append("")
        # 포트폴리오 순방향/순노출 표 (레버리지 삭제)
        if portfolio_net > 0:
            pf_dir = '롱'
        elif portfolio_net < 0:
            pf_dir = '숏'
        else:
            pf_dir = '중립'
        tmp_md_block.append("| 포트폴리오 순방향 | 순노출(%) | 순비율 절대값 합(포지션 배율) |")
        tmp_md_block.append("|---|---|---|")
        tmp_md_block.append(f"| {pf_dir} | {abs(portfolio_net):.2f} | {abs_net_ratio_sum:.2f} |")
        tmp_md_block.append("")
        md_lines.append("| 포트폴리오 순방향 | 순노출(%) | 순비율 절대값 합(포지션 배율) |")
        md_lines.append("|---|---|---|")
        md_lines.append(f"| {pf_dir} | {abs(portfolio_net):.2f} | {abs_net_ratio_sum:.2f} |")
        md_lines.append("")

        # ====== 자산 변화 추적 ======
        # 이번 구간에서 종료된 포지션(구간 끝과 close가 같은 포지션)
        closed_this_period = [p for p in all_positions_raw if p['close'] == period['end']]
        for p in closed_this_period:
            trader = p['trader']
            try:
                pnl = float(str(p.get('realized_pnl', '0')).replace('USDT','').replace(',','').strip())
            except:
                pnl = 0.0
            if trader in current_balance:
                current_balance[trader] += pnl
        # 구간 종료 후 트레이더별 자산 표
        tmp_md_block.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
        tmp_md_block.append("|---|---|")
        md_lines.append("| 트레이더 | 구간 종료 후 자산(USDT) |")
        md_lines.append("|---|---|")
        for trader in sorted(current_balance.keys()):
            tmp_md_block.append(f"| {trader} | {current_balance[trader]:,.2f} |")
            md_lines.append(f"| {trader} | {current_balance[trader]:,.2f} |")
        md_lines.append("")
        # === TopN용 값 저장 ===
        abs_net_ratio_periods.append({
            'idx': idx,
            'start': period['start'],
            'end': period['end'],
            'duration_min': period['duration_min'],
            'abs_net_ratio_sum': abs_net_ratio_sum,
            'active_positions': active_positions.copy(),
            'md_block': tmp_md_block[:],
        })
        net_exposure_periods.append({
            'idx': idx,
            'start': period['start'],
            'end': period['end'],
            'duration_min': period['duration_min'],
            'net_exposure_val': net_exposure_val,
            'active_positions': active_positions.copy(),
            'md_block': tmp_md_block[:],
        })
        # 동시 포지션 구간도 md_block 저장
        active_trader_periods[-1]['md_block'] = tmp_md_block[:]
        period_md_blocks.append(tmp_md_block[:])
    md_path = os.path.join(ROOT_DIR, 'position_overlap_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"\n분석 결과가 {md_path} 에 저장되었습니다.")

    # === Top 5 트레이더 동시 포지션 구간 저장 ===
    top5 = sorted(active_trader_periods, key=lambda x: x['num_active_traders'], reverse=True)[:5]
    top5_md = ["# Top 5 트레이더 동시 포지션 구간\n"]
    for item in top5:
        top5_md.append(f"## [{item['idx']}] {item['start']} ~ {item['end']} | {item['duration_min']:.1f}분 | 트레이더 수: {item['num_active_traders']}")
        top5_md.append(f"참여 트레이더: {', '.join(sorted(item['traders']))}")
        # 분석 결과 표/요약 포함
        # md_block은 해당 구간 분석 표/요약을 포함
        for line in item.get('md_block', []):
            top5_md.append(line)
        top5_md.append("")
    top5_md_path = os.path.join(ROOT_DIR, 'top5_active_periods.md')
    with open(top5_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(top5_md))
    print(f"Top 5 동시 포지션 구간 결과가 {top5_md_path} 에 저장되었습니다.")

    # === Top 5 순비율 절대값 합 구간 저장 ===
    top5_abs = sorted(abs_net_ratio_periods, key=lambda x: x['abs_net_ratio_sum'], reverse=True)[:5]
    abs_md = ["# Top 5 순비율 절대값 합(포지션 배율) 구간\n"]
    for item in top5_abs:
        abs_md.append(f"## [{item['idx']}] {item['start']} ~ {item['end']} | {item['duration_min']:.1f}분 | 순비율 절대값 합: {item['abs_net_ratio_sum']:.2f}")
        for line in item.get('md_block', []):
            abs_md.append(line)
        abs_md.append("")
    abs_md_path = os.path.join(ROOT_DIR, 'top5_abs_net_ratio_periods.md')
    with open(abs_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(abs_md))
    print(f"Top 5 순비율 절대값 합 구간 결과가 {abs_md_path} 에 저장되었습니다.")

    # === Top 5 순노출 음수 구간 저장 ===
    top5_neg = sorted(net_exposure_periods, key=lambda x: x['net_exposure_val'])[:5]
    neg_md = ["# Top 5 순노출 음수(가장 숏) 구간\n"]
    for item in top5_neg:
        neg_md.append(f"## [{item['idx']}] {item['start']} ~ {item['end']} | {item['duration_min']:.1f}분 | 순노출: {item['net_exposure_val']:.2f}")
        for line in item.get('md_block', []):
            neg_md.append(line)
        neg_md.append("")
    neg_md_path = os.path.join(ROOT_DIR, 'top5_negative_net_exposure_periods.md')
    with open(neg_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(neg_md))
    print(f"Top 5 순노출 음수 구간 결과가 {neg_md_path} 에 저장되었습니다.")

    # === Top 5 순노출 양수 구간 저장 ===
    top5_pos = sorted(net_exposure_periods, key=lambda x: x['net_exposure_val'], reverse=True)[:5]
    pos_md = ["# Top 5 순노출 양수(가장 롱) 구간\n"]
    for item in top5_pos:
        pos_md.append(f"## [{item['idx']}] {item['start']} ~ {item['end']} | {item['duration_min']:.1f}분 | 순노출: {item['net_exposure_val']:.2f}")
        for line in item.get('md_block', []):
            pos_md.append(line)
        pos_md.append("")
    pos_md_path = os.path.join(ROOT_DIR, 'top5_positive_net_exposure_periods.md')
    with open(pos_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pos_md))
    print(f"Top 5 순노출 양수 구간 결과가 {pos_md_path} 에 저장되었습니다.")

    # === 순노출(%) 그래프 및 CSV 저장 ===
    import matplotlib.pyplot as plt
    import csv
    if global_portfolio_net_exposures:
        # 그래프 저장
        plt.figure(figsize=(14,5))
        plt.plot(global_portfolio_times, global_portfolio_net_exposures, label='Net Exposure (%)', color='royalblue')
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.title('Portfolio Net Exposure Over Time')
        plt.xlabel('Time')
        plt.ylabel('Net Exposure (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('portfolio_net_exposure.png')
        plt.close()
        print('Portfolio net exposure graph saved as portfolio_net_exposure.png')
        # CSV 저장
        csv_path = os.path.join(ROOT_DIR, 'portfolio_net_exposure.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'net_exposure'])
            for t, v in zip(global_portfolio_times, global_portfolio_net_exposures):
                writer.writerow([t, v])
        print(f'Portfolio net exposure csv saved as {csv_path}')
