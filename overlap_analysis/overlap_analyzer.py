from datetime import datetime
from typing import List, Dict
from config import model_initial_capital

def analyze_all_periods(all_positions: List[Dict], all_positions_raw: List[Dict], current_balance, trader_weight):
    """
    포지션 데이터를 분석하여 기간별 중첩 포지션을 계산합니다.
    
    수정: 트레이더별 활성 포지션 데이터를 정확하게 추적하도록 개선했습니다.
    - 시간 기준으로 포지션을 정확히 추적
    - 해당 시점에 활성화된 포지션만 처리
    """
    # 모델 포트폴리오 자본 초기화
    model_capital = model_initial_capital
    model_capital_history = []
    
    # 중요한 시간대 식별 (포지션 열림, 닫힘 시간을 모두 포함)
    unique_times = set()
    for pos in all_positions_raw:
        unique_times.add(pos['open'])
        unique_times.add(pos['close'])
    
    # 시간 순서로 정렬
    time_events = sorted(unique_times)
    
    # 마지막 기록된 시간 출력 (디버깅용)
    if time_events:
        last_time = max(time_events)
        print(f"\n[INFO] Last recorded time event: {last_time}")

    periods = []
    last_time = None
    
    # 각 시간대에 대해 분석
    for current_time in time_events:
        # 구간 분석 (new period starts)
        if last_time is not None:
            # 현재 시점에서 모든 활성 포지션 수집 - 시간 기준으로 정확히 필터링
            active_positions = []
            
            # 각 트레이더별 활성 포지션 확인
            active_traders = set()
            for pos in all_positions_raw:
                # 포지션 활성 여부 판단 로직 전면 재정비
                # report_generator.py와 일관된 경계 조건 처리 방식 구현
                
                # 경계 케이스 처리 개선:
                # 1. 포지션이 구간의 시작 시간에 열린 경우 - 포함시킴
                # 2. 포지션이 구간의 종료 시간에 열린 경우 - 제외함(다음 구간에 포함)
                if pos['open'] == current_time:
                    # 현재 구간의 끝에서 시작하는 경우 제외 - 다음 구간에 포함될 것임
                    continue
                    
                # 1. 일반적인 활성 포지션 조건: 포지션 오픈 시간 < 현재 시간 < 포지션 클로즈 시간
                active_during_period = (pos['open'] < current_time < pos['close'])
                
                # 2. 포지션이 현재 구간에서 열리는 경우 (시작 시간에 열리는 경우 포함)
                opens_at_start = (pos['open'] == last_time and pos['close'] > current_time)
                
                # 3. 포지션이 현재 구간 내에 열리는 경우
                opens_during_period = (last_time < pos['open'] < current_time)
                
                # 4. 포지션이 현재 구간에서 닫히는 경우 (종료 시간에 닫히는 경우 포함)
                closes_during_period = (last_time < pos['close'] <= current_time)
                
                # 5. 포지션이 구간 시작 전에 열리고 현재 구간 동안 활성화
                spans_period = (pos['open'] < last_time and pos['close'] > last_time)
                
                # 활성 포지션 조건을 하나라도 만족하는 경우
                if active_during_period or opens_at_start or opens_during_period or closes_during_period or spans_period:
                    active_positions.append(pos)
                    active_traders.add(pos['trader'])
            
            # 디버깅 로그
            if current_time.strftime('%Y-%m-%d %H:%M:%S') in ['2025-05-19 22:37:00', '2025-05-20 00:08:00']:
                print(f"\n[DEBUG] Period: {current_time}")
                # 트레이더별 활성 포지션 개수
                trader_pos_counts = {}
                for trader in current_balance.keys():
                    trader_pos_counts[trader] = sum(1 for p in active_positions if p['trader'] == trader)
                print(f"Active positions by trader: {trader_pos_counts}")
                print(f"Active traders: {active_traders}")
                print(f"Total active positions: {len(active_positions)}")
            
            # === 롱/숏/순 포지션 합계 및 비율 계산 ===
            long_open = {trader: 0.0 for trader in current_balance}
            short_open = {trader: 0.0 for trader in current_balance}
            
            for p in active_positions:
                trader = p['trader']
                try:
                    # size 가져오기 - 포지션 크기 계산용
                    size = float(str(p['size']).replace('USDT','').replace(',','').strip())
                    size = abs(size)
                except:
                    size = 0.0
                    
                if trader in long_open and str(p['direction']).lower().startswith('long'):
                    long_open[trader] += size
                elif trader in short_open and (str(p['direction']).lower().startswith('short') or str(p['direction']).lower().startswith('-')):
                    short_open[trader] += size
            trader_summaries = {}
            for trader in current_balance:
                bal = current_balance[trader]
                net_amt = long_open[trader] - short_open[trader]
                long_ratio = (long_open[trader] / bal * 100) if bal > 0 else 0.0
                short_ratio = (short_open[trader] / bal * 100) if bal > 0 else 0.0
                net_ratio = (net_amt / bal * 100) if bal > 0 else 0.0
                if net_amt > 0:
                    net_dir = '롱'
                elif net_amt < 0:
                    net_dir = '숏'
                else:
                    net_dir = '중립'
                trader_summaries[trader] = {
                    'long_amt': long_open[trader],
                    'short_amt': short_open[trader],
                    'net_amt': net_amt,
                    'long_ratio': long_ratio,
                    'short_ratio': short_ratio,
                    'net_ratio': net_ratio,
                    'net_dir': net_dir
                }
            # === 구간 종료 후 자산 변화 (포지션 close == period end) ===
            closed_this_period = [p for p in all_positions_raw if p['close'] == current_time]
            
            # 모델 포트폴리오 구간 시작 자본 기록
            model_capital_at_period_start = model_capital
            
            # 구간 PnL 계산하여 트레이더와 모델 포트폴리오 자본 업데이트
            model_pnl = 0.0
            trader_pnl_info = {}
            
            for p in closed_this_period:
                trader = p['trader']
                try:
                    pnl = float(str(p.get('realized_pnl', '0')).replace('USDT','').replace(',','').strip())
                    pnl_pct = 0.0
                    try:
                        pnl_pct_str = str(p.get('realized_pnl_pct', '0')).replace('%', '').strip()
                        pnl_pct = float(pnl_pct_str) / 100  # %를 소수로 변환
                    except:
                        # PnL % 데이터가 없으면 자산 대비 계산
                        if current_balance[trader] > 0:
                            pnl_pct = pnl / current_balance[trader]
                            
                    # 트레이더 수익률에 가중치를 적용한 모델 포트폴리오 수익률 기여도 계산
                    if trader in trader_weight:
                        trader_contribution = pnl_pct * trader_weight[trader]
                        model_pnl += model_capital * trader_contribution
                        
                        # 트레이더별 PnL 정보 저장
                        trader_pnl_info[trader] = {
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'contribution': trader_contribution
                        }
                        
                except Exception as e:
                    print(f"PnL 계산 오류: {e}")
                    pnl = 0.0
                    
                if trader in current_balance:
                    current_balance[trader] += pnl
            
            # 모델 포트폴리오 자본 업데이트
            model_capital += model_pnl
            # === period dict에 상세 정보 저장 ===
            periods.append({
                'start': last_time,
                'end': current_time,
                'positions': active_positions,
                'duration_min': (current_time - last_time).total_seconds() / 60,
                'trader_summaries': trader_summaries,
                'balance_after': current_balance.copy(),
                'model_capital': {
                    'start': model_capital_at_period_start,
                    'end': model_capital,
                    'change_pct': ((model_capital / model_capital_at_period_start) - 1) * 100 if model_capital_at_period_start > 0 else 0.0,
                    'trader_pnl_info': trader_pnl_info
                }
            })
            
            # 모델 포트폴리오 자본 기록
            model_capital_history.append({
                'time': current_time,
                'capital': model_capital
            })
        last_time = current_time
    return [p for p in periods if p['duration_min'] > 0.01]
