import pandas as pd
import re
import os
from datetime import datetime
from config import target_files, trader_weight, initial_balance, TRADER_FILTERS
from data_loader import load_all_positions, detect_time_columns, parse_and_convert
from overlap_analyzer import analyze_all_periods
from report_generator import generate_markdown
# 단순 롱/숏 관련 함수 제거
from topn_analysis import top_n_by_concurrent_positions, top_n_by_net_ratio
from report_generator import generate_topn_concurrent_positions_md, generate_topn_net_ratio_md
from portfolio_net_exposure_export import export_portfolio_net_exposure

if __name__ == "__main__":
    # 결과 저장 폴더 지정
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 초기 자산은 config에서 가져옴
    current_balance = initial_balance.copy()

    # 데이터 로드 시작일 설정 (2024년 9월 1일)
    # start_date = datetime(2024, 9, 1)
    start_date = None
    
    # 전체 포지션 데이터를 로드하고 날짜 필터링 적용
    all_positions = []
    all_positions_raw = []
    filtered_count = 0  # 필터링으로 제외된 포지션 수
    
    # 전체 포지션 데이터 로드
    for trader, abs_path in target_files.items():
        df = pd.read_csv(abs_path)
        open_col, close_col, mode = detect_time_columns(df)
        
        # 해당 트레이더에 대한 필터 설정 가져오기
        trader_filter = TRADER_FILTERS.get(trader.lower(), {})  # 소문자로 비교
        position_mode = trader_filter.get("position_mode", "all")  # 기본값은 all (모든 포지션)
        excluded_bases = trader_filter.get("excluded_bases", [])
        
        for _, row in df.iterrows():
            try:
                open_t = parse_and_convert(str(row[open_col]), mode)
                close_t = parse_and_convert(str(row[close_col]), mode)
                
                # 날짜 필터링 적용 (필터가 있을 때만)
                if start_date and open_t < start_date:
                    filtered_count += 1
                    continue
                
                symbol = row.get('Symbol', '')
                direction = row.get('Direction', '')
                
                # 트레이더 필터 적용 - position_mode 기준
                if position_mode == "long_only" and direction.lower() != 'long':
                    filtered_count += 1
                    continue
                elif position_mode == "short_only" and direction.lower() != 'short':
                    filtered_count += 1
                    continue
                
                # 트레이더 필터 적용 - excluded_bases 기준
                if excluded_bases and any(base in symbol for base in excluded_bases):
                    filtered_count += 1
                    continue
                
                # 기본 포지션 정보 추가
                all_positions.append({
                    'trader': trader,
                    'open': open_t,
                    'close': close_t,
                })
                
                # Max Size USDT 필드 가져오기 (주요 필드)
                max_size_usdt = row.get('Max Size USDT', 0)
                
                # 사이즈 추출 (size) - 트레이더 요약 계산용
                try:
                    # 숫자 형식으로 변환
                    if isinstance(max_size_usdt, str):
                        # 문자열에서 숫자만 추출
                        numeric_str = re.sub(r'[^0-9.]', '', max_size_usdt)
                        size = float(numeric_str) if numeric_str else 0.0
                    else:
                        # 이미 숫자인 경우
                        size = float(max_size_usdt)
                except Exception as e:
                    print(f"[WARN] Error extracting size from {max_size_usdt}: {e}")
                    size = 0.0
                    
                # 거래량 추출 (volume) - 표시용
                if 'Volume' in row:
                    # Volume 필드가 있으면 직접 사용
                    volume = row['Volume']
                else:
                    # Volume 필드가 없으면 Max Size USDT의 2배로 계산
                    try:
                        volume = f"{size * 2:.2f} USDT"
                    except Exception as e:
                        print(f"[WARN] Error calculating volume from size {size}: {e}")
                        volume = f"{max_size_usdt} (x2)"  # 오류 발생 시 대체 값
                    
                all_positions_raw.append({
                    'trader': trader,
                    'symbol': symbol,
                    'direction': direction,
                    'size': size,  # 포지션 크기(톨더 요약 계산용)
                    'volume': volume,  # 거래량 데이터 (표시용)
                    'open': open_t,
                    'close': close_t,
                    'realized_pnl': row.get('Realized PnL', ''),
                    'realized_pnl_pct': row.get('Realized PnL %', ''),
                })
            except Exception as e:
                print(f"[WARN] {trader} row skipped ({e})")
    
    print(f"총 {len(all_positions)}개 포지션 로드 완료. (필터링으로 제외된 포지션: {filtered_count}개)")
    if start_date:
        print(f"[INFO] 데이터 로드 시작일: {start_date.strftime('%Y-%m-%d')}")
    else:
        print("[INFO] 날짜 필터링 없이 모든 데이터 로드")

    periods = analyze_all_periods(all_positions, all_positions_raw, current_balance, trader_weight)
    print(f"\n=== 모든 구간 분석 결과 (UTC+9, 한국시간) ===")
    print(f"총 {len(periods)}개 구간이 탐지되었습니다.")

    # === 순노출(%) csv/그래프 자동 저장 ===
    export_portfolio_net_exposure(periods, out_dir=OUTPUT_DIR)

    # 마크다운 파일 생성

    md_lines = generate_markdown(periods, current_balance, trader_weight)
    md_path = os.path.join(OUTPUT_DIR, 'position_overlap_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"\n분석 결과가 {md_path} 에 저장되었습니다.")

    # Top 5 최대 순노출(USDT) 구간 리포트 생성 로직 제거

    # Top 5 최대 동시 포지션 구간 마크다운
    top5_conc = top_n_by_concurrent_positions(periods, n=5)
    md_top5_conc = generate_topn_concurrent_positions_md(top5_conc, trader_weight, n=5)
    with open(os.path.join(OUTPUT_DIR, 'top5_concurrent_positions_periods.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_top5_conc))
    print("Top 5 최대 동시 포지션 구간 리포트가 저장되었습니다.")

    # 단순 롱/숏 관련 함수 제거 - 순노출 기준 리포트만 유지
    # 해당 파일들이 존재하는 경우 삭제
    for file_name in ['top5_long_ratio_periods.md', 'top5_short_ratio_periods.md']:
        file_path = os.path.join(OUTPUT_DIR, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"{file_name} 파일이 삭제되었습니다.")
            except Exception as e:
                print(f"{file_name} 파일 삭제 실패: {e}")

    # Top 5 롱 순비율(%) 최대 구간
    top5_net_long = top_n_by_net_ratio(periods, n=5, direction='long')
    md_top5_net_long = generate_topn_net_ratio_md(top5_net_long, trader_weight, n=5, direction='long')
    with open(os.path.join(OUTPUT_DIR, 'top5_net_long_ratio_periods.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_top5_net_long))
    print("Top 5 롱 순비율(%) 최대 구간 리포트가 저장되었습니다.")

    # Top 5 숏 순비율(%) 최대 구간
    top5_net_short = top_n_by_net_ratio(periods, n=5, direction='short')
    md_top5_net_short = generate_topn_net_ratio_md(top5_net_short, trader_weight, n=5, direction='short')
    with open(os.path.join(OUTPUT_DIR, 'top5_net_short_ratio_periods.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_top5_net_short))
    print("Top 5 숏 순비율(%) 최대 구간 리포트가 저장되었습니다.")
