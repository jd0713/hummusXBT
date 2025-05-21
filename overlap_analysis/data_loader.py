import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional

def detect_time_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
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

def parse_and_convert(time_str: str, mode: str) -> datetime:
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M'):
        try:
            dt = datetime.strptime(time_str, fmt)
            if mode == 'KST':
                return dt
            else:
                return dt + timedelta(hours=9)
        except Exception:
            continue
    try:
        ts = float(time_str)
        if ts > 1e12:
            ts = ts / 1000
        dt = datetime.utcfromtimestamp(ts)
        if mode == 'KST':
            return dt
        else:
            return dt + timedelta(hours=9)
    except Exception:
        pass
    raise ValueError(f"Unknown datetime format: {time_str}")

def load_all_positions(target_files, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    포지션 데이터를 로드하고 날짜 범위로 필터링
    
    Args:
        target_files: 트레이더별 CSV 파일 경로
        start_date: 이 날짜 이후의 데이터만 로드 (None이면 제한 없음)
        end_date: 이 날짜 이전의 데이터만 로드 (None이면 제한 없음)
    
    Returns:
        filtered_positions: 필터링된 포지션 리스트
    """
    all_positions = []
    filtered_count = 0
    total_count = 0
    
    # 기본 시작일: 2024년 9월 1일 (UTC)
    default_start_date = datetime(2024, 9, 1)
    if start_date is None:
        start_date = default_start_date
    
    for trader, abs_path in target_files.items():
        df = pd.read_csv(abs_path)
        open_col, close_col, mode = detect_time_columns(df)
        
        for _, row in df.iterrows():
            total_count += 1
            try:
                open_t = parse_and_convert(str(row[open_col]), mode)
                close_t = parse_and_convert(str(row[close_col]), mode)
                
                # 날짜 필터링 적용
                if start_date and open_t < start_date:
                    filtered_count += 1
                    continue
                if end_date and open_t > end_date:
                    filtered_count += 1
                    continue
                
                all_positions.append({
                    'trader': trader,
                    'open': open_t,
                    'close': close_t,
                })
                
            except Exception as e:
                print(f"[WARN] {trader} row skipped ({e})")
    
    if filtered_count > 0:
        date_range_msg = f"시작일: {start_date.strftime('%Y-%m-%d')}"
        if end_date:
            date_range_msg += f", 종료일: {end_date.strftime('%Y-%m-%d')}"
        print(f"[INFO] 날짜 필터링으로 {filtered_count}/{total_count} 포지션이 제외됨 ({date_range_msg})")

        
    return all_positions
