import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

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

def load_all_positions(target_files):
    all_positions = []
    for trader, abs_path in target_files.items():
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
