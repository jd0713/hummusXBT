#!/usr/bin/env python3
"""
트레이더 설정 파일
여러 트레이더의 설정을 관리합니다.
"""

# 트레이더 정보 딕셔너리
TRADERS = {
    "hummusXBT": {
        "name": "hummusXBT",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/D9E14F24A64472D262BF72FC7F817CBE",
        "initial_capital_period1": 100000,  # 첫 번째 기간 원금: 100K USD
        "initial_capital_period2": 2000000,  # 두 번째 기간 원금: 2M USD
        "period_split_date": "2024-06-15",  # 기간 구분 기준 날짜 (2024년 중반)
        "description": "hummusXBT 트레이더 분석"
    },
    "TRADERT22": {
        "name": "TRADERT22",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/13A05CEEA6B2B444C1AB8973255BAF9C",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "TRADERT22 트레이더 분석"
    },
    "CnTraderT": {
        "name": "CnTraderT",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/CCF3E0CB0AAD54D9D6B4CEC5E3E741D2",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "CnTraderT 트레이더 분석"
    },
    "H_VA": {
        "name": "H_VA",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/2FF76A4EDC52D8F2265E8173B280B28C",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "H_VA 트레이더 분석"
    },
    "Metabeem": {
        "name": "Metabeem",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/49A7275656A7ABF56830126ACC619FEB",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Metabeem 트레이더 분석"
    },
    "AutoRebalance": {
        "name": "AutoRebalance",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/BFE5C3E7EF7B3629438D907CD3B21D57",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "AutoRebalance 트레이더 분석"
    },
    "Chatosil": {
        "name": "Chatosil",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/456E1FCF2155C31D72C5DC61DCD2C64C",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Chatosil 트레이더 분석"
    }
}

# 새 트레이더 추가 함수
def add_trader(trader_id, name, url, initial_capital_period1=100000, initial_capital_period2=2000000, 
               period_split_date="2024-06-15", description=None):
    """새로운 트레이더를 설정에 추가합니다."""
    if description is None:
        description = f"{name} 트레이더 분석"
        
    TRADERS[trader_id] = {
        "name": name,
        "url": url,
        "initial_capital_period1": initial_capital_period1,
        "initial_capital_period2": initial_capital_period2,
        "period_split_date": period_split_date,
        "description": description
    }
    
    return TRADERS[trader_id]

# 트레이더 정보 조회 함수
def get_trader(trader_id):
    """트레이더 ID로 트레이더 정보를 조회합니다."""
    if trader_id in TRADERS:
        return TRADERS[trader_id]
    else:
        raise ValueError(f"트레이더 ID '{trader_id}'를 찾을 수 없습니다.")

# 모든 트레이더 ID 조회 함수
def get_all_trader_ids():
    """모든 트레이더 ID 목록을 반환합니다."""
    return list(TRADERS.keys())

# 모든 트레이더 정보 조회 함수
def get_all_traders():
    """모든 트레이더 정보를 반환합니다."""
    return TRADERS
