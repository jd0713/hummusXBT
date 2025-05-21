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
    "HONG_TOP": {
        "name": "HONG_TOP",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/D4D39BC5123A4F821729778454D8F723",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "HONG_TOP 트레이더 분석"
    },
    "Chatosil": {
        "name": "Chatosil",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/456E1FCF2155C31D72C5DC61DCD2C64C",
        "initial_capital": 3000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Chatosil 트레이더 분석"
    },
    "Panic": {
        "name": "惊惧 | Panic",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/2DBD6EA4CA7BC0F13BDD4868B7893CEF",
        "initial_capital_period1": 1000000,  # 첫 번째 기간 원금: 1M USD (추정)
        "initial_capital_period2": 1000000,  # 두 번째 기간 원금: 1M USD (추정)
        "period_split_date": "2024-10-31",  # 기간 구분 기준 날짜 (2024년 10월 말)
        "use_periods": True,  # 기간 구분 사용
        "description": "惊惧 | Panic 트레이더 분석 (기간 구분 및 원금 추정)"
    },
    "RebelOfBabylon": {
        "name": "RebelOfBabylon",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/07400C16C63B685FDAB8048CF4E8AFD2",
        "initial_capital": 1750000,  
        "use_periods": False,  # 기간 구분 없음
        "description": "RebelOfBabylon 트레이더 분석"
    },
    "JeromeLoo": {
        "name": "JeromeLoo 老王",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/0D85BE48AFEAE12D78FFF98E7369B72F",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "JeromeLoo 老王 트레이더 분석"
    },
    "TreeOfAlpha2": {
        "name": "TreeOfAlpha2",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/DF74DFB6CB244F8033F1D66D5AA0B171",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "TreeOfAlpha2 트레이더 분석"
    },
    "StopWhen300M": {
        "name": "赚够三亿U就收手 | Stop When You Earn 300 Million Yuan",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/D6A33BCC2092DF57673F42E6BBCFFB2F",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "赚够三亿U就收手 | Stop When You Earn 300 Million Yuan 트레이더 분석"
    },
    "CryptoNifeCatchN": {
        "name": "CryptoNifeCatchN",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/A086AC7B587E11941378E95DD6C872C6",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "CryptoNifeCatchN 트레이더 분석"
    },
    "ONLYCANDLE": {
        "name": "ONLYCANDLE",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/616EAE3E24D9FFFA7EA8E500D102D2DC",
        "initial_capital": 1000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "ONLYCANDLE 트레이더 분석"
    },
    "cake-bnb": {
        "name": "cake-bnb",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/38744FBD26B033A15516DFFCED8510E9",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "cake-bnb 트레이더 분석"
    },
    "Ohtanishohei": {
        "name": "Ohtanishohei",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/EE2861F7029B8C98B6B000A06BC938DC",
        "initial_capital": 2000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Ohtanishohei 트레이더 분석"
    },
    "Trader0x9e8": {
        "name": "Trader0x9e8",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0x9e8b1e51c642f4C8b87c6BA11c53D516a218Afc4",
        "initial_capital": 3000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Trader0x9e8 트레이더 분석"
    },
    "HyperdashVault1": {
        "name": "HyperdashVault1",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0x4078582C42fdb547B1397FaBB5D5A4beAB81bE9E",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "HyperdashVault1 트레이더 분석"
    },
    "Cyborg0578": {
        "name": "Cyborg0578",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/5B0834121B27FD470BE7EE2B954B6970",
        "initial_capital": 1000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Cyborg0578 트레이더 분석"
    },
    "haleFlower": {
        "name": "hale花花",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/D31DA0CFC550B81981E6A582E3E4B2F5",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "hale花花 트레이더 분석"
    },
    "musthaveluck": {
        "name": "musthaveluck",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/61FB1D1ADC31E34B36948A75A338CCA0",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "musthaveluck 트레이더 분석"
    },
    "Sky": {
        "name": "Sky",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0xDd0c5dE50D72E5eaa96816e920e41CE89C4B8888",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Sky 트레이더 분석"
    },
    "Elsewhere": {
        "name": "Elsewhere",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0x8fC7c0442e582bca195978C5a4FDeC2e7C5bB0f7",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "Elsewhere 트레이더 분석"
    },
    "HummusXBT_Bybit": {
        "name": "HummusXBT_Bybit",
        "url": "https://portal.mirrorly.xyz/leaderboard/Bybit/ecd2cb60-f909-4806-99f8-1e2ff9ac190c",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "HummusXBT Bybit 트레이더 분석"
    },
    "BLACKRUN": {
        "name": "BLACKRUN",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/D3DEEE53F5F824749BFCD3A50ACFB0E2",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "BLACKRUN 트레이더 분석"
    },
    "1234215": {
        "name": "1234215",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/76E597D3511540D93F9C12517227A778",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "1234215 트레이더 분석"
    },
    "taedy87": {
        "name": "taedy87",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/6C553168FC9AA9C366B8259700C4740F",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "taedy87 트레이더 분석"
    },
    "ILB-2": {
        "name": "ILB-2",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/9AD56B8C3417DDCB111839DF7743DA94",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,     # 기간 구분 없음
        "description": "ILB-2 트레이더 분석"
    },
    "u_guys_r_rly_dum": {
        "name": "u_guys_r_rly_dum",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0x987163B6b482C30c2F5f3aa2760109668Eb0091d",
        "initial_capital": 1000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "u_guys_r_rly_dum 트레이더 분석"
    },
    "a2cc5": {
        "name": "a2cc5",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/007E14EF366059DC29AB330CD93A713B",
        "initial_capital": 1000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "a2cc5 트레이더 분석"
    },
    "OnchainRetard": {
        "name": "OnchainRetard",
        "url": "https://portal.mirrorly.xyz/leaderboard/Binance/024E5503105496A7B68D72AD29F22826",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "OnchainRetard 트레이더 분석"
    },    
    "AccDownOnlyPlsGibPts": {
        "name": "AccDownOnlyPlsGibPts",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0xC4180D58e4980ae6b39C133aB3A9389ae62E8706",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "AccDownOnlyPlsGibPts 트레이더 분석"
    },
    "bigdestiny": {
        "name": "bigdestiny",
        "url": "https://portal.mirrorly.xyz/leaderboard/Hyperliquid/0x183D0567c33e7591c22540E45D2F74730b42a0ca",
        "initial_capital": None,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "bigdestiny 트레이더 분석"
    },
    "majorswinger": {
        "name": "majorswinger",
        "url": "",
        "initial_capital": 2000000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "majorswinger 트레이더 분석"
    },
    "btcinsider": {
        "name": "btcinsider",
        "url": "",
        "initial_capital": 2400000,  # 원금 정보 없음
        "use_periods": False,  # 기간 구분 없음
        "description": "btcinsider 트레이더 분석"
    },
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
