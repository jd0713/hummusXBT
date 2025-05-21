import os

# 트레이더별 파일 경로와 이름 매핑
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
target_files = {
    'Chatosil': os.path.join(ROOT_DIR, '../analysis_results/Chatosil/overall/analyzed_data.csv'),
    'RebelOfBabylon': os.path.join(ROOT_DIR, '../analysis_results/RebelOfBabylon/overall/analyzed_data.csv'),
    'Ohtanishohei': os.path.join(ROOT_DIR, '../analysis_results/Ohtanishohei/overall/analyzed_data.csv'),
    'Trader0x9e8': os.path.join(ROOT_DIR, '../analysis_results/Trader0x9e8/overall/analyzed_data.csv'),
    'btcinsider': os.path.join(ROOT_DIR, '../analysis_results/btcinsider/overall/analyzed_data.csv'),
    'majorswinger': os.path.join(ROOT_DIR, '../analysis_results/majorswinger/overall/analyzed_data.csv')
}

trader_weight = {
    'RebelOfBabylon': 0.32,
    'Chatosil': 0.32,
    'Ohtanishohei': 0.16,
    'Trader0x9e8': 0.16,
    'btcinsider': 0.32,
    'majorswinger': 0.32
}

# 초기 자산 설정
initial_balance = {
    'Trader0x9e8': 3000000,
    'Ohtanishohei': 2000000,
    'Chatosil': 3000000,
    'RebelOfBabylon': 1750000,
    'btcinsider': 2400000,
    'majorswinger': 2000000
}

# 트레이더별 포지션 필터/정책 설정 (롱/숏 제한, 특정 베이스 심볼 제외 등)
TRADER_FILTERS = {
    "majorswinger": {  # 대소문자 주의: 'majorSwinger'가 아닌 'majorswinger'로 수정
        "position_mode": "long_only",  # 롱만 카피
        "excluded_bases": ["SOL", "ARB"],
    },
    "Chatosil": {
        "position_mode": "short_only",  # 숏만 카피
        "excluded_bases": ["BTC", "ETH", "XRP", "BNB"],
    },
    "Ohtanishohei": {
        "position_mode": "long_only",  # 롱만 카피
    }
    # 필요 없는 트레이더는 생략 가능
}

# 모델 포트폴리오 초기 자산
model_initial_capital = 100000
