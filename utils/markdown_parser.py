#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

def parse_position_markdown(file_path):
    """
    포지션 분석 마크다운 파일을 파싱하여 구간별 데이터를 추출합니다.
    
    Args:
        file_path (str): 마크다운 파일의 경로
        
    Returns:
        list: 구간별 데이터 딕셔너리 리스트
    """
    # 마크다운 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 구간 분석 결과 파싱
    periods = []
    current_period = None
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # 구간 시작 - 제목 라인 분석
        if line.startswith('## ['):
            if current_period is not None:
                periods.append(current_period)
            
            # 새로운 구간 정보 초기화
            current_period = {
                'title': line,
                'traders': {},
                'positions': [],         # 포지션 정보를 저장하는 리스트
                'weights': {},          # 트레이더별 가중치
                'balance_after': {}      # 구간 종료 후 트레이더별 자산
            }
            
            # 시작 시간과 종료 시간 추출
            try:
                time_part = line.split('] ')[1].split(' | ')[0]
                start_time_str, end_time_str = time_part.split(' ~ ')
                current_period['start_time_str'] = start_time_str
                current_period['end_time_str'] = end_time_str
                current_period['start_time'] = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                current_period['end_time'] = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(f"Error parsing period time: {e}")
            
            current_section = None
            continue
        
        # 표 헤더 식별
        if line.startswith('| 트레이더 | 심볼 | 방향 |'):
            current_section = 'positions'
            continue
        elif line.startswith('| 트레이더 | Weight(%) | 롱 오픈'):
            current_section = 'weights'
            continue
        elif line.startswith('| 트레이더 | 구간 종료 후 자산'):
            current_section = 'balance_after'
            continue
        
        # 데이터 파싱
        if current_period and current_section == 'positions' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 8:  # 거래량(USDT) 칼럼이 추가되어 최소 8개 이상의 칼럼 필요
                trader = parts[0]
                symbol = parts[1]
                direction = parts[2]
                size_str = parts[3].replace(',', '').replace('USDT', '').strip()
                volume_str = parts[4].replace(',', '').replace('USDT', '').strip()  # 거래량(USDT) 추출
                
                # 오픈 시간과 클로즈 시간 추출
                open_time_str = parts[5].strip()  # 인덱스 변경
                close_time_str = parts[6].strip()  # 인덱스 변경
                
                realized_pnl_str = parts[7].replace(',', '').replace('USDT', '').strip()  # 인덱스 변경
                realized_pnl_pct_str = parts[8].replace('%', '').strip()  # 인덱스 변경
                
                try:
                    size = float(size_str)
                    volume = float(volume_str)  # 거래량 변환
                    realized_pnl = float(realized_pnl_str)
                    realized_pnl_pct = float(realized_pnl_pct_str) / 100.0  # 백분율을 소수로 변환
                    
                    # 트레이더별 실현 수익 정보 추가
                    position = {
                        'trader': trader,
                        'symbol': symbol,
                        'direction': direction,
                        'size': size,
                        'volume': volume,  # 거래량 추가
                        'open': open_time_str,    # 오픈 시간 저장
                        'close': close_time_str,  # 클로즈 시간 저장
                        'realized_pnl': realized_pnl,
                        'realized_pnl_pct': realized_pnl_pct
                    }
                    current_period['positions'].append(position)
                    
                except Exception as e:
                    print(f"Error parsing position for {trader}: {e}")
        
        elif current_period and current_section == 'weights' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 8:
                trader = parts[0]
                weight_str = parts[1].replace('%', '').strip()
                
                try:
                    weight = float(weight_str) / 100.0  # 백분율을 소수로 변환
                    current_period['weights'][trader] = weight
                except Exception as e:
                    print(f"Error parsing weight for {trader}: {e}")
        
        elif current_period and current_section == 'balance_after' and line.startswith('|') and ' | ' in line and not line.startswith('|--'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 2:
                trader = parts[0]
                balance_str = parts[1].replace(',', '').replace('USDT', '').strip()
                
                try:
                    balance = float(balance_str)
                    current_period['balance_after'][trader] = balance
                except Exception as e:
                    print(f"Error parsing balance for {trader}: {e}")
    
    # 마지막 구간 추가
    if current_period is not None:
        periods.append(current_period)
    
    return periods
