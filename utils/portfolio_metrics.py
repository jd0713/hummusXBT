#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime, timedelta

def calculate_mdd(df):
    """
    Maximum Drawdown(최대 낙폭) 계산 함수
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
    
    Returns:
        tuple: MDD 값, MDD 발생 시점, 피크 시점, 피크 값, MDD 발생 시 자본
    """
    # 누적 최대값 계산
    df['cumulative_max'] = df['end_capital'].cummax()
    # 드로다운 계산
    df['drawdown'] = (df['end_capital'] - df['cumulative_max']) / df['cumulative_max'] * 100
    # MDD 계산 (최저 드로다운)
    mdd = df['drawdown'].min()
    # MDD 발생 시점 찾기
    mdd_idx = df['drawdown'].idxmin()
    mdd_time = df.loc[mdd_idx, 'time']
    # 최대값 도달 시점 (MDD 발생 직전의 피크)
    peak_value = df.loc[mdd_idx, 'cumulative_max']
    peak_idx = df[df['end_capital'] == peak_value].index[0]
    peak_time = df.loc[peak_idx, 'time']
    
    return mdd, mdd_time, peak_time, peak_value, df.loc[mdd_idx, 'end_capital']

def calculate_monthly_mdd(df):
    """
    월별 MDD 계산 함수
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
    
    Returns:
        DataFrame: 월별 MDD 데이터프레임
    """
    df['year_month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly_mdd = df.groupby('year_month').apply(lambda x: calculate_mdd(x)[0]).reset_index()
    monthly_mdd.columns = ['year_month', 'mdd']
    return monthly_mdd

def calculate_annualized_return(df):
    """
    연환산 수익률 계산 함수
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
    
    Returns:
        float: 연환산 수익률(%)
    """
    # 처음 기록과 마지막 기록의 날짜 차이 계산 (일 단위)
    start_date = pd.to_datetime(df['time'].iloc[0])
    end_date = pd.to_datetime(df['time'].iloc[-1])
    days = (end_date - start_date).days
    
    # 리턴 계산
    total_return = df['end_capital'].iloc[-1] / df['start_capital'].iloc[0]
    
    # 연환산 수익률 계산
    if days > 0:
        annualized_return = (total_return ** (365 / days)) - 1
        return annualized_return * 100  # 퍼센트로 변환
    else:
        return 0

def calculate_daily_returns(df):
    """
    일별 수익률 계산 함수
    
    Args:
        df (DataFrame): 포트폴리오 자본 데이터프레임
    
    Returns:
        DataFrame: 일별 수익률 데이터프레임
    """
    # 시간 정렬
    df = df.sort_values('time')
    
    # 시간 변환
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    # 일별 데이터로 그룹화
    daily_data = df.groupby('date').agg({'end_capital': 'last'}).reset_index()
    
    # 일별 수익률 계산
    daily_data['prev_capital'] = daily_data['end_capital'].shift(1)
    daily_data = daily_data.dropna()  # 첫번째 행(이전 값이 없는) 제거
    
    if len(daily_data) > 0:
        daily_data['daily_return'] = (daily_data['end_capital'] / daily_data['prev_capital']) - 1
        return daily_data
    else:
        return pd.DataFrame(columns=['date', 'end_capital', 'prev_capital', 'daily_return'])

def calculate_sharpe_ratio(daily_returns_df, risk_free_rate=0.02):
    """
    샤프 비율(Sharpe Ratio) 계산 함수
    
    Args:
        daily_returns_df (DataFrame): 일별 수익률 데이터프레임
        risk_free_rate (float): 무위험 이자율(연 %)
    
    Returns:
        float: 샤프 비율
    """
    # 일별 수익률 없으면 계산 불가
    if len(daily_returns_df) <= 1:
        return 0
    
    # 일별 수익률 평균
    avg_daily_return = daily_returns_df['daily_return'].mean()
    
    # 일별 수익률 표준편차
    std_daily_return = daily_returns_df['daily_return'].std()
    
    # 일별 무위험 이자율
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/365) - 1
    
    # 샤프 비율 계산
    if std_daily_return > 0:
        sharpe_ratio = (avg_daily_return - daily_risk_free_rate) / std_daily_return
        # 연단위 샤프 비율로 변환
        annual_sharpe = sharpe_ratio * sqrt(252)  # 거래일 252일 기준
        return annual_sharpe
    else:
        return 0

def calculate_calmar_ratio(annualized_return, max_drawdown):
    """
    칼마 비율(Calmar Ratio) 계산 함수
    
    Args:
        annualized_return (float): 연환산 수익률(%)
        max_drawdown (float): 최대 낙폭(%)
    
    Returns:
        float: 칼마 비율
    """
    # 최대 낙폭이 없거나 0이면 계산 불가
    if max_drawdown >= 0:
        return 0
    
    # 칼마 비율 계산 (연환산 수익률 / 절대값(MDD))
    calmar_ratio = annualized_return / abs(max_drawdown)  # 절대값 사용
    return calmar_ratio
