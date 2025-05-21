import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import argparse

def analyze_position_time_ratio(data_file):
    """포지션 데이터를 바탕으로 포지션 유지 시간 비율 분석 (포지션 겹침 고려)"""
    # 데이터 로드
    data_df = pd.read_csv(data_file)
    
    # 날짜 형식 변환 - UTC 시간대 사용
    data_df['open_time'] = pd.to_datetime(data_df['Open_Time_UTC'])
    data_df['close_time'] = pd.to_datetime(data_df['Close_Time_UTC'])
    
    # 각 포지션의 지속 시간 계산 (초 단위)
    data_df['duration_seconds'] = (data_df['close_time'] - data_df['open_time']).dt.total_seconds()
    
    # 전체 운영 기간 계산
    first_position_start = data_df['open_time'].min()
    last_position_end = data_df['close_time'].max()
    total_operation_seconds = (last_position_end - first_position_start).total_seconds()
    
    # 총 포지션 유지 시간 계산 (포지션 겹침 고려)
    # 전체 기간을 1시간 단위로 나누고 각 시간대별로 포지션이 있는지 확인
    time_range = pd.date_range(start=first_position_start, end=last_position_end, freq='1H')
    position_coverage = np.zeros(len(time_range))
    
    # 각 시간대에 대해 포지션이 존재하는지 확인
    for _, position in data_df.iterrows():
        # 각 포지션의 시작 및 종료 시간이 time_range의 어느 인덱스에 해당하는지 확인
        start_idx = np.searchsorted(time_range, position['open_time'])
        end_idx = np.searchsorted(time_range, position['close_time'])
        
        # 해당 시간대에 포지션이 있음을 표시 (1로 설정)
        position_coverage[start_idx:end_idx] = 1
    
    # 포지션이 존재하는 총 시간 계산 (시간 단위)
    total_position_hours = position_coverage.sum()
    total_position_seconds = total_position_hours * 3600  # 시간을 초로 변환
    
    # 포지션 유지 시간 비율 계산
    position_time_ratio = (total_position_seconds / total_operation_seconds) * 100
    
    # 월별 포지션 유지 시간 분석
    data_df['year_month'] = data_df['open_time'].dt.strftime('%Y-%m')
    
    # 월별 분석을 위한 데이터프레임 생성
    monthly_data = []
    
    # 운영 기간의 모든 월 리스트 생성
    current_date = first_position_start.replace(day=1)
    end_date = last_position_end.replace(day=1)
    all_months = []
    
    while current_date <= end_date:
        all_months.append(current_date.strftime('%Y-%m'))
        # 다음 달로 이동
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    # 각 월별 포지션 유지 시간 계산 (포지션 겹침 고려)
    for year_month in all_months:
        year, month = map(int, year_month.split('-'))
        
        # 해당 월의 시작과 끝
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)
        
        # 해당 월의 총 시간 (초)
        month_total_seconds = (month_end - month_start).total_seconds()
        
        # 해당 월의 시간대 범위 생성 (1시간 간격)
        month_time_range = pd.date_range(start=month_start, end=month_end, freq='1H')
        month_position_coverage = np.zeros(len(month_time_range))
        
        # 각 포지션에 대해 해당 월에 겹치는 시간 계산
        for _, position in data_df.iterrows():
            # 해당 월과 포지션의 겹치는 부분 계산
            overlap_start = max(position['open_time'], month_start)
            overlap_end = min(position['close_time'], month_end)
            
            if overlap_end > overlap_start:
                # 겹치는 부분이 month_time_range의 어느 인덱스에 해당하는지 확인
                start_idx = max(0, np.searchsorted(month_time_range, overlap_start) - 1)
                end_idx = min(len(month_time_range), np.searchsorted(month_time_range, overlap_end))
                
                # 해당 시간대에 포지션이 있음을 표시 (1로 설정)
                month_position_coverage[start_idx:end_idx] = 1
        
        # 해당 월에 포지션이 존재하는 총 시간 계산 (시간 단위)
        month_position_hours = month_position_coverage.sum()
        month_position_seconds = month_position_hours * 3600  # 시간을 초로 변환
        
        # 해당 월의 포지션 유지 시간 비율
        month_position_ratio = (month_position_seconds / month_total_seconds) * 100
        
        monthly_data.append({
            'year_month': year_month,
            'month_total_seconds': month_total_seconds,
            'month_position_seconds': month_position_seconds,
            'month_position_ratio': month_position_ratio
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    return {
        'first_position_start': first_position_start,
        'last_position_end': last_position_end,
        'total_operation_seconds': total_operation_seconds,
        'total_position_seconds': total_position_seconds,
        'position_time_ratio': position_time_ratio,
        'data_df': data_df,
        'monthly_df': monthly_df
    }

def plot_position_time_ratio(analysis_result, output_file=None):
    """포지션 유지 시간 비율 그래프 생성"""
    monthly_df = analysis_result['monthly_df']
    
    plt.figure(figsize=(14, 10))
    
    # 그래프 1: 월별 포지션 유지 시간 비율
    ax1 = plt.subplot(2, 1, 1)
    bars = ax1.bar(monthly_df['year_month'], monthly_df['month_position_ratio'], color='skyblue', alpha=0.7)
    
    # 평균 라인 추가
    avg_ratio = monthly_df['month_position_ratio'].mean()
    ax1.axhline(y=avg_ratio, color='red', linestyle='--', 
                label=f'Avg Ratio: {avg_ratio:.2f}%')
    
    # 그래프 스타일 설정
    ax1.set_title('Monthly Position Time Ratio', fontsize=16)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Position Time Ratio (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # x축 레이블 회전
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 막대 위에 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.legend()
    
    # 그래프 2: 포지션 유지 시간 vs 비유지 시간 파이 차트
    ax2 = plt.subplot(2, 1, 2)
    
    position_time = analysis_result['total_position_seconds']
    non_position_time = analysis_result['total_operation_seconds'] - position_time
    
    labels = ['In Position', 'Out of Position']
    sizes = [position_time, non_position_time]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # 첫 번째 조각 돌출
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.axis('equal')  # 원형 파이 차트
    
    # 그래프 제목 설정
    position_ratio = analysis_result['position_time_ratio']
    ax2.set_title(f'Position Time Ratio: {position_ratio:.2f}%', fontsize=16)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Position time ratio chart saved to: {output_file}")
        
    plt.close()

def main():
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description='바이낸스 리더보드 트레이더 포지션 유지 시간 비율 분석')
    parser.add_argument('--trader', type=str, required=True,
                        help='분석할 트레이더 이름 (예: Chatosil)')
    args = parser.parse_args()
    
    trader = args.trader
    
    # 파일 경로
    data_file = os.path.join(os.path.dirname(__file__), f'analysis_results/{trader}/overall/analyzed_data.csv')
    
    # 결과 디렉토리
    output_dir = os.path.join(os.path.dirname(__file__), f'analysis_results/{trader}/time_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===== {trader} 포지션 유지 시간 비율 분석 시작 =====\n")
    
    # 포지션 유지 시간 비율 분석
    analysis_result = analyze_position_time_ratio(data_file)
    
    # 결과 저장
    monthly_df = analysis_result['monthly_df']
    output_csv = os.path.join(output_dir, 'position_time_ratio.csv')
    monthly_df.to_csv(output_csv, index=False)
    print(f"Position time ratio data saved to: {output_csv}")
    
    # 그래프 생성
    output_png = os.path.join(output_dir, 'position_time_ratio.png')
    plot_position_time_ratio(analysis_result, output_png)
    
    # 주요 통계 출력
    first_position = analysis_result['first_position_start'].strftime('%Y-%m-%d %H:%M:%S')
    last_position = analysis_result['last_position_end'].strftime('%Y-%m-%d %H:%M:%S')
    total_operation_days = analysis_result['total_operation_seconds'] / (60 * 60 * 24)
    total_position_days = analysis_result['total_position_seconds'] / (60 * 60 * 24)
    position_ratio = analysis_result['position_time_ratio']
    
    print(f"\n===== {trader.upper()} Position Time Ratio Summary =====")
    print(f"First Position Start: {first_position}")
    print(f"Last Position End: {last_position}")
    print(f"Total Operation Period: {total_operation_days:.2f} days")
    print(f"Total Position Time: {total_position_days:.2f} days")
    print(f"Position Time Ratio: {position_ratio:.2f}%")
    
    # 월별 통계
    print("\nMonthly Position Time Ratio:")
    monthly_stats = monthly_df.sort_values('month_position_ratio', ascending=False)
    for _, row in monthly_stats.head(3).iterrows():
        month_days = row['month_total_seconds'] / (60 * 60 * 24)
        position_days = row['month_position_seconds'] / (60 * 60 * 24)
        print(f"  {row['year_month']}: {row['month_position_ratio']:.2f}% ({position_days:.1f} days out of {month_days:.1f} days)")
    
    print("\nMonths with Lowest Position Time Ratio:")
    for _, row in monthly_stats.tail(3).iterrows():
        month_days = row['month_total_seconds'] / (60 * 60 * 24)
        position_days = row['month_position_seconds'] / (60 * 60 * 24)
        print(f"  {row['year_month']}: {row['month_position_ratio']:.2f}% ({position_days:.1f} days out of {month_days:.1f} days)")
    
    print("=====================================")

if __name__ == "__main__":
    main()
    
# 사용 예시:
# python analyze_position_time_ratio.py --trader trader1  # trader1 분석
# python analyze_position_time_ratio.py --trader trader2  # trader2 분석 (기본값)
