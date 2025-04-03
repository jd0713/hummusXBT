#!/usr/bin/env python3
import subprocess
import sys
from trader_analysis_part1 import *
from trader_analysis_part2 import *
# 슬리피지 분석 모듈 임포트
from analyze_slippage_impact import analyze_slippage_impact

def main():
    """트레이더 분석 실행"""
    # CSV 파일 경로
    csv_file = "closed_positions_20250403_073826.csv"
    
    # 폴더 생성 확인
    for directory in [RESULT_DIR, PERIOD1_DIR, PERIOD2_DIR, OVERALL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"폴더 생성: {directory}")
    
    print("\n===== 트레이더 데이터 분석 시작 =====")
    
    # PnL 분석 (기간별)
    df, df_period1, df_period2 = analyze_pnl_by_periods(csv_file)
    
    # 성과 지표 계산
    metrics_period1 = calculate_performance_metrics(df_period1, INITIAL_CAPITAL_PERIOD1, '기간 1')
    metrics_period2 = calculate_performance_metrics(df_period2, INITIAL_CAPITAL_PERIOD2, '기간 2')
    
    # 성과 비교 출력
    print_performance_comparison(metrics_period1, metrics_period2)
    
    # 트레이더 실력 평가
    evaluation_result = evaluate_trader_skill(metrics_period1, metrics_period2)
    
    # 모든 시각화 생성
    print("\n===== 시각화 생성 시작 =====")
    generate_all_visualizations(df, df_period1, df_period2)
    
    # 자산 성장 그래프 생성 (plot_asset_growth.py 실행)
    print("\n===== 자산 성장 그래프 생성 =====")
    try:
        # 현재 스크립트와 같은 디렉토리에 있는 plot_asset_growth.py 실행
        python_cmd = sys.executable  # 현재 실행 중인 Python 인터프리터 경로 사용
        result = subprocess.run([python_cmd, "plot_asset_growth.py"], 
                              capture_output=True, text=True, check=True)
        # 출력 결과 표시
        print(result.stdout)
        if result.stderr:
            print(f"경고: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"오류: 자산 성장 그래프 생성 중 문제가 발생했습니다.")
        print(f"오류 메시지: {e.stderr}")
    
    # 슬리피지 영향 분석 실행
    print("\n===== 슬리피지 영향 분석 시작 =====")
    try:
        # analyze_slippage_impact 함수 직접 호출
        analyze_slippage_impact()
    except Exception as e:
        print(f"오류: 슬리피지 영향 분석 중 문제가 발생했습니다.")
        print(f"오류 메시지: {str(e)}")
    
    # 분석 요약
    print("\n===== 분석 요약 =====")
    
    # 전체 거래 금액
    total_volume = calculate_total_trading_volume(df)
    print(f"전체 거래 금액: ${total_volume:,.2f} USD")
    
    # 기간별 거래 금액
    if not df_period1.empty:
        period1_volume = calculate_total_trading_volume(df_period1)
        print(f"기간 1 거래 금액: ${period1_volume:,.2f} USD")
    
    if not df_period2.empty:
        period2_volume = calculate_total_trading_volume(df_period2)
        print(f"기간 2 거래 금액: ${period2_volume:,.2f} USD")
    
    # 트레이더 평가 출력
    print("\n===== 트레이더 평가 =====")
    print(f"종합 점수: {evaluation_result['점수']}")
    print("\n세부 평가:")
    for comment in evaluation_result["평가"]:
        print(f"- {comment}")
    print(f"\n종합 의견:\n{evaluation_result['종합 의견']}")
    
    print("\n===== 분석 완료 =====")
    print(f"모든 분석 결과는 {RESULT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
