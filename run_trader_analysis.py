#!/usr/bin/env python3
import subprocess
import sys
import argparse
import os
from trader_analysis_part1 import *
from trader_analysis_part2 import *
# 슬리피지 분석 모듈 임포트
from analyze_slippage_impact import analyze_slippage_impact
# 트레이더 설정 모듈 임포트
from trader_config import get_trader, get_all_trader_ids

def main(trader_id=None):
    """트레이더 분석 실행"""
    # 트레이더 선택 또는 확인
    if trader_id is None:
        # 사용 가능한 트레이더 목록 출력
        trader_ids = get_all_trader_ids()
        print("\n=== 사용 가능한 트레이더 목록 ===")
        for i, tid in enumerate(trader_ids, 1):
            trader = get_trader(tid)
            print(f"{i}. {trader['name']} - {trader['description']}")
        
        # 사용자에게 트레이더 선택 요청
        selection = input("\n분석할 트레이더 번호를 선택하세요: ")
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(trader_ids):
                trader_id = trader_ids[idx]
            else:
                print("잘못된 선택입니다. 기본 트레이더(hummusXBT)를 사용합니다.")
                trader_id = "hummusXBT"
        except ValueError:
            print("잘못된 입력입니다. 기본 트레이더(hummusXBT)를 사용합니다.")
            trader_id = "hummusXBT"
    
    # 트레이더 정보 가져오기
    trader = get_trader(trader_id)
    trader_name = trader['name']
    
    # 최신 CSV 파일 찾기
    trader_data_dir = f"trader_data/{trader_id}"
    if not os.path.exists(trader_data_dir):
        print(f"오류: {trader_id} 트레이더의 데이터 디렉토리를 찾을 수 없습니다.")
        return
    
    # 디렉토리에서 CSV 파일 찾기
    csv_files = [f for f in os.listdir(trader_data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"오류: {trader_id} 트레이더의 CSV 파일을 찾을 수 없습니다.")
        return
    
    # 가장 최근 파일 선택 (파일명에 타임스탬프가 포함되어 있다고 가정)
    csv_files.sort(reverse=True)
    csv_file = os.path.join(trader_data_dir, csv_files[0])
    print(f"분석할 파일: {csv_file}")
    
    # 트레이더 설정에 따라 분석 방식 결정
    use_periods = trader.get('use_periods', True)  # 기본값은 기간 구분 사용
    
    # 결과 폴더 설정
    RESULT_DIR = f"analysis_results/{trader_id}"
    OVERALL_DIR = os.path.join(RESULT_DIR, "overall")
    
    # 기본 폴더 생성
    for directory in [RESULT_DIR, OVERALL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"폴더 생성: {directory}")
    
    # 기간 구분이 있는 트레이더인 경우에만 period1, period2 디렉토리 생성
    if use_periods:
        PERIOD1_DIR = os.path.join(RESULT_DIR, "period1")
        PERIOD2_DIR = os.path.join(RESULT_DIR, "period2")
        for directory in [PERIOD1_DIR, PERIOD2_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"폴더 생성: {directory}")
    
    print(f"\n===== {trader_name} 트레이더 데이터 분석 시작 =====")
    
    # 분석 시작
    
    if use_periods:
        # 기간 구분이 있는 경우 (hummusXBT 같은 트레이더)
        initial_capital_period1 = trader.get('initial_capital_period1', INITIAL_CAPITAL_PERIOD1)
        initial_capital_period2 = trader.get('initial_capital_period2', INITIAL_CAPITAL_PERIOD2)
        period_split_date_str = trader.get('period_split_date', '2024-06-15')
        
        # 기간 구분 날짜 설정
        period_split_date = datetime.strptime(period_split_date_str, '%Y-%m-%d')
        
        # PnL 분석 (기간별)
        df, df_period1, df_period2 = analyze_pnl_by_periods(csv_file, period_split_date, trader_id)
        
        # 성과 지표 계산
        metrics_period1 = calculate_performance_metrics(df_period1, initial_capital_period1, '기간 1')
        metrics_period2 = calculate_performance_metrics(df_period2, initial_capital_period2, '기간 2')
        
        # 성과 비교 출력
        print_performance_comparison(metrics_period1, metrics_period2, trader_id)
        
        # 트레이더 실력 평가
        evaluation_result = evaluate_trader_skill(metrics_period1, metrics_period2, trader_id)
        
        # 평가 결과 출력
        print_trader_evaluation(evaluation_result)
    else:
        # 기간 구분이 없는 경우 (TRADERT22 같은 트레이더)
        initial_capital = trader.get('initial_capital', 100000)  # 기본 원금 설정
        
        # 전체 기간 분석
        df = analyze_pnl_without_periods(csv_file, trader_id)
        df_period1 = pd.DataFrame()  # 빈 데이터프레임
        df_period2 = pd.DataFrame()  # 빈 데이터프레임
        
        # 전체 성과 지표 계산
        if initial_capital is None:
            # 원금 정보가 없는 경우 원금 없이 분석
            metrics_overall = calculate_performance_metrics_without_capital(df, '전체 기간')
            evaluation_result = {
                "점수": "N/A",
                "평가": ["원금 정보가 없어 성과 평가를 할 수 없습니다."],
                "종합 의견": "원금 정보가 없어 종합적인 평가를 제공할 수 없습니다. 거래 패턴과 수익/손실 분포만 분석할 수 있습니다."
            }
            
            # 성과 지표 출력
            print("\n===== 전체 기간 성과 지표 =====")
            print(f"총 PnL: ${metrics_overall['Total PnL']:,.2f}")
            print(f"총 거래 금액: ${metrics_overall['Total Volume']:,.2f}")
            print(f"거래 기간: {metrics_overall['Start Date'].strftime('%Y-%m-%d')} ~ {metrics_overall['End Date'].strftime('%Y-%m-%d')} ({metrics_overall['Trading Days']}일)")
            print(f"일평균 PnL: ${metrics_overall['Daily Avg PnL']:,.2f}")
            print(f"승률: {metrics_overall['Win Rate (%)']:.2f}%")
            print(f"평균 수익: ${metrics_overall['Avg Win']:,.2f}")
            print(f"평균 손실: ${metrics_overall['Avg Loss']:,.2f}")
            
            # 심볼별 성과 출력
            print("\n----- 심볼별 성과 (상위 5개) -----")
            for symbol, pnl in metrics_overall['Top Symbols'].items():
                print(f"{symbol}: ${pnl:,.2f}")
                
            print("\n----- 심볼별 성과 (하위 5개) -----")
            for symbol, pnl in metrics_overall['Bottom Symbols'].items():
                print(f"{symbol}: ${pnl:,.2f}")
                
            # 방향별 성과 출력
            print("\n----- 방향별 성과 -----")
            for direction, pnl in metrics_overall['Direction PnL'].items():
                print(f"{direction}: ${pnl:,.2f}")
        else:
            # 원금 정보가 있는 경우 원금 기준 분석
            metrics_overall = calculate_performance_metrics(df, initial_capital, '전체 기간')
            evaluation_result = {
                "점수": "N/A",
                "평가": ["기간 구분이 없어 성과 비교를 할 수 없습니다."],
                "종합 의견": "기간 구분이 없어 성과 변화를 평가할 수 없습니다. 전체 기간의 성과만 분석할 수 있습니다."
            }
            
            # 성과 지표 출력 (원금 정보 포함)
            print("\n===== 전체 기간 성과 지표 =====")
            print(f"초기 원금: ${initial_capital:,.2f}")
            print(f"최종 자본: ${metrics_overall['Final Capital']:,.2f}")
            print(f"총 수익률: {metrics_overall['Total Return']:.2f}%")
            print(f"연간 수익률: {metrics_overall['Annualized Return']:.2f}%")
            print(f"총 PnL: ${metrics_overall['Total PnL']:,.2f}")
            print(f"총 거래 금액: ${metrics_overall['Total Volume']:,.2f}")
            print(f"거래 기간: {metrics_overall['Start Date'].strftime('%Y-%m-%d')} ~ {metrics_overall['End Date'].strftime('%Y-%m-%d')} ({metrics_overall['Trading Days']}일)")
            print(f"일평균 PnL: ${metrics_overall['Daily Avg PnL']:,.2f}")
            print(f"승률: {metrics_overall['Win Rate (%)']:.2f}%")
            print(f"평균 수익: ${metrics_overall['Avg Win']:,.2f}")
            print(f"평균 손실: ${metrics_overall['Avg Loss']:,.2f}")
            
            # 심볼별 성과 출력
            print("\n----- 심볼별 성과 (상위 5개) -----")
            for symbol, pnl in metrics_overall['Top Symbols'].items():
                print(f"{symbol}: ${pnl:,.2f}")
                
            print("\n----- 심볼별 성과 (하위 5개) -----")
            for symbol, pnl in metrics_overall['Bottom Symbols'].items():
                print(f"{symbol}: ${pnl:,.2f}")
                
            # 방향별 성과 출력
            print("\n----- 방향별 성과 -----")
            for direction, pnl in metrics_overall['Direction PnL'].items():
                print(f"{direction}: ${pnl:,.2f}")
    
    # 모든 시각화 생성
    print("\n===== 시각화 생성 시작 =====")
    try:
        if use_periods:
            generate_all_visualizations(df, df_period1, df_period2, trader_id)
        else:
            # 기간 구분이 없는 트레이더를 위한 시각화
            generate_visualizations_without_periods(df, trader_id)
        print("시각화가 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"시각화 생성 중 오류 발생: {str(e)}")
    
    # 자산 성장 그래프 생성 (plot_asset_growth.py 실행)
    print("\n===== 자산 성장 그래프 생성 =====")
    try:
        # 현재 스크립트와 같은 디렉토리에 있는 plot_asset_growth.py 실행
        python_cmd = sys.executable  # 현재 실행 중인 Python 인터프리터 경로 사용
        result = subprocess.run([python_cmd, "plot_asset_growth.py", "-t", trader_id, "-o", RESULT_DIR], 
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
        analyze_slippage_impact(trader_id)
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
    print_trader_evaluation(evaluation_result)
    
    print("\n===== 분석 완료 =====")
    print(f"모든 분석 결과는 {RESULT_DIR} 폴더에 저장되었습니다.")

def print_trader_evaluation(evaluation_result):
    """트레이더 평가 결과 출력"""
    print("\n===== 트레이더 평가 =====")
    print(f"종합 점수: {evaluation_result['점수']}")
    
    print("\n세부 평가:")
    for comment in evaluation_result["평가"]:
        print(f"- {comment}")
    
    print(f"\n종합 의견:\n{evaluation_result['종합 의견']}")

if __name__ == "__main__":
    # 커맨드 라인 인자 처리
    parser = argparse.ArgumentParser(description="트레이더 분석 도구")
    parser.add_argument("-t", "--trader", help="분석할 트레이더 ID", type=str)
    parser.add_argument("-l", "--list", help="사용 가능한 트레이더 목록 출력", action="store_true")
    
    args = parser.parse_args()
    
    # 트레이더 목록 출력 옵션
    if args.list:
        trader_ids = get_all_trader_ids()
        print("\n=== 사용 가능한 트레이더 목록 ===")
        for i, tid in enumerate(trader_ids, 1):
            trader = get_trader(tid)
            print(f"{i}. {tid} - {trader['name']} - {trader['description']}")
        sys.exit(0)
    
    # 트레이더 ID가 지정되었는지 확인
    trader_id = args.trader
    
    # 분석 실행
    main(trader_id)
