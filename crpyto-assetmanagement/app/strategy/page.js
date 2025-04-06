'use client';

import Navigation from "../components/Navigation";
import { FaChartLine, FaShieldAlt, FaExchangeAlt, FaChartBar, FaBalanceScale } from "react-icons/fa";
import { motion } from "framer-motion";

export default function StrategyPage() {
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Navigation />
      
      {/* 헤더 섹션 */}
      <header className="bg-gradient-to-r from-blue-900 to-black text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">암호화폐 자산 관리 <span className="text-yellow-400">전략</span></h1>
          <p className="text-xl max-w-3xl mx-auto">
            시장 사이클에 따른 효과적인 자산 관리 전략으로 위험을 최소화하고 수익을 극대화하세요
          </p>
        </div>
      </header>

      {/* 전략 개요 섹션 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row gap-10">
            <div className="md:w-1/2">
              <h2 className="text-3xl font-bold mb-6">전략 <span className="text-yellow-500">개요</span></h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                암호화폐 시장에서의 성공적인 자산 관리는 시장 사이클을 이해하고 각 단계에 맞는 전략을 구사하는 것에서 시작합니다. 
                가장 중요한 것은 '공격 모드'와 '수비 모드'를 적절히 전환하는 능력입니다.
              </p>
              <p className="text-gray-600 dark:text-gray-300">
                대부분의 투자자들은 상승장에서 공격적인 전략을 구사하는 것은 쉽게 할 수 있지만, 
                <span className="font-bold text-yellow-600 dark:text-yellow-400"> 적절한 시점에 수비 모드로 전환하는 것</span>에 
                실패합니다. 이것이 많은 투자자들이 암호화폐 시장에서 큰 손실을 입는 주요 원인입니다.
              </p>
            </div>
            <div className="md:w-1/2 bg-gray-100 dark:bg-gray-700 rounded-lg p-8 flex items-center justify-center">
              <div className="text-center">
                <div className="flex justify-center items-center mb-6">
                  <div className="bg-red-500 text-white p-4 rounded-full">
                    <FaChartLine size={32} />
                  </div>
                  <div className="mx-4 text-4xl font-bold">vs</div>
                  <div className="bg-blue-500 text-white p-4 rounded-full">
                    <FaShieldAlt size={32} />
                  </div>
                </div>
                <h3 className="text-2xl font-bold mb-2">공격 vs 수비</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  시장 상황에 따른 전략 전환이 성공의 핵심
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 공격 모드 전략 */}
      <section className="py-16 bg-gray-100 dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            <span className="text-red-500">공격 모드</span> 전략
          </h2>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaChartLine className="text-red-500 mr-3" />
              언제 사용하는가?
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>명확한 상승장 국면 (불장)</li>
                <li>주요 지표가 강한 상승 신호를 보일 때</li>
                <li>제도권의 긍정적 수용과 함께 대중 관심이 증가할 때</li>
                <li>비트코인 할빙 이후 6-12개월 기간</li>
              </ul>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaChartBar className="text-red-500 mr-3" />
              주요 전략
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>
                  <span className="font-bold">알트코인 레버리지 롱:</span> 
                  <p className="mt-1">상승 모멘텀이 강한 중소형 알트코인에 레버리지를 활용한 롱 포지션</p>
                </li>
                <li>
                  <span className="font-bold">비트코인 네이키드 롱:</span> 
                  <p className="mt-1">비트코인 직접 매수를 통한 시장 상승 수익 추구</p>
                </li>
                <li>
                  <span className="font-bold">신규 프로젝트 발굴:</span> 
                  <p className="mt-1">성장 가능성이 높은 초기 프로젝트 발굴 및 투자</p>
                </li>
              </ul>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaBalanceScale className="text-red-500 mr-3" />
              리스크 관리
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>
                  <span className="font-bold">단계적 이익 실현:</span> 
                  <p className="mt-1">목표 가격에 도달할 때마다 일부 포지션 청산</p>
                </li>
                <li>
                  <span className="font-bold">스톱로스 설정:</span> 
                  <p className="mt-1">모든 포지션에 명확한 손절 기준 설정</p>
                </li>
                <li>
                  <span className="font-bold">포트폴리오 다각화:</span> 
                  <p className="mt-1">다양한 자산 클래스와 코인에 분산 투자</p>
                </li>
                <li>
                  <span className="font-bold">시장 모니터링:</span> 
                  <p className="mt-1">상승 사이클 종료 신호를 지속적으로 모니터링</p>
                </li>
              </ul>
            </div>
          </motion.div>
        </div>
      </section>

      {/* 수비 모드 전략 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            <span className="text-blue-500">수비 모드</span> 전략
          </h2>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-100 dark:bg-gray-700 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaShieldAlt className="text-blue-500 mr-3" />
              언제 사용하는가?
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>불확실한 시장 상황 (대부분의 시간)</li>
                <li>하락장 또는 횡보장 국면</li>
                <li>주요 지표가 약세 신호를 보일 때</li>
                <li>규제 불확실성이 높을 때</li>
                <li>현재와 같은 시장 상황</li>
              </ul>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-gray-100 dark:bg-gray-700 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaChartBar className="text-blue-500 mr-3" />
              주요 전략
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>
                  <span className="font-bold">카피트레이딩:</span> 
                  <p className="mt-1">검증된 탑 트레이더의 전략을 복제하여 안정적인 수익 추구</p>
                </li>
                <li>
                  <span className="font-bold">차익거래:</span> 
                  <p className="mt-1">거래소 간 가격 차이를 이용한 무위험 수익 추구</p>
                </li>
                <li>
                  <span className="font-bold">이벤트 드리븐 전략:</span> 
                  <p className="mt-1">주요 이벤트(상장, 업데이트 등)를 활용한 단기 트레이딩</p>
                </li>
                <li>
                  <span className="font-bold">스테이블코인 활용:</span> 
                  <p className="mt-1">자산의 상당 부분을 스테이블코인으로 보유하며 수익 기회 대기</p>
                </li>
              </ul>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="bg-gray-100 dark:bg-gray-700 rounded-xl p-8 shadow-lg"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaExchangeAlt className="text-blue-500 mr-3" />
              탑 트레이더 활용법
            </h3>
            <div className="pl-10">
              <ul className="list-disc space-y-3 text-gray-600 dark:text-gray-300">
                <li>
                  <span className="font-bold">트레이더 선정 기준:</span> 
                  <p className="mt-1">최소 1년 이상의 검증된 수익률, 안정적인 리스크 관리</p>
                </li>
                <li>
                  <span className="font-bold">플랫폼 다각화:</span> 
                  <p className="mt-1">Binance, Bybit, Hyperliquid, GMX 등 다양한 플랫폼의 트레이더 활용</p>
                </li>
                <li>
                  <span className="font-bold">자본 배분:</span> 
                  <p className="mt-1">여러 트레이더에게 자본을 분산하여 리스크 최소화</p>
                </li>
                <li>
                  <span className="font-bold">정기적 성과 평가:</span> 
                  <p className="mt-1">트레이더의 성과를 정기적으로 평가하고 필요시 재배분</p>
                </li>
              </ul>
            </div>
          </motion.div>
        </div>
      </section>

      {/* 전환점 파악 섹션 */}
      <section className="py-16 bg-gray-100 dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            <span className="text-green-500">전환점</span> 파악하기
          </h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
            <p className="text-gray-600 dark:text-gray-300 mb-8 text-center text-lg">
              공격에서 수비로, 수비에서 공격으로 전환하는 시점을 파악하는 것이 성공적인 자산 관리의 핵심입니다.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-red-500">공격 → 수비 전환 신호</h3>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>비트코인 도미넌스 급격한 하락</li>
                  <li>과도한 시장 열기와 FOMO 현상</li>
                  <li>비이성적인 가격 상승 (작은 시가총액 코인의 폭등)</li>
                  <li>거래량 감소와 함께 가격 상승</li>
                  <li>주요 기술적 지표의 과매수 신호</li>
                  <li>주요 저항선에서의 반복적인 거부</li>
                </ul>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-blue-500">수비 → 공격 전환 신호</h3>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>장기 이동평균선 위로의 돌파</li>
                  <li>비트코인 도미넌스 안정화</li>
                  <li>거래량 증가와 함께 가격 상승</li>
                  <li>제도권의 긍정적 규제 발표</li>
                  <li>기관 투자자들의 참여 증가</li>
                  <li>비트코인 할빙 이후 6개월 경과</li>
                </ul>
              </div>
            </div>
            
            <div className="mt-10 p-6 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
              <h3 className="text-xl font-bold mb-4 text-yellow-600 dark:text-yellow-400">핵심 원칙</h3>
              <p className="text-gray-600 dark:text-gray-300">
                <span className="font-bold">의심스러울 때는 항상 수비 모드를 유지하세요.</span> 공격 모드로 전환하는 것은 
                여러 지표가 명확한 신호를 보일 때만 해야 합니다. 수비 모드에서는 자본을 보존하면서도 
                안정적인 수익을 낼 수 있는 전략을 구사하는 것이 중요합니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA 섹션 */}
      <section className="py-16 bg-gradient-to-r from-blue-900 to-black text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-6">지금은 <span className="text-yellow-400">수비 모드</span>가 필요한 시기</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            현재 시장 상황에서는 수비 모드를 유지하며 검증된 트레이더들의 전략을 활용하는 것이 최선의 접근법입니다.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="/case-study" className="bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
              사례 연구 보기
            </a>
            <a href="/traders" className="bg-transparent border-2 border-white hover:bg-white hover:text-black px-8 py-3 rounded-full font-bold transition-all">
              탑 트레이더 분석
            </a>
          </div>
        </div>
      </section>

      {/* 푸터 */}
      <footer className="bg-gray-900 text-white py-10">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <div className="flex items-center space-x-2 text-xl font-bold">
                <FaChartLine className="text-yellow-500 text-2xl" />
                <span>암호화폐 자산 관리</span>
              </div>
              <p className="text-gray-400 mt-2">효과적인 자산 관리로 암호화폐 시장에서 성공하기</p>
            </div>
            <div className="text-gray-400 text-sm">
              © 2025 암호화폐 자산 관리 전략. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
