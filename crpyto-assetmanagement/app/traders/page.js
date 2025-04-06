'use client';

import { useState } from 'react';
import Navigation from "../components/Navigation";
import { FaUserTie, FaChartLine, FaExchangeAlt, FaBalanceScale, FaSearch } from "react-icons/fa";
import { motion } from "framer-motion";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

// Chart.js 등록
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function TradersPage() {
  const [activeTab, setActiveTab] = useState('all');
  
  // 예시 차트 데이터
  const monthlyPerformanceData = {
    labels: ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'],
    datasets: [
      {
        label: 'hummusXBT',
        data: [5.2, 7.8, -2.3, 12.5, 8.9, 4.2, 15.6, 9.8, 6.7, 11.2, 13.5, 10.1],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'TRADERT22',
        data: [4.1, 6.5, 3.2, 8.7, 10.2, 5.6, 9.8, 7.4, 11.3, 6.9, 8.5, 12.7],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
      {
        label: 'AutoRebalance',
        data: [6.8, 4.2, 7.9, 9.3, 5.7, 11.2, 8.4, 10.6, 7.2, 9.8, 12.1, 8.9],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Chatosil',
        data: [3.5, 8.2, 5.7, 7.3, 9.1, 6.8, 10.5, 8.3, 4.9, 7.6, 11.8, 9.4],
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.5)',
      },
    ],
  };
  
  // 트레이더 데이터
  const traders = [
    {
      id: 'hummusXBT',
      name: 'hummusXBT',
      platform: 'Binance',
      annualReturn: 187,
      winRate: 68,
      avgProfit: 12.5,
      avgLoss: 4.2,
      maxDrawdown: 28,
      description: 'Binance 선물 거래소에서 활동하는 탑 트레이더로, 주로 비트코인과 이더리움 선물 거래에 집중합니다. 기술적 분석과 시장 심리를 결합한 전략으로 안정적인 수익률을 보여주고 있습니다.',
      tradingStyle: '스윙 트레이딩, 추세 추종',
      riskManagement: '철저한 포지션 사이징과 손절 관리',
      category: 'binance'
    },
    {
      id: 'TRADERT22',
      name: 'TRADERT22',
      platform: 'Binance',
      annualReturn: 142,
      winRate: 63,
      avgProfit: 9.8,
      avgLoss: 3.7,
      maxDrawdown: 22,
      description: '중장기 포지션을 선호하며 거시경제 지표와 온체인 데이터를 활용한 분석으로 트레이딩합니다. 변동성이 낮은 안정적인 수익 곡선이 특징입니다.',
      tradingStyle: '중장기 포지션, 펀더멘털 분석',
      riskManagement: '다각화된 포트폴리오, 단계적 진입/청산',
      category: 'binance'
    },
    {
      id: 'AutoRebalance',
      name: 'AutoRebalance',
      platform: 'Binance',
      annualReturn: 165,
      winRate: 59,
      avgProfit: 14.2,
      avgLoss: 5.8,
      maxDrawdown: 35,
      description: '알고리즘 기반 자동 리밸런싱 전략을 사용하여 시장 변동성을 활용합니다. 고빈도 거래와 통계적 차익거래 기법을 결합한 독특한 접근법을 보여줍니다.',
      tradingStyle: '알고리즘 트레이딩, 고빈도 거래',
      riskManagement: '자동화된 리스크 관리, 포지션 분산',
      category: 'binance'
    },
    {
      id: 'Chatosil',
      name: 'Chatosil',
      platform: 'Binance',
      annualReturn: 128,
      winRate: 72,
      avgProfit: 8.3,
      avgLoss: 3.2,
      maxDrawdown: 19,
      description: '보수적인 접근법으로 낮은 레버리지와 철저한 리스크 관리가 특징입니다. 승률이 높고 최대 손실폭이 작아 안정적인 수익을 추구하는 투자자에게 적합합니다.',
      tradingStyle: '스캘핑, 단기 거래',
      riskManagement: '낮은 레버리지, 높은 승률 전략',
      category: 'binance'
    },
    {
      id: 'GMXMaster',
      name: 'GMXMaster',
      platform: 'GMX',
      annualReturn: 156,
      winRate: 64,
      avgProfit: 11.7,
      avgLoss: 4.5,
      maxDrawdown: 26,
      description: '탈중앙화 거래소 GMX에서 활동하는 트레이더로, 온체인 데이터와 기술적 분석을 결합한 전략을 사용합니다. DeFi 생태계에 대한 깊은 이해를 바탕으로 트레이딩합니다.',
      tradingStyle: '온체인 데이터 활용, 기술적 분석',
      riskManagement: '스마트 컨트랙트 리스크 관리, 다중 지갑 운용',
      category: 'dex'
    },
    {
      id: 'HyperTrader',
      name: 'HyperTrader',
      platform: 'Hyperliquid',
      annualReturn: 173,
      winRate: 61,
      avgProfit: 13.8,
      avgLoss: 5.2,
      maxDrawdown: 32,
      description: 'Hyperliquid 플랫폼에서 활동하는 트레이더로, 고급 파생상품 전략과 변동성 거래에 특화되어 있습니다. 시장 비효율성을 활용한 차익거래 전략도 구사합니다.',
      tradingStyle: '변동성 거래, 파생상품 전략',
      riskManagement: '헤지 전략, 동적 포지션 조정',
      category: 'dex'
    },
    {
      id: 'ByteWizard',
      name: 'ByteWizard',
      platform: 'Bybit',
      annualReturn: 149,
      winRate: 67,
      avgProfit: 10.2,
      avgLoss: 3.8,
      maxDrawdown: 24,
      description: 'Bybit 거래소에서 활동하며 기술적 지표와 시장 심리를 결합한 전략을 사용합니다. 특히 변동성이 높은 시장에서 뛰어난 성과를 보여주고 있습니다.',
      tradingStyle: '기술적 분석, 심리적 지표 활용',
      riskManagement: '변동성 기반 포지션 사이징, 트레일링 스톱',
      category: 'cefi'
    },
    {
      id: 'QuantumTrade',
      name: 'QuantumTrade',
      platform: 'Bybit',
      annualReturn: 138,
      winRate: 70,
      avgProfit: 9.5,
      avgLoss: 3.4,
      maxDrawdown: 21,
      description: '양적 모델과 머신러닝 알고리즘을 활용한 트레이딩 전략을 구사합니다. 데이터 기반의 객관적인 접근법으로 감정을 배제한 트레이딩이 특징입니다.',
      tradingStyle: '양적 분석, 머신러닝 활용',
      riskManagement: '통계적 리스크 모델, 포트폴리오 최적화',
      category: 'cefi'
    },
  ];
  
  // 필터링된 트레이더 목록
  const filteredTraders = activeTab === 'all' 
    ? traders 
    : traders.filter(trader => trader.category === activeTab);
  
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Navigation />
      
      {/* 헤더 섹션 */}
      <header className="bg-gradient-to-r from-blue-900 to-black text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">수비 모드를 위한 <span className="text-yellow-400">탑 트레이더</span></h1>
          <p className="text-xl max-w-3xl mx-auto">
            상승장과 하락장 모두에서 안정적인 수익을 내는 검증된 트레이더들을 분석하고 활용하세요
          </p>
        </div>
      </header>

      {/* 트레이더 성과 차트 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">트레이더 <span className="text-yellow-500">월별 성과</span></h2>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg mb-10">
            <div className="h-[400px]">
              <Line 
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                    title: {
                      display: true,
                      text: '2024년 월별 수익률 (%)',
                    },
                  },
                }} 
                data={monthlyPerformanceData} 
              />
            </div>
            <p className="text-center text-gray-600 dark:text-gray-300 mt-4">
              상승장과 하락장 모두에서 안정적인 수익을 내는 트레이더들의 월별 성과 비교
            </p>
          </div>
        </div>
      </section>

      {/* 트레이더 목록 */}
      <section className="py-16 bg-gray-100 dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-6 text-center">탑 <span className="text-yellow-500">트레이더</span> 분석</h2>
          <p className="text-center text-gray-600 dark:text-gray-300 mb-12 max-w-3xl mx-auto">
            각 플랫폼에서 검증된 트레이더들의 상세 분석과 트레이딩 스타일을 확인하고 
            자신의 투자 성향에 맞는 트레이더를 선택하세요
          </p>
          
          {/* 필터 탭 */}
          <div className="flex justify-center mb-10">
            <div className="inline-flex rounded-md shadow-sm">
              <button
                onClick={() => setActiveTab('all')}
                className={`px-4 py-2 text-sm font-medium rounded-l-lg ${
                  activeTab === 'all'
                    ? 'bg-yellow-500 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                전체
              </button>
              <button
                onClick={() => setActiveTab('binance')}
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'binance'
                    ? 'bg-yellow-500 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                Binance
              </button>
              <button
                onClick={() => setActiveTab('cefi')}
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'cefi'
                    ? 'bg-yellow-500 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                Bybit
              </button>
              <button
                onClick={() => setActiveTab('dex')}
                className={`px-4 py-2 text-sm font-medium rounded-r-lg ${
                  activeTab === 'dex'
                    ? 'bg-yellow-500 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                탈중앙화(GMX/Hyperliquid)
              </button>
            </div>
          </div>
          
          {/* 트레이더 카드 그리드 */}
          <div className="grid md:grid-cols-2 gap-8">
            {filteredTraders.map((trader) => (
              <motion.div
                key={trader.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
              >
                <div className="flex items-center mb-4">
                  <div className="bg-purple-500 text-white p-3 rounded-full mr-4">
                    <FaUserTie size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">{trader.name}</h3>
                    <p className="text-gray-600 dark:text-gray-300">{trader.platform} 플랫폼</p>
                  </div>
                </div>
                
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  {trader.description}
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">연간 수익률</p>
                    <p className="text-lg font-bold text-green-500">+{trader.annualReturn}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">승률</p>
                    <p className="text-lg font-bold">{trader.winRate}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">평균 수익</p>
                    <p className="text-lg font-bold text-green-500">{trader.avgProfit}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">최대 낙폭</p>
                    <p className="text-lg font-bold text-red-500">-{trader.maxDrawdown}%</p>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="flex items-start">
                    <FaChartLine className="text-blue-500 mt-1 mr-2 flex-shrink-0" />
                    <div>
                      <p className="font-semibold mb-1">트레이딩 스타일</p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">{trader.tradingStyle}</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <FaBalanceScale className="text-blue-500 mt-1 mr-2 flex-shrink-0" />
                    <div>
                      <p className="font-semibold mb-1">리스크 관리</p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">{trader.riskManagement}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 카피트레이딩 전략 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">효과적인 <span className="text-yellow-500">카피트레이딩</span> 전략</h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg"
            >
              <div className="bg-blue-500 text-white p-3 rounded-full inline-block mb-4">
                <FaSearch size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">트레이더 선정 기준</h3>
              <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                <li>최소 1년 이상의 검증된 수익률</li>
                <li>안정적인 수익 곡선 (급격한 변동 없음)</li>
                <li>적절한 리스크 관리 (낮은 최대 낙폭)</li>
                <li>투자 성향과의 일치성</li>
                <li>투명한 거래 기록</li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg"
            >
              <div className="bg-green-500 text-white p-3 rounded-full inline-block mb-4">
                <FaBalanceScale size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">자본 배분 전략</h3>
              <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                <li>여러 트레이더에게 자본 분산</li>
                <li>플랫폼 다각화 (중앙화/탈중앙화)</li>
                <li>트레이딩 스타일 다각화</li>
                <li>초기에는 소액으로 테스트</li>
                <li>성과에 따른 동적 자본 재배분</li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg"
            >
              <div className="bg-purple-500 text-white p-3 rounded-full inline-block mb-4">
                <FaExchangeAlt size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">모니터링 및 최적화</h3>
              <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                <li>정기적인 성과 평가 (월간/분기)</li>
                <li>시장 상황에 따른 전략 조정</li>
                <li>지속적인 새로운 트레이더 탐색</li>
                <li>성과 저조 시 즉각적인 자본 회수</li>
                <li>성공적인 트레이더의 전략 학습</li>
              </ul>
            </motion.div>
          </div>
          
          <div className="mt-12 p-6 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
            <h3 className="text-xl font-bold mb-4 text-center text-yellow-600 dark:text-yellow-400">주의사항</h3>
            <p className="text-gray-600 dark:text-gray-300 text-center max-w-3xl mx-auto">
              카피트레이딩은 효과적인 수비 모드 전략이지만, 모든 투자에는 리스크가 따릅니다. 
              항상 자신이 감당할 수 있는 금액만 투자하고, 트레이더의 성과를 지속적으로 모니터링하세요. 
              과거의 성과가 미래 수익을 보장하지는 않습니다.
            </p>
          </div>
        </div>
      </section>

      {/* CTA 섹션 */}
      <section className="py-16 bg-gradient-to-r from-blue-900 to-black text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-6">검증된 트레이더와 함께 <span className="text-yellow-400">안정적인 수익</span>을 추구하세요</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            수비 모드에서는 검증된 트레이더들의 전략을 활용하여 시장 상황에 관계없이 꾸준한 수익을 창출할 수 있습니다.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="/strategy" className="bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
              전략 알아보기
            </a>
            <a href="/case-study" className="bg-transparent border-2 border-white hover:bg-white hover:text-black px-8 py-3 rounded-full font-bold transition-all">
              사례 연구 보기
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
                <FaUserTie className="text-yellow-500 text-2xl" />
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
