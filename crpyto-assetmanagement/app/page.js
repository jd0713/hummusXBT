import Image from "next/image";
import Navigation from "./components/Navigation";
import { FaChartLine, FaShieldAlt, FaExchangeAlt, FaUserTie, FaBitcoin } from "react-icons/fa";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Navigation />
      
      {/* 히어로 섹션 */}
      <section className="relative h-[80vh] flex items-center justify-center bg-gradient-to-r from-blue-900 to-black text-white overflow-hidden">
        <div className="container mx-auto px-6 z-10 text-center">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">암호화폐 시장의 <span className="text-yellow-400">효과적인 자산 관리</span></h1>
          <p className="text-xl md:text-2xl mb-8 max-w-3xl mx-auto">공격과 수비의 균형으로 어떤 시장 상황에서도 자산을 보호하고 성장시키는 전략</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="#strategy-section" className="bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
              전략 알아보기
            </a>
            <a href="#case-study-section" className="bg-transparent border-2 border-white hover:bg-white hover:text-black px-8 py-3 rounded-full font-bold transition-all">
              사례 연구
            </a>
          </div>
        </div>
      </section>

      {/* 핵심 개념 섹션 */}
      <section className="py-20 bg-white dark:bg-gray-800" id="strategy-section">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">암호화폐 자산 관리의 <span className="text-yellow-500">핵심 원칙</span></h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-10">
            {/* 카드 1: 공격 모드 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-red-500 text-white p-3 rounded-full inline-block mb-4">
                <FaChartLine size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">공격 모드</h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                명확한 상승장에서 적극적인 포지션 구축으로 수익 극대화
              </p>
              <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2">
                <li>알트코인 레버리지 롱</li>
                <li>비트코인 네이키드 롱</li>
                <li>고위험 고수익 전략 활용</li>
              </ul>
            </div>
            
            {/* 카드 2: 수비 모드 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-blue-500 text-white p-3 rounded-full inline-block mb-4">
                <FaShieldAlt size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">수비 모드</h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                불확실한 시장에서 자산 보존과 안정적인 수익 추구
              </p>
              <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2">
                <li>카피트레이딩</li>
                <li>차익거래</li>
                <li>이벤트 드리븐 전략</li>
              </ul>
            </div>
            
            {/* 카드 3: 전환점 파악 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-green-500 text-white p-3 rounded-full inline-block mb-4">
                <FaExchangeAlt size={24} />
              </div>
              <h3 className="text-xl font-bold mb-4">전환점 파악</h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                시장 사이클 전환 시점을 파악하는 전략적 접근
              </p>
              <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2">
                <li>기술적 지표 분석</li>
                <li>시장 심리 모니터링</li>
                <li>리스크 관리 시스템</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 사례 연구 섹션 */}
      <section className="py-20 bg-gray-100 dark:bg-gray-900" id="case-study-section">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">실제 <span className="text-yellow-500">사례 연구</span></h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg mb-10">
            <h3 className="text-2xl font-bold mb-4">2017-2018 불장: 공격에서 수비로</h3>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="md:w-1/2">
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  2017년 10월부터 3개월간의 알트코인 불장 동안 공격적인 전략으로 4억에서 100억으로 자산 성장
                </p>
                <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                  <li>하루 수익률 3%의 차익거래 활용</li>
                  <li>BTC, ETH, XRP, LTC 등 메이저 코인 집중 투자</li>
                  <li>2018년 초 전체 포지션 청산으로 수비 모드 전환</li>
                </ul>
              </div>
              <div className="md:w-1/2 bg-gray-100 dark:bg-gray-700 rounded-lg p-4 flex items-center justify-center">
                <div className="text-center">
                  <p className="text-5xl font-bold text-green-500 mb-2">25배</p>
                  <p className="text-gray-600 dark:text-gray-300">자산 성장률</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
            <h3 className="text-2xl font-bold mb-4">2020-2021 사이클: 교훈</h3>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="md:w-1/2">
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  2020년 3월 코로나 사태 이후 공격 모드로 전환하여 큰 성과를 거두었으나, 하락장에서 수비 모드로 전환하지 못한 실패 사례
                </p>
                <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                  <li>주식 + 코인 레버리지 롱으로 큰 수익 달성</li>
                  <li>하락 사이클 진입 시 수비 모드 전환 실패</li>
                  <li>교훈: 수비 모드의 중요성과 적시 전환의 필요성</li>
                </ul>
              </div>
              <div className="md:w-1/2 bg-gray-100 dark:bg-gray-700 rounded-lg p-4 flex items-center justify-center">
                <div className="text-center">
                  <p className="text-xl text-red-500 mb-2">"빠르게 올랐던 만큼 떨어지는 것도 빨랐다"</p>
                  <p className="text-gray-600 dark:text-gray-300">핵심 교훈</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 탑 트레이더 섹션 */}
      <section className="py-20 bg-white dark:bg-gray-800" id="traders-section">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">수비 모드를 위한 <span className="text-yellow-500">탑 트레이더</span></h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* 트레이더 1 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-purple-500 text-white p-3 rounded-full mr-4">
                  <FaUserTie size={20} />
                </div>
                <h3 className="text-lg font-bold">hummusXBT</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">Binance 선물 거래소</p>
              <div className="bg-gray-100 dark:bg-gray-600 rounded-lg p-3">
                <p className="text-sm font-semibold mb-1">연간 수익률</p>
                <p className="text-2xl font-bold text-green-500">+187%</p>
              </div>
            </div>
            
            {/* 트레이더 2 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-purple-500 text-white p-3 rounded-full mr-4">
                  <FaUserTie size={20} />
                </div>
                <h3 className="text-lg font-bold">TRADERT22</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">Binance 선물 거래소</p>
              <div className="bg-gray-100 dark:bg-gray-600 rounded-lg p-3">
                <p className="text-sm font-semibold mb-1">연간 수익률</p>
                <p className="text-2xl font-bold text-green-500">+142%</p>
              </div>
            </div>
            
            {/* 트레이더 3 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-purple-500 text-white p-3 rounded-full mr-4">
                  <FaUserTie size={20} />
                </div>
                <h3 className="text-lg font-bold">AutoRebalance</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">Binance 선물 거래소</p>
              <div className="bg-gray-100 dark:bg-gray-600 rounded-lg p-3">
                <p className="text-sm font-semibold mb-1">연간 수익률</p>
                <p className="text-2xl font-bold text-green-500">+165%</p>
              </div>
            </div>
            
            {/* 트레이더 4 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-purple-500 text-white p-3 rounded-full mr-4">
                  <FaUserTie size={20} />
                </div>
                <h3 className="text-lg font-bold">Chatosil</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">Binance 선물 거래소</p>
              <div className="bg-gray-100 dark:bg-gray-600 rounded-lg p-3">
                <p className="text-sm font-semibold mb-1">연간 수익률</p>
                <p className="text-2xl font-bold text-green-500">+128%</p>
              </div>
            </div>
          </div>
          
          <div className="text-center mt-10">
            <a href="/traders" className="inline-block bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
              트레이더 분석 보기
            </a>
          </div>
        </div>
      </section>

      {/* CTA 섹션 */}
      <section className="py-16 bg-gradient-to-r from-blue-900 to-black text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">암호화폐 시장에서 <span className="text-yellow-400">생존하고 번영하기</span></h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">지금은 수비 모드가 필요한 시기입니다. 효과적인 자산 관리 전략으로 다음 사이클을 준비하세요.</p>
          <a href="/strategy" className="inline-block bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
            전략 자세히 알아보기
          </a>
        </div>
      </section>

      {/* 푸터 */}
      <footer className="bg-gray-900 text-white py-10">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <div className="flex items-center space-x-2 text-xl font-bold">
                <FaBitcoin className="text-yellow-500 text-2xl" />
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
