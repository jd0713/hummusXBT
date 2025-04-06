'use client';

import Navigation from "../components/Navigation";
import { FaChartLine, FaArrowUp, FaArrowDown, FaExchangeAlt, FaLightbulb } from "react-icons/fa";
import { motion } from "framer-motion";

export default function CaseStudyPage() {
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Navigation />
      
      {/* 헤더 섹션 */}
      <header className="bg-gradient-to-r from-blue-900 to-black text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">실제 <span className="text-yellow-400">사례 연구</span></h1>
          <p className="text-xl max-w-3xl mx-auto">
            2017년부터 현재까지의 암호화폐 시장 사이클에서 얻은 실제 경험과 교훈
          </p>
        </div>
      </header>

      {/* 2017-2018 사이클 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            2017-2018 <span className="text-green-500">불장과 하락장</span>
          </h2>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaArrowUp className="text-green-500 mr-3" />
              상승 사이클 (2017년 10월 - 2018년 1월)
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-xl font-semibold mb-4">시장 상황</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300 mb-6">
                  <li>ICO 붐과 함께 알트코인 시장 폭발적 성장</li>
                  <li>비트코인 20,000달러 돌파</li>
                  <li>일반 대중의 암호화폐 관심 급증</li>
                  <li>미디어의 광범위한 보도</li>
                </ul>
                
                <h4 className="text-xl font-semibold mb-4">구사한 전략</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>하루 수익률 3%의 차익거래 활용</li>
                  <li>BTC, ETH, XRP, LTC 등 메이저 코인 집중 투자</li>
                  <li>ICO 참여로 높은 초기 수익률 달성</li>
                  <li>레버리지 활용한 공격적 포지션</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/30 p-6 rounded-lg">
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">성과</h4>
                <div className="mb-6">
                  <p className="text-5xl font-bold text-green-500 mb-2">4억 → 100억</p>
                  <p className="text-gray-600 dark:text-gray-300">3개월 간의 자산 성장</p>
                </div>
                
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">핵심 결정</h4>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  <span className="font-bold">2018년 초 전체 포지션 청산</span> - 시장 과열 신호를 인식하고 
                  수비 모드로 전환하여 자산을 보존했습니다.
                </p>
                <div className="flex items-center">
                  <FaLightbulb className="text-yellow-500 mr-2" />
                  <p className="text-gray-600 dark:text-gray-300 italic">
                    "욕심을 부리지 않고 적시에 수비 모드로 전환한 것이 성공의 핵심이었습니다."
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaArrowDown className="text-red-500 mr-3" />
              하락 사이클 (2018년 - 2020년 초)
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-xl font-semibold mb-4">시장 상황</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300 mb-6">
                  <li>비트코인 3,000달러까지 하락</li>
                  <li>ICO 프로젝트 대부분 실패</li>
                  <li>암호화폐 시장에 대한 회의적 시각 확산</li>
                  <li>규제 불확실성 증가</li>
                </ul>
                
                <h4 className="text-xl font-semibold mb-4">구사한 전략</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>OTC(장외거래) 중개를 통한 안정적 수익 창출</li>
                  <li>차익거래 전략으로 시장 방향성 무관한 수익 추구</li>
                  <li>스테이블코인 활용한 자산 보존</li>
                  <li>소규모 투자로 유망 프로젝트 발굴</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/30 p-6 rounded-lg">
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">성과</h4>
                <div className="mb-6">
                  <p className="text-3xl font-bold text-green-500 mb-2">자산 보존 + 꾸준한 성장</p>
                  <p className="text-gray-600 dark:text-gray-300">하락장에서도 안정적 수익 창출</p>
                </div>
                
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">핵심 교훈</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>하락장에서는 자본 보존이 최우선</li>
                  <li>방향성 무관한 전략의 중요성</li>
                  <li>꾸준한 수익이 폭발적 수익보다 중요</li>
                </ul>
                <div className="flex items-center mt-4">
                  <FaLightbulb className="text-yellow-500 mr-2" />
                  <p className="text-gray-600 dark:text-gray-300 italic">
                    "하락장은 다음 상승장을 준비하는 시간입니다."
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* 2020-2021 사이클 */}
      <section className="py-16 bg-gray-100 dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            2020-2021 <span className="text-yellow-500">사이클과 교훈</span>
          </h2>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg mb-10"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaArrowUp className="text-green-500 mr-3" />
              상승 사이클 (2020년 3월 - 2021년 초)
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-xl font-semibold mb-4">시장 상황</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300 mb-6">
                  <li>코로나 사태로 인한 글로벌 유동성 확대</li>
                  <li>기관 투자자들의 비트코인 매수 증가</li>
                  <li>DeFi 붐과 새로운 생태계 확장</li>
                  <li>비트코인 69,000달러까지 상승</li>
                </ul>
                
                <h4 className="text-xl font-semibold mb-4">구사한 전략</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>펀드 해산 후 개인 자산으로 전환</li>
                  <li>주식 + 코인 레버리지 롱 포지션</li>
                  <li>DeFi 프로토콜 초기 참여</li>
                  <li>NFT 시장 참여</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/30 p-6 rounded-lg">
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">성과</h4>
                <div className="mb-6">
                  <p className="text-3xl font-bold text-green-500 mb-2">세자리수 중반 억대 자산 달성</p>
                  <p className="text-gray-600 dark:text-gray-300">공격적 전략으로 큰 수익 실현</p>
                </div>
                
                <h4 className="text-xl font-semibold mb-4 text-red-600 dark:text-red-400">실패 지점</h4>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  <span className="font-bold">수비 모드로 전환 실패</span> - 상승장이 끝나는 시점에 
                  공격 모드를 유지하여 하락장에서 큰 손실 발생
                </p>
                <div className="flex items-center">
                  <FaLightbulb className="text-yellow-500 mr-2" />
                  <p className="text-gray-600 dark:text-gray-300 italic">
                    "빠르게 올랐던 만큼 떨어지는 것도 빨랐습니다."
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg"
          >
            <h3 className="text-2xl font-bold mb-6 flex items-center">
              <FaExchangeAlt className="text-purple-500 mr-3" />
              핵심 교훈과 비교 분석
            </h3>
            
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg overflow-hidden">
                <thead className="bg-gray-100 dark:bg-gray-700">
                  <tr>
                    <th className="py-3 px-4 text-left">항목</th>
                    <th className="py-3 px-4 text-left">2017-2018 사이클</th>
                    <th className="py-3 px-4 text-left">2020-2021 사이클</th>
                    <th className="py-3 px-4 text-left">교훈</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
                  <tr>
                    <td className="py-3 px-4 font-medium">전략 전환</td>
                    <td className="py-3 px-4 text-green-500">적시에 수비 모드 전환 성공</td>
                    <td className="py-3 px-4 text-red-500">수비 모드 전환 실패</td>
                    <td className="py-3 px-4">시장 과열 신호를 무시하지 말 것</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">자산 관리</td>
                    <td className="py-3 px-4 text-green-500">이익 실현 및 자산 보존</td>
                    <td className="py-3 px-4 text-red-500">레버리지 유지로 손실 확대</td>
                    <td className="py-3 px-4">단계적 이익 실현의 중요성</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">심리적 요인</td>
                    <td className="py-3 px-4 text-green-500">욕심 통제 성공</td>
                    <td className="py-3 px-4 text-red-500">FOMO와 과신으로 판단 오류</td>
                    <td className="py-3 px-4">감정을 배제한 객관적 판단</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">시장 분석</td>
                    <td className="py-3 px-4 text-green-500">과열 신호 인식</td>
                    <td className="py-3 px-4 text-red-500">"이번엔 다르다" 사고방식</td>
                    <td className="py-3 px-4">역사는 반복된다는 사실 인식</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div className="mt-8 p-6 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
              <h4 className="text-xl font-semibold mb-4 text-yellow-600 dark:text-yellow-400">종합 교훈</h4>
              <p className="text-gray-600 dark:text-gray-300">
                두 사이클의 경험을 통해 얻은 가장 중요한 교훈은 <span className="font-bold">시장 사이클에 따른 전략 전환의 중요성</span>입니다. 
                상승장에서는 공격적으로, 하락장에서는 방어적으로 접근하는 것이 장기적인 성공의 핵심입니다. 
                특히 상승장에서 하락장으로 전환되는 시점을 파악하고 적절히 대응하는 것이 가장 중요합니다.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* 현재 사이클 분석 */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">
            현재 사이클 <span className="text-blue-500">분석 및 전략</span>
          </h2>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-8 shadow-lg">
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-xl font-semibold mb-4">현재 시장 상황</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300 mb-6">
                  <li>비트코인 할빙 이후 불확실한 시장 상황</li>
                  <li>거시경제적 불확실성 (금리, 인플레이션)</li>
                  <li>기관 투자자들의 참여 증가</li>
                  <li>규제 환경의 변화</li>
                </ul>
                
                <h4 className="text-xl font-semibold mb-4">현재 구사 중인 전략</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
                  <li>수비 모드 유지</li>
                  <li>검증된 탑 트레이더 카피트레이딩</li>
                  <li>자산의 상당 부분 스테이블코인으로 보유</li>
                  <li>소규모 DCA(달러 코스트 애버리징) 전략</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/30 p-6 rounded-lg">
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">전략적 접근</h4>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  현재는 <span className="font-bold">수비 모드</span>를 유지하며 다음 상승 사이클을 준비하는 시기입니다. 
                  과거의 교훈을 바탕으로 자본을 보존하면서도 안정적인 수익을 창출할 수 있는 전략에 집중하고 있습니다.
                </p>
                
                <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">탑 트레이더 활용</h4>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  Binance, Bybit, Hyperliquid, GMX 등의 플랫폼에서 검증된 트레이더들의 전략을 활용하여 
                  시장 상황에 관계없이 꾸준한 수익을 추구하고 있습니다.
                </p>
                
                <div className="flex items-center mt-4">
                  <FaLightbulb className="text-yellow-500 mr-2" />
                  <p className="text-gray-600 dark:text-gray-300 italic">
                    "지금은 인내의 시간입니다. 다음 기회가 왔을 때 최대한 활용할 수 있도록 준비하세요."
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA 섹션 */}
      <section className="py-16 bg-gradient-to-r from-blue-900 to-black text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-6">과거의 <span className="text-yellow-400">교훈</span>으로 미래를 준비하세요</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            암호화폐 시장의 사이클을 이해하고 적절한 전략을 구사하는 것이 장기적인 성공의 핵심입니다.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="/strategy" className="bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-3 rounded-full font-bold transition-all">
              전략 알아보기
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
