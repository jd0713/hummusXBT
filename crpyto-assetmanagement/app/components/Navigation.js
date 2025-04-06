'use client';

import Link from 'next/link';
import { useState } from 'react';
import { FaBitcoin, FaChartLine, FaUserTie } from 'react-icons/fa';
import { IoMenu, IoClose } from 'react-icons/io5';

export default function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="bg-gray-900 text-white py-4 px-6 sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="flex items-center space-x-2 text-xl font-bold">
          <FaBitcoin className="text-yellow-500 text-2xl" />
          <span>암호화폐 자산 관리</span>
        </Link>

        {/* 모바일 메뉴 버튼 */}
        <button 
          className="md:hidden text-2xl"
          onClick={toggleMenu}
        >
          {isMenuOpen ? <IoClose /> : <IoMenu />}
        </button>

        {/* 데스크톱 메뉴 */}
        <div className="hidden md:flex space-x-6">
          <Link href="/strategy" className="hover:text-yellow-400 flex items-center space-x-1">
            <FaChartLine />
            <span>전략</span>
          </Link>
          <Link href="/case-study" className="hover:text-yellow-400 flex items-center space-x-1">
            <FaChartLine />
            <span>사례 연구</span>
          </Link>
          <Link href="/traders" className="hover:text-yellow-400 flex items-center space-x-1">
            <FaUserTie />
            <span>탑 트레이더</span>
          </Link>
        </div>
      </div>

      {/* 모바일 메뉴 */}
      {isMenuOpen && (
        <div className="md:hidden mt-4 flex flex-col space-y-4 py-4">
          <Link href="/strategy" className="hover:text-yellow-400 flex items-center space-x-2 px-4">
            <FaChartLine />
            <span>전략</span>
          </Link>
          <Link href="/case-study" className="hover:text-yellow-400 flex items-center space-x-2 px-4">
            <FaChartLine />
            <span>사례 연구</span>
          </Link>
          <Link href="/traders" className="hover:text-yellow-400 flex items-center space-x-2 px-4">
            <FaUserTie />
            <span>탑 트레이더</span>
          </Link>
        </div>
      )}
    </nav>
  );
}
