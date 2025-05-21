#!/usr/bin/env python3
import time
import os
import pandas as pd
import argparse
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib

# 트레이더 설정 모듈 임포트
from trader_config import get_trader, get_all_trader_ids

# XPath for the Next button (multiple options to try)
NEXT_BUTTON_XPATHS = [
    "/html/body/div[1]/main/div[2]/div/button[2]",  # 원래 경로
    "/html/body/div[1]/main/div[3]/div/button[2]"   # 새로운 경로
]

# 성공한 XPath를 저장할 변수
SUCCESSFUL_XPATH = None

def setup_driver():
    """Set up the Chrome WebDriver"""
    chrome_options = Options()
    # We need to see the browser to interact with it
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Set up the driver with local ChromeDriver path
    chromedriver_path = "/Users/jdpark/Downloads/chromedriver/chromedriver-mac-arm64/chromedriver"
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def parse_closed_positions(soup):
    """Parse the closed positions table from the BeautifulSoup object"""
    closed_positions = []
    
    # Find the closed positions table
    closed_positions_table = None
    
    # Find the heading "Closed Positions" and then the next table
    closed_positions_h1 = soup.find('h1', string='Closed Positions')
    if closed_positions_h1:
        # The table is the next table element after the heading
        closed_positions_table = closed_positions_h1.find_next('table')
    
    if not closed_positions_table:
        print("Closed positions table not found")
        return closed_positions
    
    # Find all rows in the table body
    rows = closed_positions_table.find('tbody').find_all('tr')
    
    for row in rows:
        # Extract data from each cell
        cells = row.find_all('td')
        if len(cells) >= 6:  # Ensure we have all the required cells
            # Extract symbol
            symbol_cell = cells[0]
            symbol_span = symbol_cell.find('span', class_='font-bold')
            symbol = symbol_span.text.strip() if symbol_span else "Unknown"
            
            # Extract direction (Long/Short) based on the SVG color class
            direction_svg = symbol_cell.find('svg')
            direction = "Long" if direction_svg and "color-data-green" in str(direction_svg.get('class', [])) else "Short"
            
            # Extract entry price
            entry_cell = cells[1]
            entry_span = entry_cell.find('span', class_='color-white-blue')
            entry_price = entry_span.text.strip() if entry_span else "Unknown"
            
            # Extract max size
            max_size_cell = cells[2]
            max_size_spans = max_size_cell.find_all('span')
            max_size = max_size_spans[0].text.strip() if len(max_size_spans) > 0 else "Unknown"
            max_size_usdt = max_size_spans[1].text.strip() if len(max_size_spans) > 1 else "Unknown"
            
            # Extract realized profit/loss
            realized_cell = cells[3]
            realized_spans = realized_cell.find_all('span')
            realized_pnl = realized_spans[0].text.strip() if len(realized_spans) > 0 else "Unknown"
            realized_pnl_percent = realized_spans[1].text.strip() if len(realized_spans) > 1 else "Unknown"
            
            # Extract period (open and close time)
            period_cell = cells[4]
            period_spans = period_cell.find_all('span', class_='color-white-blue')
            open_time = period_spans[0].text.strip() if len(period_spans) > 0 else "Unknown"
            close_time = period_spans[1].text.strip() if len(period_spans) > 1 else "Unknown"
            
            # Extract runup/drawdown
            runup_drawdown_cell = cells[5]
            runup_drawdown_spans = runup_drawdown_cell.find_all('span')
            runup = runup_drawdown_spans[0].text.strip() if len(runup_drawdown_spans) > 0 else "Unknown"
            drawdown = runup_drawdown_spans[1].text.strip() if len(runup_drawdown_spans) > 1 else "Unknown"
            
            # Create a dictionary for this position
            position = {
                'Symbol': symbol,
                'Direction': direction,
                'Entry Price': entry_price,
                'Max Size': max_size,
                'Max Size USDT': max_size_usdt,
                'Realized PnL': realized_pnl,
                'Realized PnL %': realized_pnl_percent,
                'Open Time': open_time,
                'Close Time': close_time,
                'Runup': runup,
                'Drawdown': drawdown
            }
            
            closed_positions.append(position)
    
    return closed_positions

def force_click_next_button(driver):
    """Force click the Next button regardless of its disabled state"""
    global SUCCESSFUL_XPATH
    
    # 성공한 XPath가 있는 경우 해당 XPath만 사용
    if SUCCESSFUL_XPATH:
        try:
            next_button = driver.find_element(By.XPATH, SUCCESSFUL_XPATH)
            try:
                next_button.click()
                print(f"Clicked Next button using remembered XPath: {SUCCESSFUL_XPATH}")
                return True
            except (ElementClickInterceptedException, StaleElementReferenceException):
                try:
                    driver.execute_script("arguments[0].click();", next_button)
                    print(f"Clicked Next button using JavaScript with remembered XPath: {SUCCESSFUL_XPATH}")
                    return True
                except Exception:
                    pass
        except Exception:
            # 성공했던 XPath가 실패하면 다시 모든 XPath 시도
            print(f"Previously successful XPath '{SUCCESSFUL_XPATH}' failed, trying all XPaths again")
            SUCCESSFUL_XPATH = None
    
    # 여러 XPath를 시도
    for xpath in NEXT_BUTTON_XPATHS:
        try:
            next_button = driver.find_element(By.XPATH, xpath)
            
            # Try multiple click methods
            try:
                # Method 1: Regular click
                next_button.click()
                print(f"Clicked Next button using regular click (XPath: {xpath})")
                SUCCESSFUL_XPATH = xpath  # 성공한 XPath 저장
                return True
            except (ElementClickInterceptedException, StaleElementReferenceException):
                try:
                    # Method 2: JavaScript click
                    driver.execute_script("arguments[0].click();", next_button)
                    print(f"Clicked Next button using JavaScript (XPath: {xpath})")
                    SUCCESSFUL_XPATH = xpath  # 성공한 XPath 저장
                    return True
                except:
                    # Method 3: More aggressive JavaScript click
                    driver.execute_script(f"""
                        var element = document.evaluate(
                            "{xpath}", 
                            document, 
                            null, 
                            XPathResult.FIRST_ORDERED_NODE_TYPE, 
                            null
                        ).singleNodeValue;
                        if(element) {{
                            element.click();
                        }}
                    """)
                    print(f"Clicked Next button using XPath JavaScript (XPath: {xpath})")
                    SUCCESSFUL_XPATH = xpath  # 성공한 XPath 저장
                    return True
        except Exception as e:
            print(f"Failed to click Next button with XPath '{xpath}': {e}")
    
    # 모든 XPath가 실패한 경우
    print("Failed to click Next button after trying all XPaths")
    return False

def get_page_hash(soup):
    """Generate a hash of the closed positions table to detect duplicate pages"""
    # Find the closed positions table
    closed_positions_table = None
    closed_positions_h1 = soup.find('h1', string='Closed Positions')
    if closed_positions_h1:
        closed_positions_table = closed_positions_h1.find_next('table')
    
    if not closed_positions_table:
        return None
    
    # Get the HTML of the table body
    table_body = closed_positions_table.find('tbody')
    if not table_body:
        return None
    
    # Generate a hash of the table body HTML
    return hashlib.md5(str(table_body).encode('utf-8')).hexdigest()

def unlimited_scrape(trader_id=None):
    """
    Scraper that continues clicking Next until it detects the same page twice,
    indicating we've reached the end of the data
    
    Args:
        trader_id (str): ID of the trader to scrape. If None, will prompt for selection.
    """
    # 트레이더 선택 또는 확인
    if trader_id is None:
        # 사용 가능한 트레이더 목록 출력
        trader_ids = get_all_trader_ids()
        print("\n=== 사용 가능한 트레이더 목록 ===")
        for i, tid in enumerate(trader_ids, 1):
            trader = get_trader(tid)
            print(f"{i}. {trader['name']} - {trader['description']}")
        
        # 사용자에게 트레이더 선택 요청
        selection = input("\n스크래핑할 트레이더 번호를 선택하세요: ")
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
    trader_url = trader['url']
    trader_name = trader['name']
    
    driver = setup_driver()
    all_closed_positions = []
    page_num = 1
    page_hashes = set()  # Set to store hashes of pages we've seen
    consecutive_duplicates = 0  # Counter for consecutive duplicate pages
    max_consecutive_duplicates = 3  # Stop after this many consecutive duplicates
    
    # 트레이더별 폴더 확인 및 생성
    trader_data_dir = f"trader_data/{trader_id}"
    if not os.path.exists(trader_data_dir):
        os.makedirs(trader_data_dir)
        print(f"트레이더 데이터 디렉토리 생성: {trader_data_dir}")
    
    # Create DataFrame and file for incremental saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{trader_data_dir}/{trader_id}_closed_positions_{timestamp}.csv"
    df = pd.DataFrame()  # Empty DataFrame to start
    
    try:
        # Navigate to the URL
        print(f"\n=== {trader_name} 트레이더 스크래핑 시작 ===")
        print(f"URL: {trader_url}")
        driver.get(trader_url)
        
        # Wait for the page to load
        print("\n페이지 로드 중...")
        time.sleep(5)  # Initial wait
        
        print("\n=== 무제한 스크래핑 모드 ===")
        print("1. 쿠키 수락 창이 나타나면 수락하세요")
        print("2. 'Closed Positions' 섹션으로 스크롤하세요")
        print("3. 자동 스크래핑을 시작할 준비가 되면 Enter를 누르세요")
        print("===============================\n")
        
        # Wait for user to set up the page
        input("Press Enter when you're ready to start automatic scraping: ")
        
        # Now automatically scrape all pages
        while True:
            print(f"\nProcessing page {page_num}...")
            
            # Parse the current page
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Get hash of current page
            current_hash = get_page_hash(soup)
            if not current_hash:
                print("Could not generate hash for current page")
                break
            
            # Check if we've seen this page before
            if current_hash in page_hashes:
                consecutive_duplicates += 1
                print(f"Duplicate page detected ({consecutive_duplicates}/{max_consecutive_duplicates})")
                
                if consecutive_duplicates >= max_consecutive_duplicates:
                    print(f"Detected {max_consecutive_duplicates} consecutive duplicate pages. Reached the end of data.")
                    break
            else:
                # New page, reset counter and add hash to set
                consecutive_duplicates = 0
                page_hashes.add(current_hash)
                
                # Parse positions from this page
                positions = parse_closed_positions(soup)
                
                if positions:
                    # Add positions to our collection
                    all_closed_positions.extend(positions)
                    print(f"Found {len(positions)} positions on page {page_num}")
                    
                    # Add to DataFrame and save incrementally
                    new_df = pd.DataFrame(positions)
                    df = pd.concat([df, new_df], ignore_index=True)
                    
                    # Save incrementally to CSV only
                    df.to_csv(csv_filename, index=False)
                    print(f"Incrementally saved {len(df)} total positions to {csv_filename}")
                else:
                    print(f"No positions found on page {page_num}")
            
            # Try to click the Next button
            print("Attempting to click Next button...")
            if force_click_next_button(driver):
                # Wait for the page to load after clicking Next
                time.sleep(3)
                page_num += 1
            else:
                print("Failed to click Next button after multiple attempts. Stopping.")
                break
        
        # Final save of all data
        if all_closed_positions:
            print(f"\n스크래핑 완료! {page_num}개 페이지에서 {len(all_closed_positions)}개 포지션을 수집했습니다.")
            print(f"데이터가 {csv_filename}에 저장되었습니다.")
        else:
            print("No closed positions found")
        
    finally:
        # Ask if user wants to close the browser
        close_input = input("\nDo you want to close the browser? (y/n): ")
        if close_input.lower() in ['y', 'yes']:
            print("Closing browser...")
            driver.quit()
        else:
            print("Browser will remain open. You can close it manually when done.")

if __name__ == "__main__":
    # 커맨드 라인 인자 처리
    parser = argparse.ArgumentParser(description="바이낸스 트레이더 포지션 스크래퍼")
    parser.add_argument("-t", "--trader", help="스크래핑할 트레이더 ID", type=str)
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
    
    # 스크래핑 실행
    unlimited_scrape(trader_id)
