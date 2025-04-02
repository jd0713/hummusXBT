#!/usr/bin/env python3
import time
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib

# URL of the trader's page
URL = "https://portal.mirrorly.xyz/leaderboard/Binance/D9E14F24A64472D262BF72FC7F817CBE"

# XPath for the Next button (provided by the user)
NEXT_BUTTON_XPATH = "/html/body/div[1]/main/div[2]/div/button[2]"

def setup_driver():
    """Set up the Chrome WebDriver"""
    chrome_options = Options()
    # We need to see the browser to interact with it
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Set up the driver
    service = Service(ChromeDriverManager().install())
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
    try:
        next_button = driver.find_element(By.XPATH, NEXT_BUTTON_XPATH)
        
        # Try multiple click methods
        try:
            # Method 1: Regular click
            next_button.click()
            print("Clicked Next button using regular click")
            return True
        except (ElementClickInterceptedException, StaleElementReferenceException):
            try:
                # Method 2: JavaScript click
                driver.execute_script("arguments[0].click();", next_button)
                print("Clicked Next button using JavaScript")
                return True
            except:
                # Method 3: More aggressive JavaScript click
                driver.execute_script("""
                    var element = document.evaluate(
                        "/html/body/div[1]/main/div[2]/div/button[2]", 
                        document, 
                        null, 
                        XPathResult.FIRST_ORDERED_NODE_TYPE, 
                        null
                    ).singleNodeValue;
                    if(element) {
                        element.click();
                    }
                """)
                print("Clicked Next button using XPath JavaScript")
                return True
    except Exception as e:
        print(f"Failed to click Next button: {e}")
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

def unlimited_scrape():
    """
    Scraper that continues clicking Next until it detects the same page twice,
    indicating we've reached the end of the data
    """
    driver = setup_driver()
    all_closed_positions = []
    page_num = 1
    page_hashes = set()  # Set to store hashes of pages we've seen
    consecutive_duplicates = 0  # Counter for consecutive duplicate pages
    max_consecutive_duplicates = 3  # Stop after this many consecutive duplicates
    
    # Create DataFrame and files for incremental saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"closed_positions_{timestamp}.csv"
    excel_filename = f"closed_positions_{timestamp}.xlsx"
    df = pd.DataFrame()  # Empty DataFrame to start
    
    try:
        # Navigate to the URL
        print(f"Navigating to {URL}...")
        driver.get(URL)
        
        # Wait for the page to load
        print("Waiting for page to load...")
        time.sleep(5)  # Initial wait
        
        print("\n=== UNLIMITED SCRAPING MODE ===")
        print("1. Accept any cookies if prompted")
        print("2. Scroll down to the 'Closed Positions' section")
        print("3. When ready to start automatic scraping, press Enter")
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
                    
                    # Save incrementally to CSV and Excel
                    df.to_csv(csv_filename, index=False)
                    df.to_excel(excel_filename, index=False)
                    print(f"Incrementally saved {len(df)} total positions to {csv_filename} and {excel_filename}")
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
            print(f"\nScraping complete! Collected {len(all_closed_positions)} positions from {page_num} pages.")
            print(f"Data saved to {csv_filename} and {excel_filename}")
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
    unlimited_scrape()
