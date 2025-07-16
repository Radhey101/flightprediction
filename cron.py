from codecs import BOM_BE
import os
from re import search
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import InvalidSessionIdException
import time
import pandas as pd
import re
from datetime import datetime, timedelta
import random

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=options)

 # List of IATA codes
airports = ['BOM', 'DEL', 'BLR', 'HYD', 'CCU', 'MAA', 'AMD', 'PNQ', 'GOI', 'COK']
airports = ['BOM', 'CCU']
combinations = list(itertools.permutations(airports, 2))
for Source, Destination in combinations:
    driver = get_driver()
    try:
        # Get today's date
        start_date = datetime.today() + timedelta(days=11)

        # Define end date one month later
        end_date = start_date + timedelta(days=20)
        start_date = '2026-06-02'
        # Define end date one month later
        end_date = '2026-06-02'
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)

        # Print the list of dates

        for date in date_range:
            flights = []
            cubin_class = ['e']
            for cubin in cubin_class:
                # Open Flipkart Flights
                DepartureDate = date.strftime("%d%m%Y")
                com_str = Source + '-' + Destination + '-' + DepartureDate
                url = "https://www.flipkart.com/travel/flights/search?trips=" + com_str + "&travellers=1-0-0&class="+cubin+"&tripType=ONE_WAY&isIntl=false"
                driver.get(url)

                # Wait until the first card loads
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "DOjaWF"))
                )

                # Lazy load scroll loop
                SCROLL_PAUSE = 2
                max_scrolls = 10
                prev_count = 0

                for i in range(max_scrolls):
                    # Scroll to bottom
                    driver.execute_script("window.scrollBy(0, 1000);")
                    time.sleep(SCROLL_PAUSE)

                    # Count current cards
                    cards = driver.find_elements(By.CLASS_NAME, "eivht0")
                    current_count = len(cards)

                    print(f"Scroll {i + 1}: {current_count} flights loaded")

                    if current_count == prev_count:
                        print("No new flights loaded. Reached end.")
                        break
                    prev_count = current_count

                print(f"✅ Total flights found: {len(cards)}")

                # Print basic data from each card
                for i, card in enumerate(cards):
                    try:
                        airline = card.find_elements(By.TAG_NAME, "span")

                        price_raw = card.find_element(By.CSS_SELECTOR, ".p23Ra6 div").text.strip()
                        price = re.sub(r'₹|,', '', price_raw)
                        now = datetime.now()
                        # Format as string
                        SearchDate = now.strftime("%Y-%m-%d %H:%M:%S")
                        flights.append({
                            "Airline": airline[0].text,
                            "Departure": airline[2].text,
                            "Arrival": airline[5].text,
                            "Duration": airline[3].text,
                            "Stop": airline[4].text,
                            "Price": price,
                            "SearchDate": SearchDate,
                            "DepartureDate": DepartureDate,
                            "Source": Source,
                            "Destination": Destination,
                            "Day": airline[6].text,
                            "Class": cubin
                        })


                    except Exception as e:
                        print(f"Error on flight {i + 1}: {e}")

            df = pd.DataFrame(flights)
            # Save to CSV
            csv_file = "flipkart_flights_june_for_demo.csv"
            # Check if CSV exists
            if os.path.exists(csv_file):
                df_existing = pd.read_csv(csv_file)

                # Concatenate and drop duplicates
                df_combined = pd.concat([df_existing, df])
                df_combined.drop_duplicates(inplace=True)

                # Save back to CSV
                df_combined.to_csv(csv_file, index=False)
                print("✅ New unique records appended (if any) and saved.")
            else:
                # File doesn't exist — just save new data
                df.to_csv(csv_file, index=False)
                print("✅ File created and records saved.")

    except Exception as e:
        print(f"Error for route {Source}-{Destination}: {e}")
    finally:
        driver.quit()