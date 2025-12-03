import os
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re
from datetime import datetime

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=options)

def fetch_data_from_source(Source, Destination, DepartureDate, cubin):

    driver = get_driver()
    flights = []
    try:
        # Get today's date
        # Print the list of dates

        com_str = Source + '-' + Destination + '-' + DepartureDate
        url = "https://www.flipkart.com/travel/flights/search?trips=" + com_str + "&travellers=1-0-0&class=" + cubin + "&tripType=ONE_WAY&isIntl=false"
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
        return df
    except Exception as e:
        print(f"Error for route {Source}-{Destination}: {e}")
    finally:
        driver.quit()