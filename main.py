import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json

letters = "abcdefghijklmnopqrstuvwxyz"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def build_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }

def fetch_page(url, tries=3, delay_range=(4, 12)):
    for _ in range(tries):
        try:
            headers = build_headers()
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                return BeautifulSoup(response.text, "lxml")
            else:
                print(f"[!] Error {response.status_code} for {url}")

        except requests.RequestException as e:
            print(f"[!] Connection error: {e}")

        sleep_time = random.uniform(*delay_range)
        print(f"Retrying in {sleep_time:.2f}s...")
        time.sleep(sleep_time)

    return None

fighter_links = set()
count = 1

def fetch():
    global count
    for letter in letters:
        BASE_URL = f"http://ufcstats.com/statistics/fighters?char={letter}&page=all"
        soup = fetch_page(BASE_URL)

        if soup:
            print(f"\n--- Processing letter: {letter.upper()} ---")
            rows = soup.find_all("tr", class_="b-statistics__table-row")

            for row in rows:
                last_name_link = row.find("a", class_="b-link b-link_style_black")
                if last_name_link:
                    print(f'Adding person number: {count}')
                    count += 1
                    fighter_links.add(last_name_link["href"])

                    
                    time.sleep(random.uniform(0.3, 1.2))
        else:
            print("[!] Failed to fetch the page.")

        print(f"Total collected so far: {len(fighter_links)}")
        print(f"Saving current files")
        save_json()

        
        long_sleep = random.uniform(7, 15)
        print(f"Sleeping {long_sleep:.2f}s before next letter...")
        time.sleep(long_sleep)


def save_json():
    os.makedirs("stats", exist_ok=True)
    with open("stats/fighter_links.json", "w", encoding="utf-8") as f:
        json.dump(list(fighter_links), f, indent=2)
    print(f"[+] Saved {len(fighter_links)} fighter links to fighter_links.json")

if __name__ == "__main__":
    print("Starting Scraper...")
    fetch()
