import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json
from datetime import datetime, date

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

def append_fighter(fighter_info, file_path="Stats/fighter_stats.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                fighters = json.load(f)
            except json.JSONDecodeError:
                fighters = []
    else:
        fighters = []
    fighters.append(fighter_info)
    with open(file_path, "w") as f:
        json.dump(fighters, f, indent=2)


links=[
  "http://ufcstats.com/fighter-details/12f91bfa8f1f723b",
  "http://ufcstats.com/fighter-details/9d62c2d8ee151f08",
  "http://ufcstats.com/fighter-details/5717efc6f271cd52",
  "http://ufcstats.com/fighter-details/6dbb1107de9a4a08",
  "http://ufcstats.com/fighter-details/dde70a112e053a6c",
  "http://ufcstats.com/fighter-details/e56daf7725a0b5ab"]
def fetch_stats():
    # with open("Stats/fighter_links.json","r") as f:
    #     links = json.load(f)
    for link in links:
        soup = fetch_page(link)
        if soup:
            print(f"\n--- Processing link: {link} ---")
            Name = soup.find("span", class_="b-content__title-highlight")
            Name = Name.text.strip()
            print(f"Name: {Name}")

            infos = soup.find_all("li", class_="b-list__box-list-item_type_block")

            for info in infos:

                label = info.find("i").text.strip()

                if label.startswith("Height"):

                    Height = info.find("i").next_sibling.strip()
                    for char in ["'", "/",'"']:
                        Height = Height.replace(char, "")
                    Height = Height.replace(" ","'")
                    print(f"Height: {Height}")

                if label.startswith("Weight"):
                    Weight = info.find("i").next_sibling.strip()
                    print(f"Weight: {Weight}")

                if label.startswith("Reach"):
                    Reach = info.find("i").next_sibling.strip()
                    print(f"Reach: {Reach}")

                if label.startswith("STANCE"):
                    Stance = info.find("i").next_sibling.strip()
                    print(f"Stance: {Stance}")
                
                if label.startswith("DOB"):
                    Age = info.find("i").next_sibling.strip()
                    birth_date = datetime.strptime(Age, "%b %d, %Y").date()
                    today = date.today()
                    age = today.year - birth_date.year
                    if (today.month, today.day) < (birth_date.month, birth_date.day):
                        age -= 1
                    print(f"Age: {Age}")

        fighter_info = {
            "Name": Name,
            "Height": Height,
            "Weight": Weight,
            "Reach": Reach,
            "Stance": Stance,
            "Age": Age,
        }
        append_fighter(fighter_info)
        

if __name__ == "__main__":
    fetch_stats()