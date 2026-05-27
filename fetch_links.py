import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def fetch_page(url: str):
    for _ in range(3):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": random.choice(USER_AGENTS)},
                timeout=10,
            )
            if response.status_code == 200:
                return BeautifulSoup(response.text, "lxml")
            print(f"[!] {response.status_code} for {url}")
        except requests.RequestException as e:
            print(f"[!] {e}")
        time.sleep(random.uniform(1, 3))
    return None


def fetch_links_for_letter(letter: str) -> list[str]:
    url = f"http://ufcstats.com/statistics/fighters?char={letter}&page=all"
    soup = fetch_page(url)
    if not soup:
        return []
    links = parse_fighter_links(soup)
    print(f"[{letter}] {len(links)} fighters found")
    return links


def parse_fighter_links(soup) -> list[str]:
    table = soup.find("table", class_="b-statistics__table")
    if not table:
        return []
    return [a["href"] for a in table.find_all("a", href=True)]


def alphabet() -> list[str]:
    return [chr(char) for char in range(97, 123)]


def deduplicate(links: list[str]) -> list[str]:
    s = set()
    out = []
    for link in links:
        if link not in s:
            s.add(link)
            out.append(link)
    return out


def collect_all_links() -> list[str]:
    all_links = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_links_for_letter, letter): letter for letter in alphabet()}
        for future in as_completed(futures):
            all_links.extend(future.result())
    return deduplicate(all_links)


def save_links(links: list[str], path: str = "stats/fighter_links.json") -> None:
    os.makedirs("stats", exist_ok=True)
    with open(path, "w") as f:
        json.dump(links, f, indent=2)
    print(f"Saved {len(links)} links → {path}")


if __name__ == "__main__":
    links = collect_all_links()
    save_links(links)
