import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from threading import Lock

from legacy.Dynamic_stats import fetch_recent_fights
from scrapers.fetch_fights import fetch_soup as fetch_page  # PoW-aware fetcher

_file_lock = Lock()
_error_lock = Lock()


def log_error(url: str, reason: str, path: str = "stats/failed_links.json") -> None:
    os.makedirs("stats", exist_ok=True)
    with _error_lock:
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    errors = json.load(f)
                except json.JSONDecodeError:
                    errors = []
        else:
            errors = []
        errors.append({"url": url, "reason": reason})
        with open(path, "w") as f:
            json.dump(errors, f, indent=2)


def append_fighter(fighter_info, file_path="stats/fighter_stats.json"):
    os.makedirs("stats", exist_ok=True)
    with _file_lock:
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


def height_to_inches(h):
    feet, inches = h.split()
    return int(feet) * 12 + int(inches)


def load_links(path: str = "stats/fighter_links.json") -> list:
    with open(path, "r") as file:
        return json.load(file)


def scrape_fighter(link: str) -> None:
    time.sleep(random.uniform(0.5, 1.5))
    soup = fetch_page(link)
    if not soup:
        return

    print(f"\n{link}")

    Name = soup.find("span", class_="b-content__title-highlight")
    Name = Name.text.strip()
    print(f"Name: {Name}")

    Height = Weight = Reach = Stance = age = None
    infos = soup.find_all("li", class_="b-list__box-list-item_type_block")
    skip_fighter = False

    for info in infos:
        label = info.find("i").text.strip()

        if label.startswith("Height"):
            raw = info.find("i").next_sibling.strip()
            if raw == "--":
                skip_fighter = True
                break
            for char in ["'", "/", '"']:
                raw = raw.replace(char, "")
            Height = height_to_inches(raw)
            print(f"Height: {Height}")

        elif label.startswith("Weight"):
            raw = info.find("i").next_sibling.strip()
            if raw == "--":
                skip_fighter = True
                break
            Weight = int(raw.replace(" lbs.", ""))
            print(f"Weight: {Weight}")

        elif label.startswith("Reach"):
            raw = info.find("i").next_sibling.strip()
            if raw == "--":
                skip_fighter = True
                break
            for char in ["/", '"']:
                raw = raw.replace(char, "")
            Reach = int(raw)
            print(f"Reach: {Reach}")

        elif label.startswith("STANCE"):
            raw = info.find("i").next_sibling.strip()
            if raw == "--":
                skip_fighter = True
                break
            Stance = 1 if raw == "Orthodox" else 0
            print(f"Stance: {Stance}")

        elif label.startswith("DOB"):
            raw = info.find("i").next_sibling.strip()
            try:
                birth_date = datetime.strptime(raw, "%b %d, %Y").date()
                today = date.today()
                age = today.year - birth_date.year
                if (today.month, today.day) < (birth_date.month, birth_date.day):
                    age -= 1
            except Exception:
                skip_fighter = True
                break
            print(f"Age: {age}")

    if skip_fighter:
        print(f"[~] {Name}: missing fields")

    Stats = {}
    cstats = soup.find_all(
        "li", class_="b-list__box-list-item b-list__box-list-item_type_block"
    )

    for stat_item in cstats:
        label_tag = stat_item.find("i")
        if not label_tag:
            continue
        label = label_tag.text.strip()
        value = label_tag.next_sibling.strip() if label_tag.next_sibling else None

        if not value or value == "--":
            continue

        if label.startswith("SLpM"):
            Stats["SLpM"] = float(value)
        elif label.startswith("Str. Acc."):
            Stats["StrAcc"] = float(value.strip("%")) / 100
        elif label.startswith("SApM"):
            Stats["SApM"] = float(value)
        elif label.startswith("Str. Def"):
            Stats["StrDef"] = float(value.strip("%")) / 100
        elif label.startswith("TD Avg."):
            Stats["TDAvg"] = float(value)
        elif label.startswith("TD Acc."):
            Stats["TDAcc"] = float(value.strip("%")) / 100
        elif label.startswith("TD Def."):
            Stats["TDDef"] = float(value.strip("%")) / 100
        elif label.startswith("Sub. Avg."):
            Stats["SubAvg"] = float(value)

    fetch_recent_fight = fetch_recent_fights(soup, max_fights=3)

    fighter_info = {
        "Name": Name,
        "Height": Height,
        "Weight": Weight,
        "Reach": Reach,
        "Stance": Stance,
        "Age": age,
        "CareerStats": Stats,
        "RecentFights": fetch_recent_fight,
    }

    append_fighter(fighter_info)
    if any(v is None for v in fighter_info.values()):
        print(f"[~] {Name}: saved with nulls")


def fetch_stats():
    links = load_links()
    print(f"scraping {len(links)} fighters...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scrape_fighter, link) for link in links]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 50 == 0:
                print(f"{i}/{len(links)}")


if __name__ == "__main__":
    fetch_stats()
