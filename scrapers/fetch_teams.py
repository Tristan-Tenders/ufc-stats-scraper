"""Scrape fighter team affiliations ("Trains at") from ufc.com athlete pages.

Cache: stats/fighter_teams.json  {fighter name: team or null}
Used by build_dataset.py for the camp-quality features (expanding team win rate).

  python -m scrapers.fetch_teams            # all fighters in the historical dataset
  python -m scrapers.fetch_teams --upcoming # just fighters on upcoming cards (fast)
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from scrapers.fetch_fighter_images import _get, candidate_slugs

CACHE = "stats/fighter_teams.json"
TEAM_RE = re.compile(r"Trains at\s*</div>\s*<div[^>]*>\s*([^<]+?)\s*<", re.S)


def fetch_team(name: str) -> str | None:
    for slug in candidate_slugs(name):
        r = _get(f"https://www.ufc.com/athlete/{slug}")
        if r is not None and r.status_code == 200:
            m = TEAM_RE.search(r.text)
            if m:
                return " ".join(m.group(1).split())
            return None  # page found, no team listed
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upcoming", action="store_true",
                    help="only fighters on upcoming cards")
    args = ap.parse_args()

    if args.upcoming:
        up = pd.read_csv("stats/upcoming.csv")
        names = sorted(set(up["F1"]) | set(up["F2"]))
    else:
        r = pd.read_csv("stats/raw/ufc_fight_results.csv")
        split = r["BOUT"].str.split(" vs. ", n=1, expand=True)
        names = sorted(set(split[0].str.strip()) | set(split[1].dropna().str.strip()))

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
    todo = [n for n in names if n not in cache]
    print(f"{len(names)} fighters, {len(todo)} to fetch ({len(cache)} cached)")

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_team, n): n for n in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), unit="fighter"):
            cache[futures[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                with open(CACHE, "w") as f:
                    json.dump(cache, f, indent=1)

    os.makedirs("stats", exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    found = sum(1 for v in cache.values() if v)
    print(f"saved {CACHE}: teams for {found}/{len(cache)} fighters")
    print("rebuild dataset to use them: python -m core.build_dataset")


if __name__ == "__main__":
    main()
