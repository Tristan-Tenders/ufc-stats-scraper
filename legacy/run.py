"""Full pipeline: scrape -> build dataset -> train.

Steps 1-2 hit ufcstats.com and can take a while; skip them if stats/raw/
is already populated (e.g. from the mirror repo:
git clone https://github.com/Greco1899/scrape_ufc_stats and copy the CSVs).
"""

import os
import subprocess
import sys


def main():
    if not os.path.exists("stats/raw/ufc_fight_results.csv"):
        print("scraping fight results (this hits ufcstats.com)...")
        subprocess.run([sys.executable, "fetch_fights.py"], check=True)
    else:
        print("stats/raw/ already populated - skipping scrape")

    print("\nbuilding dataset...")
    subprocess.run([sys.executable, "build_dataset.py"], check=True)

    print("\ntraining...")
    subprocess.run([sys.executable, "train.py"], check=True)


if __name__ == "__main__":
    main()
