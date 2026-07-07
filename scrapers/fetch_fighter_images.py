"""Download fighter profile images -> site/img/ (shown on the prediction site).

Sources, in order:
  1. ufc.com athlete page og:image  (official headshot)
  2. Wikipedia page thumbnail       (fallback, only if the page is about a fighter)

Images are DOWNLOADED locally (not hotlinked) so they always render, even
offline. Cache: stats/fighter_images.json maps name -> "img/<file>" or null.
Re-runs only fetch fighters not already cached with a local file.

  python -m scrapers.fetch_fighter_images                     # everyone in upcoming.csv
  python -m scrapers.fetch_fighter_images "Max Holloway" ...  # specific fighters
"""

import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from scrapers.fetch_fights import build_headers

CACHE = "stats/fighter_images.json"
IMG_DIR = "site/img"
OG_IMAGE_RE = re.compile(r'property="og:image"\s+content="([^"]+)"')
FIGHTER_WORDS = ("martial", "fighter", "mma", "boxer", "kickboxer", "wrestler")


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'’]", "", s.lower().strip())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def candidate_slugs(name: str) -> list[str]:
    parts = name.split()
    slugs = [slugify(name)]
    if len(parts) > 2:
        slugs.append(slugify(f"{parts[0]} {parts[-1]}"))
    return slugs


def _get(url: str, **kw):
    try:
        time.sleep(random.uniform(0.2, 0.6))
        return requests.get(url, headers=build_headers(), timeout=15, **kw)
    except requests.RequestException:
        return None


def ufc_image_url(name: str) -> str | None:
    for slug in candidate_slugs(name):
        r = _get(f"https://www.ufc.com/athlete/{slug}")
        if r is not None and r.status_code == 200:
            m = OG_IMAGE_RE.search(r.text)
            if m and "/images/" in m.group(1):
                return m.group(1)
    return None


def wikipedia_image_url(name: str) -> str | None:
    for title in (name.replace(" ", "_"), f"{name.replace(' ', '_')}_(fighter)"):
        r = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
        if r is None or r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        blob = f"{data.get('description', '')} {data.get('extract', '')}".lower()
        if not any(w in blob for w in FIGHTER_WORDS):
            continue  # wrong person with the same name
        img = (data.get("originalimage") or data.get("thumbnail") or {}).get("source")
        if img:
            return img
    return None


def download(url: str, name: str) -> str | None:
    """Save image bytes locally; return path relative to site/ (e.g. img/x.png)."""
    r = _get(url)
    if r is None or r.status_code != 200 or not r.content:
        return None
    ctype = r.headers.get("Content-Type", "")
    ext = ".jpg" if "jpeg" in ctype else ".png" if "png" in ctype else \
        os.path.splitext(url.split("?")[0])[1] or ".jpg"
    fname = f"{slugify(name)}{ext}"
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(os.path.join(IMG_DIR, fname), "wb") as f:
        f.write(r.content)
    return f"img/{fname}"


def fetch_fighter_image(name: str) -> str | None:
    url = ufc_image_url(name) or wikipedia_image_url(name)
    return download(url, name) if url else None


def load_cache() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
        # migrate old hotlink entries / drop entries whose local file vanished
        return {k: v for k, v in cache.items()
                if v is None or (not str(v).startswith("http")
                                 and os.path.exists(os.path.join("site", v)))}
    return {}


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        if not os.path.exists("stats/upcoming.csv"):
            sys.exit("no stats/upcoming.csv — run fetch_upcoming.py first, "
                     "or pass fighter names as arguments")
        up = pd.read_csv("stats/upcoming.csv")
        names = sorted(set(up["F1"]) | set(up["F2"]))

    cache = load_cache()
    todo = [n for n in names if cache.get(n) is None]  # retry past misses too
    print(f"{len(names)} fighters, {len(todo)} to fetch ({len(names) - len(todo)} cached)")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_fighter_image, n): n for n in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), unit="fighter"):
            cache[futures[fut]] = fut.result()

    os.makedirs("stats", exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    found = sum(1 for v in cache.values() if v)
    print(f"saved {CACHE}: images for {found}/{len(cache)} fighters "
          f"(files in {IMG_DIR}/)")
    print("rebuild the page: python -m web.make_site")


if __name__ == "__main__":
    main()
