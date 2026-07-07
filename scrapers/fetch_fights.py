"""Threaded, resumable UFCStats scraper — everything build_dataset.py needs.

Scrapes (mirror-compatible schemas, to stats/raw/):
  ufc_event_details.csv   events: name, url, date, location
  ufc_fight_results.csv   per fight: bout, outcome, weightclass, method, ...
  ufc_fight_stats.csv     per round & fighter: strikes, TDs, control, ...
  ufc_fighter_tott.csv    per fighter: height, weight, reach, stance, DOB

Anti-blocking:
  - rotating User-Agent + Referer headers per request
  - per-thread requests.Session (keep-alive), random delay between requests
  - exponential backoff with fresh headers on 403/429

Terminal feedback:
  - tqdm progress bars per phase
  - live "now scraping: ..." status (updates every few seconds)
  - errors printed immediately and appended to stats/raw/scrape_errors.log

Resumable: already-scraped events and fighters are skipped, progress is
checkpointed — safe to Ctrl-C and re-run. Run via scrape.py for all phases.
"""

import argparse
import hashlib
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from scrapers import ufcstats_parsers as LIB

RAW = "stats/raw"
ERROR_LOG = f"{RAW}/scrape_errors.log"
EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
CHALLENGE_POST_URL = "http://ufcstats.com/__c"
CHECKPOINT_EVERY = 10
DELAY_RANGE = (0.4, 1.2)  # seconds between requests, per thread
HEARTBEAT_SECS = 3

# ufcstats.com now gates every page with a SHA-256 proof-of-work JS challenge.
# The challenge HTML embeds a nonce and a difficulty (number of leading hex
# zeros); we brute-force n such that sha256(f"{nonce}:{n}") starts with that
# many zeros, POST the proof to /__c, and the server sets a session cookie
# (_fmc) that unlocks subsequent GETs on the same session.
CHALLENGE_NONCE_RE = re.compile(r'nonce="([0-9a-f]+)"')
CHALLENGE_TARGET_RE = re.compile(r"target=new Array\((\d+)\+1\)\.join\('0'\)")

RESULTS_COLS = ["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD", "ROUND",
                "TIME", "TIME FORMAT", "REFEREE", "DETAILS", "URL"]
TOTALS_COLS = ["ROUND", "FIGHTER", "KD", "SIG.STR.", "SIG.STR. %", "TOTAL STR.",
               "TD", "TD %", "SUB.ATT", "REV.", "CTRL"]
SIG_COLS = ["ROUND", "FIGHTER", "SIG.STR.", "SIG.STR. %", "HEAD", "BODY", "LEG",
            "DISTANCE", "CLINCH", "GROUND"]
TOTT_COLS = ["FIGHTER", "HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB", "URL"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]
REFERERS = ["https://www.google.com/", "https://www.bing.com/",
            "https://duckduckgo.com/", "http://ufcstats.com/"]

_tls = threading.local()
_save_lock = threading.Lock()


class Activity:
    """Thread-safe board of what each worker is doing + error collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current: dict[int, str] = {}
        self.errors = 0
        self.bar: tqdm | None = None

    def now(self, what: str) -> None:
        with self._lock:
            self._current[threading.get_ident()] = what

    def done(self) -> None:
        with self._lock:
            self._current.pop(threading.get_ident(), None)

    def snapshot(self) -> str:
        with self._lock:
            items = list(self._current.values())
        shown = " | ".join(i[:38] for i in items[:3])
        extra = f" (+{len(items) - 3})" if len(items) > 3 else ""
        return shown + extra

    def log(self, msg: str) -> None:
        out = tqdm.write if self.bar is not None else print
        out(msg)

    def error(self, msg: str) -> None:
        self.errors += 1
        self.log(f"[!] {msg}")
        with self._lock:
            os.makedirs(RAW, exist_ok=True)
            with open(ERROR_LOG, "a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")


ACTIVITY = Activity()


def _heartbeat(stop: threading.Event) -> None:
    """Every few seconds, show who/what is being scraped on the progress bar."""
    while not stop.wait(HEARTBEAT_SECS):
        if ACTIVITY.bar is not None:
            ACTIVITY.bar.set_postfix_str(
                f"errors={ACTIVITY.errors} | now: {ACTIVITY.snapshot()}")


def build_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def get_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        _tls.session = requests.Session()
    return _tls.session


def _solve_challenge(html: str, referer: str) -> bool:
    """If html is the PoW challenge page, solve it and POST proof.
    Returns True if a challenge was solved (caller should retry the GET)."""
    m_nonce = CHALLENGE_NONCE_RE.search(html)
    if not m_nonce:
        return False
    nonce = m_nonce.group(1)
    m_target = CHALLENGE_TARGET_RE.search(html)
    zeros = int(m_target.group(1)) if m_target else 2
    prefix = "0" * zeros
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(prefix):
        n += 1
    try:
        get_session().post(
            CHALLENGE_POST_URL,
            data={"nonce": nonce, "n": str(n)},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": referer,
                "User-Agent": get_session().headers.get("User-Agent")
                              or random.choice(USER_AGENTS),
            },
            timeout=15,
        )
    except requests.RequestException as e:
        ACTIVITY.log(f"[~] challenge POST failed: {e}")
        return False
    return True


def fetch_soup(url: str, tries: int = 5):
    """Throttled, retrying fetch with rotating headers + PoW challenge solver."""
    for attempt in range(tries):
        time.sleep(random.uniform(*DELAY_RANGE))
        try:
            r = get_session().get(url, headers=build_headers(), timeout=15)
            if r.status_code == 200:
                if _solve_challenge(r.text, url):
                    continue  # retry immediately with fresh cookie
                return BeautifulSoup(r.text, "lxml")
            if r.status_code in (403, 429):
                wait = 2 * (2 ** attempt) + random.uniform(1, 3)
                ACTIVITY.log(f"[~] {r.status_code} on {url} — backing off {wait:.0f}s")
                time.sleep(wait)
            else:
                ACTIVITY.log(f"[~] {r.status_code} for {url}")
        except requests.RequestException as e:
            ACTIVITY.log(f"[~] {e}")
            time.sleep(random.uniform(1, 3))
    ACTIVITY.error(f"gave up after {tries} tries: {url}")
    return None


# the vendored parsers call get_soup() internally — route through our fetcher
LIB.get_soup = fetch_soup


def load_csv(name: str, columns: list[str]) -> pd.DataFrame:
    path = f"{RAW}/{name}"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def save_csv(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RAW, exist_ok=True)
    df.to_csv(f"{RAW}/{name}", index=False)


# ---------- events + fights ----------

def scrape_event(event_url: str, event_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One event page -> (fight_results_df, fight_stats_df) for all its fights."""
    ACTIVITY.now(f"event: {event_name}")
    soup = fetch_soup(event_url)
    if soup is None:
        ACTIVITY.done()
        return pd.DataFrame(), pd.DataFrame()

    fight_urls = [tr["data-link"] for tr in soup.find_all(
        "tr", class_="b-fight-details__table-row b-fight-details__table-row__hover "
                     "js-fight-details-click") if tr.get("data-link")]

    results, stats = [], []
    for i, fu in enumerate(fight_urls, 1):
        ACTIVITY.now(f"{event_name} fight {i}/{len(fight_urls)}")
        fsoup = fetch_soup(fu)
        if fsoup is None:
            continue
        try:
            r_df, s_df = LIB.parse_organise_fight_results_and_stats(
                fsoup, fu, RESULTS_COLS, TOTALS_COLS, SIG_COLS)
            results.append(r_df)
            stats.append(s_df)
        except Exception as e:
            ACTIVITY.error(f"parse failed for {fu}: {e}")

    ACTIVITY.done()
    return (pd.concat(results, ignore_index=True) if results else pd.DataFrame(),
            pd.concat(stats, ignore_index=True) if stats else pd.DataFrame())


def scrape_events_and_fights(workers: int, fresh: bool) -> None:
    print("fetching event list...")
    soup = fetch_soup(EVENTS_URL)
    if soup is None:
        raise RuntimeError("could not load events listing")
    events = LIB.parse_event_details(soup)
    save_csv(events, "ufc_event_details.csv")

    results = load_csv("ufc_fight_results.csv", RESULTS_COLS)
    stats = load_csv("ufc_fight_stats.csv", [])
    done = set() if fresh else set(results["EVENT"].astype(str).str.strip())
    todo = events[~events["EVENT"].str.strip().isin(done)]
    print(f"{len(events)} events total — {len(todo)} to scrape, {len(done)} already done")
    if todo.empty:
        return

    def checkpoint():
        with _save_lock:
            save_csv(results, "ufc_fight_results.csv")
            save_csv(stats, "ufc_fight_stats.csv")

    stop = threading.Event()
    ACTIVITY.bar = tqdm(total=len(todo), unit="event", desc="events")
    threading.Thread(target=_heartbeat, args=(stop,), daemon=True).start()

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(scrape_event, row["URL"], row["EVENT"]): row["EVENT"]
                       for _, row in todo.iterrows()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    r_df, s_df = fut.result()
                except Exception as e:
                    ACTIVITY.error(f"event failed: {name}: {e}")
                    ACTIVITY.bar.update(1)
                    continue
                with _save_lock:
                    results = pd.concat([r_df, results], ignore_index=True)
                    stats = pd.concat([s_df, stats], ignore_index=True)
                completed += 1
                ACTIVITY.bar.update(1)
                if r_df.empty:
                    ACTIVITY.error(f"no fights parsed for event: {name}")
                if completed % CHECKPOINT_EVERY == 0:
                    checkpoint()
    finally:
        checkpoint()
        stop.set()
        ACTIVITY.bar.close()
        ACTIVITY.bar = None
    print(f"events done: {len(results)} fight results, {len(stats)} stat rows, "
          f"{ACTIVITY.errors} errors (see {ERROR_LOG})")


# ---------- fighter tale of the tape ----------

def collect_fighter_urls() -> list[str]:
    urls = []
    letters = LIB.generate_alphabetical_urls()
    for letter_url in tqdm(letters, desc="fighter listings", unit="page"):
        soup = fetch_soup(letter_url)
        if soup is None:
            continue
        table = soup.find("table", class_="b-statistics__table")
        if table:
            urls.extend(a["href"] for a in table.find_all("a", href=True))
    return list(dict.fromkeys(urls))


def scrape_fighter(url: str) -> pd.DataFrame:
    ACTIVITY.now(f"fighter: {url.rsplit('/', 1)[-1]}")
    soup = fetch_soup(url)
    if soup is None:
        ACTIVITY.done()
        return pd.DataFrame()
    try:
        name = soup.find("span", class_="b-content__title-highlight")
        if name is not None:
            ACTIVITY.now(f"fighter: {name.text.strip()}")
        tott = LIB.parse_fighter_tott(soup)
        return LIB.organise_fighter_tott(tott, TOTT_COLS, url)
    except Exception as e:
        ACTIVITY.error(f"tott parse failed for {url}: {e}")
        return pd.DataFrame()
    finally:
        ACTIVITY.done()


def scrape_fighters(workers: int, fresh: bool) -> None:
    print("\ncollecting fighter urls (a-z listings)...")
    urls = collect_fighter_urls()
    tott = load_csv("ufc_fighter_tott.csv", TOTT_COLS)
    done = set() if fresh else set(tott["URL"].astype(str))
    todo = [u for u in urls if u not in done]
    print(f"{len(urls)} fighters total — {len(todo)} to scrape, {len(done)} already done")
    if not todo:
        return

    stop = threading.Event()
    ACTIVITY.bar = tqdm(total=len(todo), unit="fighter", desc="fighters")
    threading.Thread(target=_heartbeat, args=(stop,), daemon=True).start()

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(scrape_fighter, u) for u in todo]
            for fut in as_completed(futures):
                df = fut.result()
                if not df.empty:
                    with _save_lock:
                        tott = pd.concat([df, tott], ignore_index=True)
                completed += 1
                ACTIVITY.bar.update(1)
                if completed % 50 == 0:
                    with _save_lock:
                        save_csv(tott, "ufc_fighter_tott.csv")
    finally:
        save_csv(tott.drop_duplicates("URL"), "ufc_fighter_tott.csv")
        stop.set()
        ACTIVITY.bar.close()
        ACTIVITY.bar = None
    print(f"fighters done: {len(tott)} rows, {ACTIVITY.errors} total errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fighters", action="store_true",
                    help="also scrape fighter tale-of-the-tape")
    ap.add_argument("--fighters-only", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore checkpoints, rescrape everything")
    args = ap.parse_args()

    if not args.fighters_only:
        scrape_events_and_fights(args.workers, args.fresh)
    if args.fighters or args.fighters_only:
        scrape_fighters(args.workers, args.fresh)
    print("\nall done. next: python -m core.build_dataset && python -m analysis.train")


if __name__ == "__main__":
    main()
