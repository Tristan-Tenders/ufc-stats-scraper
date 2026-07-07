"""Scrape UPCOMING fight cards from ufcstats.com -> stats/upcoming.csv

Columns: EVENT, DATE, LOCATION, F1, F2, WEIGHTCLASS

Reuses fetch_fights' throttled fetcher (UA rotation, backoff, PoW solver).
Run:  python -m scrapers.fetch_upcoming
"""

import os

import pandas as pd

from scrapers.fetch_fights import fetch_soup

UPCOMING_URL = "http://ufcstats.com/statistics/events/upcoming?page=all"
OUT = "stats/upcoming.csv"

WEIGHT_WORDS = ("weight", "catch")  # matches e.g. "Lightweight", "Catch Weight"


def parse_upcoming_events(soup) -> list[dict]:
    events = []
    for row in soup.find_all("tr", class_="b-statistics__table-row"):
        link = row.find("a", class_="b-link")
        date_tag = row.find("span", class_="b-statistics__date")
        if not link or not date_tag:
            continue
        cells = row.find_all("td")
        location = cells[-1].get_text(strip=True) if cells else ""
        events.append({"EVENT": link.get_text(strip=True), "URL": link["href"],
                       "DATE": date_tag.get_text(strip=True), "LOCATION": location})
    return events


def parse_event_card(soup) -> list[dict]:
    """Fight rows on an upcoming event page: two fighter links per row."""
    fights = []
    for row in soup.find_all("tr", class_="b-fight-details__table-row"):
        names = [a.get_text(strip=True) for a in row.find_all("a", class_="b-link")]
        names = [n for n in names if n]
        if len(names) < 2:
            continue
        weightclass = ""
        for td in row.find_all("td"):
            text = td.get_text(strip=True)
            if text and any(w in text.lower() for w in WEIGHT_WORDS) and len(text) < 40:
                weightclass = text
                break
        fights.append({"F1": names[0], "F2": names[1], "WEIGHTCLASS": weightclass})
    return fights


def main():
    print("fetching upcoming events...")
    soup = fetch_soup(UPCOMING_URL)
    if soup is None:
        raise RuntimeError("could not load upcoming events listing")
    events = parse_upcoming_events(soup)
    print(f"{len(events)} upcoming events")

    rows = []
    for ev in events:
        print(f"  {ev['EVENT']} ({ev['DATE']})")
        esoup = fetch_soup(ev["URL"])
        if esoup is None:
            continue
        for fight in parse_event_card(esoup):
            rows.append({**{k: ev[k] for k in ("EVENT", "DATE", "LOCATION")}, **fight})

    os.makedirs("stats", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {len(rows)} scheduled fights -> {OUT}")
    print("next: python -m web.make_site  (add --market for live prices)")


if __name__ == "__main__":
    main()
