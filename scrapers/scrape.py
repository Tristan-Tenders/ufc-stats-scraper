"""Master scraping runner — one command to (re)collect all data.

  python -m scrapers.scrape                # events + fights + fighters (resumable)
  python -m scrapers.scrape --workers 4    # gentler on ufcstats.com
  python -m scrapers.scrape --legacy       # also run the old fighter-profile scrapers
  python -m scrapers.scrape --fresh        # ignore checkpoints, rescrape everything

Phases (each with its own progress bar, live "now scraping" status, and
error notifications; failures are logged to stats/raw/scrape_errors.log):
  1/3 events + fight results + per-round fight stats
  2/3 fighter tale-of-the-tape (height, reach, DOB, stance)
  3/3 summary + data health check
"""

import argparse
import sys
import time

import pandas as pd

from scrapers import fetch_fights
from scrapers.fetch_fights import ACTIVITY, ERROR_LOG, load_csv


def banner(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def health_check() -> None:
    banner("3/3  data health check")
    checks = [
        ("ufc_event_details.csv", ["EVENT", "DATE"]),
        ("ufc_fight_results.csv", ["EVENT", "BOUT", "OUTCOME"]),
        ("ufc_fight_stats.csv", ["EVENT", "BOUT", "ROUND", "FIGHTER"]),
        ("ufc_fighter_tott.csv", ["FIGHTER", "HEIGHT", "REACH", "DOB"]),
    ]
    ok = True
    for name, required in checks:
        df = load_csv(name, [])
        missing = [c for c in required if c not in df.columns]
        status = "OK " if len(df) and not missing else "BAD"
        if status == "BAD":
            ok = False
        print(f"  [{status}] {name:28s} {len(df):>7} rows"
              + (f"  missing cols: {missing}" if missing else ""))

    # cross-check: every result event should have stat rows
    results = load_csv("ufc_fight_results.csv", [])
    stats = load_csv("ufc_fight_stats.csv", [])
    if len(results) and len(stats):
        r_ev = set(results["EVENT"].astype(str).str.strip())
        s_ev = set(stats["EVENT"].astype(str).str.strip())
        gap = r_ev - s_ev
        if gap:
            ok = False
            print(f"  [WARN] {len(gap)} events have results but no fight stats "
                  f"(re-run to retry them)")

    if ACTIVITY.errors:
        print(f"\n  {ACTIVITY.errors} errors this run — details in {ERROR_LOG}")
    print("\nhealth check:", "all good" if ok else "issues found (see above)")
    if ok:
        print("next: python -m core.build_dataset && python -m analysis.train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="also run fetch_links.py + Fetch_stats.py (fighter profiles JSON)")
    args = ap.parse_args()

    t0 = time.time()

    banner("1/3  events + fight results + per-round stats")
    try:
        fetch_fights.scrape_events_and_fights(args.workers, args.fresh)
    except KeyboardInterrupt:
        print("\ninterrupted — progress is checkpointed, re-run to resume")
        sys.exit(1)

    banner("2/3  fighter tale-of-the-tape")
    try:
        fetch_fights.scrape_fighters(args.workers, args.fresh)
    except KeyboardInterrupt:
        print("\ninterrupted — progress is checkpointed, re-run to resume")
        sys.exit(1)

    if args.legacy:
        banner("extra: legacy fighter-profile scrapers")
        from legacy.fetch_links import collect_all_links, save_links
        from legacy.Fetch_stats import fetch_stats
        save_links(collect_all_links())
        fetch_stats()

    health_check()
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
