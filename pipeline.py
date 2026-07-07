"""ONE command: scrape -> data -> models -> predictions -> website.

  python pipeline.py                # full run (live market prices included),
                                    # ends serving http://localhost:8000
  python pipeline.py --no-market    # skip Polymarket/Kalshi price fetching
  python pipeline.py --skip-scrape  # reuse existing stats/raw data
  python pipeline.py --fast         # quick model fit (uncalibrated, dev only)
  python pipeline.py --static       # no server: just open site/index.html
  python pipeline.py --no-open      # don't launch the browser

Everything runs in this one process (no subshells). Phases:
  1. scrape results + fighters from ufcstats.com
  2. scrape upcoming cards
  3. build the leak-free training dataset
  4. fit models, predict the card, log predictions,
     render site/index.html + stats/upcoming_predictions.json
  5. grade past predictions -> scoreboard
  6. serve the web app: the saved card loads instantly,
     the live API handles only custom matchups you type in
"""

import argparse
import os
import sys
import time
import webbrowser


def banner(text: str) -> None:
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def best_effort(label: str, fn) -> None:
    try:
        fn()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[~] {label} failed (continuing): {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-market", action="store_true",
                    help="skip live Polymarket/Kalshi prices (fetched by default)")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="skip phases 1-2 (reuse existing data)")
    ap.add_argument("--fast", action="store_true",
                    help="quick uncalibrated model fit (dev only)")
    ap.add_argument("--static", action="store_true",
                    help="open the static site instead of serving the app")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--teams", action="store_true",
                    help="also refresh team affiliations for the card")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    t0 = time.time()

    if not args.skip_scrape:
        banner("1/6  fight results + fighters (ufcstats.com)")
        from scrapers import fetch_fights
        fetch_fights.scrape_events_and_fights(args.workers, fresh=False)
        fetch_fights.scrape_fighters(args.workers, fresh=False)

        banner("2/6  upcoming cards")
        from scrapers import fetch_upcoming
        best_effort("fetch_upcoming", fetch_upcoming.main)
        if args.teams:
            import subprocess
            best_effort("teams", lambda: subprocess.run(
                [sys.executable, "-m", "scrapers.fetch_teams", "--upcoming"], check=True))
    else:
        print("skipping scrape phases (--skip-scrape)")

    banner("3/6  build training dataset")
    from core import build_dataset
    build_dataset.main()

    banner("4/6  fit models + predict the card + render site")
    from web import make_site
    make_site.run(market=not args.no_market, fast=args.fast)

    banner("5/6  grade past predictions")
    from web.log_predictions import grade
    best_effort("grading", grade)

    print(f"\npipeline done in {(time.time() - t0) / 60:.1f} min")

    if args.static:
        path = os.path.abspath("site/index.html")
        print(f"static site: {path}")
        if not args.no_open:
            webbrowser.open(f"file://{path}")
        return

    # find a free port (handles "address already in use" from an old server)
    import socket
    port = args.port
    for candidate in range(args.port, args.port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", candidate)) != 0:
                port = candidate
                break
    else:
        raise SystemExit(f"no free port near {args.port} — "
                         f"kill the old server: lsof -ti :{args.port} | xargs kill")
    if port != args.port:
        print(f"[~] port {args.port} is busy (old server still running?) — "
              f"using {port} instead. To free it: lsof -ti :{args.port} | xargs kill")

    banner(f"6/6  serving the web app (http://127.0.0.1:{port})")
    print("saved card -> instant | custom matchups -> live API")
    if not args.no_open:
        import threading

        def open_later():
            time.sleep(20)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=open_later, daemon=True).start()
    if args.fast:
        os.environ["UFC_FAST"] = "1"
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
