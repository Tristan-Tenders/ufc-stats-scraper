"""Live fight odds from prediction markets (Polymarket + Kalshi).

Both APIs are public and need no auth for market data:
  Polymarket gamma API:  https://gamma-api.polymarket.com/events?tag_slug=ufc
  Kalshi trade API:      https://api.elections.kalshi.com/trade-api/v2/markets
                         ?series_ticker=KXUFCFIGHT

Usage:
  python -m scrapers.fetch_market_odds "Alessandro Costa" "Cody Durden"

Or from code / predict.py:
  probs = get_market_probs("Fighter A", "Fighter B")
  # {"polymarket": (pa, pb, note), "kalshi": (pa, pb, note)}  (missing if not found)

Prices on prediction markets are direct probabilities (a 0.69 YES price
means the market prices that fighter at 69%). We use the bid/ask midpoint
where available and normalize each pair to sum to 1.
"""

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher

import requests

GAMMA_URL = "https://gamma-api.polymarket.com/events"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
TIMEOUT = 15
HEADERS = {"User-Agent": "ufc-stats-scraper/1.0", "Accept": "application/json"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


def _name_match(name: str, text: str) -> float:
    """Score how well a fighter name matches a market label/question."""
    name_n, text_n = _norm(name), _norm(text)
    if name_n and name_n in text_n:
        return 1.0
    last = name_n.split()[-1] if name_n else ""
    if last and re.search(rf"\b{re.escape(last)}\b", text_n):
        return 0.9
    return SequenceMatcher(None, name_n, text_n).ratio()


def _get(url: str, params: dict) -> dict | list | None:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        print(f"[~] {r.status_code} from {url}")
    except requests.RequestException as e:
        print(f"[~] {e}")
    return None


# ---------- Polymarket ----------

def polymarket_probs(fighter_a: str, fighter_b: str):
    """Search open UFC events; markets have outcomes = the two fighter names."""
    offset = 0
    while offset <= 300:
        events = _get(GAMMA_URL, {"tag_slug": "ufc", "closed": "false",
                                  "limit": 100, "offset": offset})
        if not events:
            break
        for ev in events:
            for m in ev.get("markets", []):
                try:
                    outcomes = json.loads(m.get("outcomes", "[]"))
                    prices = [float(p) for p in json.loads(m.get("outcomePrices", "[]"))]
                except (ValueError, TypeError):
                    continue
                if len(outcomes) != 2 or len(prices) != 2:
                    continue
                blob = f"{m.get('question', '')} {outcomes[0]} {outcomes[1]}"
                if (_name_match(fighter_a, blob) >= 0.9
                        and _name_match(fighter_b, blob) >= 0.9):
                    # orient prices to fighter_a
                    if _name_match(fighter_a, outcomes[0]) >= _name_match(fighter_a, outcomes[1]):
                        pa, pb = prices[0], prices[1]
                    else:
                        pa, pb = prices[1], prices[0]
                    total = pa + pb
                    if total <= 0:
                        continue
                    note = (f"{m.get('question', '')[:60]} "
                            f"(vol ${float(m.get('volumeNum', 0) or 0):,.0f})")
                    return pa / total, pb / total, note
        if len(events) < 100:
            break
        offset += 100
    return None


# ---------- Kalshi ----------

def _kalshi_mid(m: dict) -> float | None:
    try:
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        last = float(m.get("last_price_dollars") or 0)
        return last if last > 0 else None
    except (TypeError, ValueError):
        return None


def kalshi_probs(fighter_a: str, fighter_b: str):
    """KXUFCFIGHT series: one YES/NO market per fighter, grouped by event."""
    cursor, markets = None, []
    for _ in range(10):
        params = {"series_ticker": "KXUFCFIGHT", "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get(KALSHI_URL, params)
        if not data:
            break
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            break

    by_event: dict[str, list] = {}
    for m in markets:
        by_event.setdefault(m.get("event_ticker", ""), []).append(m)

    for ev_markets in by_event.values():
        best_a = best_b = None
        for m in ev_markets:
            label = m.get("yes_sub_title") or m.get("title", "")
            if _name_match(fighter_a, label) >= 0.9:
                best_a = m
            elif _name_match(fighter_b, label) >= 0.9:
                best_b = m
        if best_a is not None and best_b is not None:
            pa, pb = _kalshi_mid(best_a), _kalshi_mid(best_b)
            if pa and pb and (pa + pb) > 0:
                note = f"{best_a.get('event_ticker', '')} (vol {float(best_a.get('volume_fp', 0) or 0):,.0f})"
                return pa / (pa + pb), pb / (pa + pb), note
    return None


# ---------- combined ----------

def get_market_probs(fighter_a: str, fighter_b: str) -> dict:
    out = {}
    pm = polymarket_probs(fighter_a, fighter_b)
    if pm:
        out["polymarket"] = pm
    ks = kalshi_probs(fighter_a, fighter_b)
    if ks:
        out["kalshi"] = ks
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python -m scrapers.fetch_market_odds "Fighter A" "Fighter B"')
    a, b = sys.argv[1], sys.argv[2]
    probs = get_market_probs(a, b)
    if not probs:
        print("no open market found for this fight on Polymarket or Kalshi")
        return
    for source, (pa, pb, note) in probs.items():
        print(f"{source:12s} {a}: {pa:6.1%}   {b}: {pb:6.1%}   [{note}]")
    if len(probs) > 1:
        avg_a = sum(p[0] for p in probs.values()) / len(probs)
        print(f"{'average':12s} {a}: {avg_a:6.1%}   {b}: {1 - avg_a:6.1%}")


if __name__ == "__main__":
    main()
