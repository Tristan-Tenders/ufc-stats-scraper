"""Prediction log — the project's scoreboard.

Every generated prediction is appended to stats/predictions_log.csv BEFORE the
fight happens. After results are scraped (python -m scrapers.scrape), grade them:

  python -m web.log_predictions grade    # fill in winners, print scoreboard
  python -m web.log_predictions show     # scoreboard + recent graded picks

make_site.py logs automatically. Metrics vs the market:
  accuracy   — % of graded fights where the model's pick won
  brier      — mean (prob - outcome)^2; lower is better, 0.25 = coin flip
Beating the market's brier over 100+ fights is the real test of the model.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from core.build_dataset import norm_name

LOG = "stats/predictions_log.csv"

COLS = ["logged_at", "event", "event_date", "f1", "f2", "p1", "market_p1",
        "method_dec", "method_ko", "method_sub",
        "winner", "correct", "brier", "market_brier"]


def _load() -> pd.DataFrame:
    if os.path.exists(LOG):
        df = pd.read_csv(LOG)
    else:
        df = pd.DataFrame({c: pd.Series(dtype="object" if c in
                          ("logged_at", "event", "event_date", "f1", "f2", "winner")
                          else "float64") for c in COLS})
    df["winner"] = df["winner"].astype("object").fillna("")
    return df


def log_card(results: list[dict]) -> int:
    """Append predictions (from make_site/predict result dicts) not yet logged."""
    log = _load()
    have = set(zip(log["event"].astype(str), log["f1"].astype(str),
                   log["f2"].astype(str)))
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        if (str(r["event"]), str(r["f1"]), str(r["f2"])) in have:
            continue
        m = r.get("method") or {}
        rows.append({
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "event": r["event"], "event_date": r["date"],
            "f1": r["f1"], "f2": r["f2"], "p1": round(r["p1"], 4),
            "market_p1": round(r["market"], 4) if r.get("market") is not None else np.nan,
            "method_dec": round(m.get("dec", np.nan), 3) if m else np.nan,
            "method_ko": round(m.get("ko", np.nan), 3) if m else np.nan,
            "method_sub": round(m.get("sub", np.nan), 3) if m else np.nan,
            "winner": "", "correct": np.nan, "brier": np.nan, "market_brier": np.nan,
        })
    if rows:
        add = pd.DataFrame(rows)
        log = add if log.empty else pd.concat([log, add], ignore_index=True)
        os.makedirs("stats", exist_ok=True)
        log.to_csv(LOG, index=False)
    print(f"prediction log: +{len(rows)} new, {len(log)} total -> {LOG}")
    return len(rows)


def grade() -> None:
    """Match ungraded predictions against scraped results and score them."""
    log = _load()
    if log.empty:
        print("log is empty — generate predictions first (make_site.py)")
        return

    res = pd.read_csv("stats/raw/ufc_fight_results.csv")
    res = res[res["OUTCOME"].isin(["W/L", "L/W"])]
    split = res["BOUT"].str.split(" vs. ", n=1, expand=True)
    w = np.where(res["OUTCOME"] == "W/L", split[0], split[1])
    winners = {}  # frozenset of both names -> winner key
    for f1, f2, win in zip(split[0], split[1], w):
        if pd.notna(f1) and pd.notna(f2):
            winners[frozenset((norm_name(f1), norm_name(f2)))] = norm_name(win)

    graded = 0
    for i, row in log.iterrows():
        if isinstance(row["winner"], str) and row["winner"]:
            continue
        k1, k2 = norm_name(row["f1"]), norm_name(row["f2"])
        win = winners.get(frozenset((k1, k2)))
        if win is None:
            continue
        won1 = float(win == k1)
        log.at[i, "winner"] = row["f1"] if won1 else row["f2"]
        log.at[i, "correct"] = float((row["p1"] >= 0.5) == won1)
        log.at[i, "brier"] = (row["p1"] - won1) ** 2
        if pd.notna(row["market_p1"]):
            log.at[i, "market_brier"] = (row["market_p1"] - won1) ** 2
        graded += 1

    log.to_csv(LOG, index=False)
    print(f"graded {graded} new results")
    show(log)


def show(log: pd.DataFrame | None = None) -> None:
    log = _load() if log is None else log
    done = log[log["correct"].notna()]
    pending = len(log) - len(done)
    print(f"\n=== scoreboard ===  ({len(done)} graded, {pending} pending)")
    if done.empty:
        return
    print(f"model accuracy: {done['correct'].mean():.1%}   "
          f"model brier: {done['brier'].mean():.4f}")
    mk = done[done["market_brier"].notna()]
    if len(mk):
        edge = mk["market_brier"].mean() - mk["brier"].mean()
        verdict = "model AHEAD of market" if edge > 0 else "market ahead of model"
        print(f"vs market ({len(mk)} fights): market brier "
              f"{mk['market_brier'].mean():.4f} — {verdict} by {abs(edge):.4f}")
    print("\nrecent graded picks:")
    for _, r in done.tail(8).iterrows():
        pick = r["f1"] if r["p1"] >= 0.5 else r["f2"]
        mark = "+" if r["correct"] else "x"
        print(f"  [{mark}] {r['f1']} vs {r['f2']} — picked {pick} "
              f"({max(r['p1'], 1 - r['p1']):.0%}), winner {r['winner']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    {"grade": grade, "show": show}.get(cmd, show)()
