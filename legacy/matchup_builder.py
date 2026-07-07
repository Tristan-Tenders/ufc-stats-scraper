# DEPRECATED: synthetic pairwise matchups with leaked labels.
# Use build_dataset.py instead (real fight outcomes, as-of-date features).

import csv
import json
import os
from itertools import combinations


def load_fighters(path: str = "stats/fighter_stats.json") -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def save_matchups_csv(matchups: list[dict], path: str = "stats/matchups.csv") -> None:
    os.makedirs("stats", exist_ok=True)
    if not matchups:
        print("No matchups to save.")
        return
    cols = get_column_order(matchups[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matchups)
    print(f"Saved {len(matchups)} matchup rows → {path}")


def get_column_order(row: dict) -> list[str]:
    cols = list(row.keys())
    return (
        ["label"]
        + [c for c in cols if c not in ("label", "a_name", "b_name")]
        + ["a_name", "b_name"]
    )


def average(values: list) -> float:
    if not values:
        return 0
    return sum(values) / len(values)


def flatten_fighter(fighter: dict) -> dict:
    career = fighter.get("CareerStats", {})
    recent = fighter.get("RecentFights", []) or []
    name = fighter["Name"]
    results = [fight["Result"] for fight in recent]
    sig_str = [fight[name]["Sig. str %"] for fight in recent if name in fight]
    head = [fight[name]["Head %"] for fight in recent if name in fight]
    body = [fight[name]["Body %"] for fight in recent if name in fight]
    leg = [fight[name]["Leg %"] for fight in recent if name in fight]
    distance = [fight[name]["Distance %"] for fight in recent if name in fight]
    clinch = [fight[name]["Clinch %"] for fight in recent if name in fight]
    ground = [fight[name]["Ground %"] for fight in recent if name in fight]

    return {
        "Name": name,
        "Height": fighter.get("Height", 0),
        "Weight": fighter.get("Weight", 0),
        "Reach": fighter.get("Reach", 0),
        "Stance": fighter.get("Stance", 0),
        "Age": fighter.get("Age", 0),
        "SLpM": career.get("SLpM", 0),
        "StrAcc": career.get("StrAcc", 0),
        "SApM": career.get("SApM", 0),
        "StrDef": career.get("StrDef", 0),
        "TDAvg": career.get("TDAvg", 0),
        "TDAcc": career.get("TDAcc", 0),
        "TDDef": career.get("TDDef", 0),
        "SubAvg": career.get("SubAvg", 0),
        "Recent_Win_Rate": average(results),
        "Recent_Wins": sum(results),
        "Recent_Losses": len(results) - sum(results),
        "Avg_Sig_Str": average(sig_str),
        "Avg_Head": average(head),
        "Avg_Body": average(body),
        "Avg_Leg": average(leg),
        "Avg_Distance": average(distance),
        "Avg_Clinch": average(clinch),
        "Avg_Ground": average(ground),
    }


def build_matchup_row(fighter_a: dict, fighter_b: dict, label: int) -> dict:
    row = {"label": label}
    for key, val in fighter_a.items():
        if key != "Name":
            row[f"a_{key}"] = val
    for key, val in fighter_b.items():
        if key != "Name":
            row[f"b_{key}"] = val
    row["a_name"] = fighter_a["Name"]
    row["b_name"] = fighter_b["Name"]
    return row


def build_all_matchups(fighters: list[dict]) -> list[dict]:
    flat = [flatten_fighter(f) for f in fighters]
    matchups = []
    for a, b in combinations(flat, 2):
        label = 1 if a["Recent_Win_Rate"] >= b["Recent_Win_Rate"] else 0
        matchups.append(build_matchup_row(a, b, label))
    return matchups


if __name__ == "__main__":
    fighters = load_fighters()
    matchups = build_all_matchups(fighters)
    save_matchups_csv(matchups)
