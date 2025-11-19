import json
import pandas as pd

# 1️⃣ Load your JSON file with all fighters
with open("stats/fighter_stats.json", "r") as f:
    fighters = json.load(f)


flat_data = []

for fighter in fighters:
    flat_stats = {}


    recent_stats = fighter.get("RecentFights", [])
    if recent_stats:
        for stat in ["Sig. str %", "Head %", "Body %", "Leg %", "Distance %", "Clinch %", "Ground %"]:
   
            values = [f[fighter["Name"]].get(stat, 0) for f in recent_stats if fighter["Name"] in f]
            flat_stats[f"Avg_{stat}"] = sum(values) / len(values) if values else 0
    else:
        for stat in ["Sig. str %", "Head %", "Body %", "Leg %", "Distance %", "Clinch %", "Ground %"]:
            flat_stats[f"Avg_{stat}"] = 0


    flat_stats.update(fighter.get("CareerStats", {}))


    for key in ["Height", "Weight", "Reach", "Stance", "Age"]:
        flat_stats[key] = fighter.get(key, 0)


    flat_stats["Name"] = fighter.get("Name", "Unknown")

    flat_data.append(flat_stats)


df = pd.DataFrame(flat_data)
df.to_csv("stats/fighters_flat.csv", index=False)
print(df.head())
