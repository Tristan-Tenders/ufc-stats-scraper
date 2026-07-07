import random
import time


def parse_single_stat(text: str) -> float | None:
    """Parse ONE fighter's cell value into a 0-1 ratio.

    '61%'      -> 0.61
    '23 of 38' -> 0.605  (landed / attempted)
    '---'      -> None
    """
    text = text.strip()
    if not text or text in ("---", "--"):
        return None
    if "%" in text:
        try:
            return float(text.replace("%", "").strip()) / 100
        except ValueError:
            return None
    if " of " in text:
        try:
            landed, attempted = (int(p) for p in text.split(" of "))
            return round(landed / attempted, 3) if attempted else None
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_recent_fights(soup, max_fights=3):
    """Scrape a fighter's recent fights with per-fighter stats.

    Fix vs old version: each cell in the significant-strikes table contains
    TWO <p> tags (one per fighter). The old code called get_text() on the
    whole cell, concatenating both fighters' values into garbage like
    '12 of 340 of 21'. We now parse each <p> separately and store both
    fighters' stats under their own names.
    """
    from legacy.Fetch_stats import fetch_page

    fights_table = soup.find("table", class_="b-fight-details__table")
    if not fights_table:
        return []

    rows = fights_table.find_all(
        "tr",
        class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click",
    )

    if len(rows) < max_fights:
        return None

    stat_labels = [
        "Sig. str %", "Head %", "Body %", "Leg %",
        "Distance %", "Clinch %", "Ground %",
    ]

    all_fights = []
    for row in rows[:max_fights]:
        result_text = row.find_all("td")[0].get_text(strip=True).lower()
        result = 1 if result_text == "win" else 0

        fight_link = row.get("data-link")
        if not fight_link:
            continue

        fight_soup = fetch_page(fight_link)
        if not fight_soup:
            continue
        time.sleep(random.uniform(1.3, 2))

        sig_table = None
        for table in fight_soup.find_all("table"):
            prev_section = table.find_previous("section")
            if prev_section and "Significant Strikes" in prev_section.get_text():
                sig_table = table
                break
        if not sig_table:
            continue

        sig_rows = sig_table.find_all("tr")[1:]
        fight_data = {"Result": result}

        for sig_row in sig_rows:
            cols = sig_row.find_all("td")
            if len(cols) < 8:
                continue

            # each cell holds one <p> per fighter, in the same order as the names
            names = [p.get_text(strip=True) for p in cols[0].find_all("p")]
            if len(names) != 2:
                continue

            for idx, name in enumerate(names):
                stats = {}
                for col_i, label in enumerate(stat_labels, start=1):
                    ps = cols[col_i].find_all("p")
                    raw = ps[idx].get_text(strip=True) if len(ps) > idx else ""
                    stats[label] = parse_single_stat(raw)
                fight_data[name] = stats
            break  # first data row covers both fighters

        if len(fight_data) > 1:
            all_fights.append(fight_data)

    return all_fights
