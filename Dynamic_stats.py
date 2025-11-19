def fetch_recent_fights(soup, max_fights=3):
    from Fetch_stats import fetch_page

    fights_table = soup.find("table", class_="b-fight-details__table")
    if not fights_table:
        return []

    rows = fights_table.find_all(
        "tr",
        class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
    )

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

            fighter_link = cols[0].find("a")
            if not fighter_link:
                continue
            fighter_name = fighter_link.get_text(strip=True)

            def parse_stat(stat_text, winner_idx=0):
                stat_text = stat_text.strip()
                if "%" in stat_text:
                    parts = [p.strip() for p in stat_text.split("%") if p]
                    try:
                        return float(parts[winner_idx])
                    except:
                        return 0.0
                elif " of " in stat_text:
                    try:
                        parts = stat_text.split(" of ")
                        landed = int(parts[winner_idx])
                        total = int(parts[1]) if len(parts) > 1 else 1
                        return round((landed / total) * 100, 1) if total != 0 else 0.0
                    except:
                        return 0.0
                else:
                    try:
                        return float(stat_text)
                    except:
                        return 0.0

            stats = {
                "Sig. str %": parse_stat(cols[1].get_text(strip=True)),
                "Head %": parse_stat(cols[2].get_text(strip=True)),
                "Body %": parse_stat(cols[3].get_text(strip=True)),
                "Leg %": parse_stat(cols[4].get_text(strip=True)),
                "Distance %": parse_stat(cols[5].get_text(strip=True)),
                "Clinch %": parse_stat(cols[6].get_text(strip=True)),
                "Ground %": parse_stat(cols[7].get_text(strip=True)),
            }

            fight_data[fighter_name] = stats
            break

        if len(fight_data) > 1:
            all_fights.append(fight_data)

    return all_fights