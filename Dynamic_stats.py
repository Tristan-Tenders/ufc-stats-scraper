import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json

def fetch_recent_fights(soup, max_fights =3):
    from Fetch_stats import fetch_page 
    fights_table = soup.find("table", class_="b-fight-details__table")
    if not fights_table:
        return None
    
    rows = fights_table.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
    
    for row in rows[:max_fights]:
        cells = row.find_all("td")
        Result = cells[0].get_text(strip=True).lower()
        link = row.get("data-link")
        if not link:
            continue
        results = fetch_page(link)
        fight_data = results.fetch
        for link in row:
            fight_data={}
            if Result != "win": Result = int(0)
            else: Result = int(1)

            fight_data["Result"]=Result



    
    
    print(fight_data)
    return fight_data
