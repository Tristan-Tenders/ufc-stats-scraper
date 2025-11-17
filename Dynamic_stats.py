import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json

def fetch_recent_fights(soup, max_fights =3):
    recent_fights=[]
    fights_table = soup.find("table", class_="b-fight-details__table")
    if not fights_table:
        return recent_fights
    
    rows = fights_table.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
    
    for row in rows[:max_fights]:
        cells = row.find_all("td")
        fight_data={}

    fight_data["Result"] = cells[0].get_text(strip=True).lower()
    recent_fights.append(fight_data)
    
    print(fight_data)
    return recent_fights