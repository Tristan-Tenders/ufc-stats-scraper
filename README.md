# UFC Stats Scraper

A simple Python scraper that grabs fighter profile links from UFCStats.com. It goes through the alphabet, collects all the fighter URLs, and saves them to a JSON file.

## What it does

- Goes through UFCStats.com fighter listings (A-Z)
- Collects all fighter profile links
- Saves everything to `fighter_links.json`
- Has several delays to not get blocked by the server

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python Stats.py
```

The script will show progress as it goes. It saves after each letter, so you won't lose everything if something goes wrong.

## What you'll get

A `fighter_links.json` file with all the fighter profile URLs, ready for whatever you want to do with them next.
