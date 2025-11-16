# UFC Stats Scraper

### `Create_links.py` ###
## What it does

- Goes through UFCStats.com fighter listings (A-Z)
- Collects all fighter profile links
- Saves everything to `fighter_links.json`


### `Fetch_stats.py` ###
## What it does

- Goes through all links in `fighter_links.json`
- Scrapes Fighter info like name,age,reach...
- Saves everything to `fighter_stats.json`



## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python Create_links.py
python Fetch_stats.py
```


## What you'll get

A `fighter_links.json` file with all the fighter profile URLs, ready for whatever you want to do with them next.
