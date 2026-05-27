from fetch_links import collect_all_links, save_links
from Fetch_stats import fetch_stats
from matchup_builder import build_all_matchups, load_fighters, save_matchups_csv


def main():
    print("fetching fighter links...")
    links = collect_all_links()
    save_links(links)

    print("\nscraping stats...")
    fetch_stats()

    print("\nbuilding matchups...")
    fighters = load_fighters()
    matchups = build_all_matchups(fighters)
    save_matchups_csv(matchups)

    print("\ndone.")


if __name__ == "__main__":
    main()
