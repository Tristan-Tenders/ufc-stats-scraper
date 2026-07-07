# UFC Stats Scraper + Fight Predictor

Predicts UFC fight outcomes from real historical fights, with leak-free
as-of-date features and temporal evaluation.

## Repo layout

```
pipeline.py        <- the one command: python pipeline.py
core/              dataset builder, strategy ensemble, predictor, utils
scrapers/          everything that fetches: ufcstats, upcoming cards,
                   images, teams, live market odds
web/               FastAPI app, static site generator, prediction log
analysis/          train benchmark, backtest, residual + round models
legacy/            the original scrapers and deprecated pipeline
stats/             data: raw scrapes, train_dataset.csv, logs, caches
site/              the generated website (index.html, app.html)
```

Standalone commands now run as modules from the repo root, e.g.
`python -m web.make_site --market` or `python -m analysis.backtest`.

## Quickstart — one command

```bash
pip install -r requirements.txt
python pipeline.py                # scrape -> data -> models -> website
                                  # (live Polymarket/Kalshi prices included)
python pipeline.py --skip-scrape  # reuse existing data (much faster)
python pipeline.py --no-market    # skip live market prices
```

Runs the whole thing end-to-end IN ONE PROCESS and finishes serving
http://localhost:8000. Every fight card shows a plain-English "why these
odds" breakdown — one sentence per angle the model considered (e.g.
"Striking volume, accuracy and defense -> Holloway 55%"), sorted by how much
the ensemble trusts each angle. No fighter photos — clean initials discs
instead (fetch_fighter_images.py still exists if you ever want them back).
The upcoming card is PRECOMPUTED and saved (stats/upcoming_predictions.json),
so the website loads it instantly and the predictions match the static site
and the prediction log exactly; the live API is used only for custom
matchups you type in. `--static` skips the server and opens site/index.html.

## Pipeline pieces (all runnable standalone)

1. **`fetch_fights.py`** — threaded, resumable scraper for everything the
   pipeline needs, straight from ufcstats.com: events, fight results,
   per-round fight stats, and (with `--fighters`) fighter tale-of-the-tape.
   Rotating User-Agent/Referer headers, per-thread sessions, exponential
   backoff on 403/429, checkpointed progress (safe to Ctrl-C and re-run).
   Parsers vendored from
   [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)
   (GPL-3.0) in `ufcstats_parsers.py` — same schemas, so the mirror repo's CSVs
   remain a drop-in alternative for `stats/raw/`. Odds still come from
   [ultimate_ufc_dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset)
   (`stats/raw/ufc_odds.csv`).

   ```bash
   python -m scrapers.scrape                # all phases: progress bars, live status, error log
   python -m scrapers.scrape --workers 4    # gentler rate
   ```

   `scrape.py` runs every phase with tqdm progress bars, a live
   "now scraping: ..." status line, immediate error notifications
   (also logged to `stats/raw/scrape_errors.log`), and a final data
   health check. Interrupt anytime — progress is checkpointed.

2. **`build_dataset.py`** — one row per real fight, label = actual winner.
   Optionally drop `ufc-master.csv` into `stats/raw/ufc_odds.csv` for odds features.
   Every feature (record, streak, SLpM, striking/TD accuracy & defense, control
   share, target/position mix, age, reach...) is computed only from fights
   strictly before the fight date. A/B orientation is deterministically
   randomized, and `diff_*` columns encode the matchup. Output: `stats/train_dataset.csv`.
3. **`train.py`** — temporal split (train on past, test on the most recent 15%),
   train-only median imputation, logistic regression / random forest /
   gradient boosting, reported as accuracy + ROC-AUC + log-loss against naive
   baselines.

## Current results (test = fights after 2024-03-30, strictly future)

Training uses mirror augmentation (every fight both orientations) and
time-decay sample weights (half-life 6y). "gbm monotonic" is gradient
boosting with domain monotonic constraints (e.g. P(win) can only increase
with Elo advantage) and native NaN handling.

| model | acc | auc | logloss |
|---|---|---|---|
| always pick side A | 0.519 | — | — |
| bookmaker favorite | 0.699 | — | — |
| logistic regression, stats only | 0.651 | 0.697 | 0.632 |
| gbm monotonic, stats only | 0.645 | 0.695 | 0.637 |
| gbm monotonic, stats + odds | 0.668 | 0.729 | 0.608 |
| **logistic regression, stats + odds** | **0.679** | **0.736** | **0.603** |

Method-of-victory model: 55.3% vs 53.0% majority baseline.
Ensemble probabilities are isotonic-calibrated. The bookmaker favorite is
still the strongest single signal; see ML_IDEAS.md for the deferred plan to
model market residuals directly.

## Predicting a fight or card

```bash
python -m core.predict "Islam Makhachev" "Charles Oliveira" --odds -350 280
python -m core.predict "Fighter A" "Fighter B" --market   # live Polymarket + Kalshi prices
python -m core.predict --card card.txt --market
```

`--market` pulls live probabilities from the Polymarket gamma API and the
Kalshi trade API (both public, no keys needed) via `fetch_market_odds.py`,
shows each source with volume, and computes model-vs-market edge and EV at
the actual contract prices. Standalone:
`python -m scrapers.fetch_market_odds "Fighter A" "Fighter B"`.

Weighs 8 independent strategies — form & momentum, activity & layoff,
physical & age, striking, grappling, matchup history (head-to-head + common
opponents), Elo, and a full ML model — with weights learned by out-of-fold
stacking (`strategies.py`). Prints each strategy's probability, the ensemble
probability and fair line, and when odds are given, the model-vs-market edge
and EV per side.

## Backtesting the market mismatch (`backtest.py`)

Bets flat stakes on the temporal test set whenever the (market-independent)
ensemble disagrees with the vig-free closing line, settled at real odds.
Result: **negative ROI at every threshold** (-10% to -22%) — when this model
disagrees with the closing line, the market is usually right. Treat flagged
mismatches as research leads, not bets.


## Prediction website

```bash
python -m scrapers.fetch_upcoming         # scrape scheduled fights from ufcstats.com
python -m web.make_site --market     # -> site/index.html
```

Generates a self-contained dark-mode page: every upcoming fight with win
probability bars, predicted winner, records, Elo, how-it-ends probabilities,
and (with `--market`) live prediction-market comparison. `--fast` skips the
out-of-fold stacking for quicker rebuilds.


## Live web app (API + frontend)

```bash
python -m web.app            # http://localhost:8000  (first start fits + caches models)
UFC_FAST=1 python -m web.app # quick dev startup — UNCALIBRATED, numbers run hot
```

Type any two fighters (autocomplete), get instant calibrated probabilities with
the full strategy breakdown, method prediction, and optional live market prices.
Second tab shows the upcoming card. Models cache to `stats/model_cache.pkl` and
refit automatically when the dataset changes.

## Calibration & the prediction log

The ensemble output is isotonic-calibrated on out-of-fold predictions, so
"70%" empirically wins ~70% of the time. Every site build logs its predictions
to `stats/predictions_log.csv` BEFORE the fights; after each event:

```bash
python -m scrapers.scrape && python -m web.log_predictions grade
```

prints the scoreboard — model accuracy and Brier score vs the market on every
fight you've predicted. This is the project's real test: beating the market's
Brier over 100+ graded fights.

## Research models

- `residual_model.py` — predicts where the MARKET is wrong (outcome minus
  implied probability). First model here to beat the closing line out-of-time,
  barely: logloss 0.5812 vs 0.5819. Small edges only; big disagreements lose.
- `round_model.py` — round-by-round survival: P(ends round r via KO/sub) +
  P(decision), calibrated (ends-in-R1 24.2% pred vs 23.2% actual).
- Ratings: classic Elo + margin-of-victory Elo + Glicko (with uncertainty RD).
- The stacker sees strategy disagreement + dominance forecasts, and flags
  "too close to call" picks instead of forcing a winner.


## Legacy scrapers

`fetch_links.py` + `Fetch_stats.py` scrape per-fighter profile pages
(`stats/fighter_stats.json`). `matchup_builder.py` is **deprecated** — it built
synthetic all-vs-all matchups whose labels leaked from the features.
