# TODO — ufc-stats-scraper

Detailed work list for polishing and extending this repo. Written to be handed
to an AI coding assistant: each task states the file(s), current behavior, the
exact change wanted, and acceptance criteria.

**Explicitly out of scope: pre-UFC / outside-UFC fighter records (Sherdog,
Tapology, etc.). Do not add scrapers or features based on non-UFC fight data.**

---

## Repo architecture (context for any model working on a task)

Pipeline: `scrape.py` (orchestrates `fetch_fights.py`) scrapes ufcstats.com into
`stats/raw/*.csv` → `build_dataset.py` computes leak-free as-of-fight-date
features into `stats/train_dataset.csv` (one row per real fight, label = actual
winner, deterministic A/B orientation, `a_*`/`b_*`/`diff_*` columns) →
`strategies.py` defines a 19-strategy stacked ensemble (per-angle logistic
models + Elo + full ML model, meta-weighted via out-of-fold stacking, isotonic
calibrated) → consumers: `train.py` (benchmark), `backtest.py` (market-mismatch
ROI), `predict.py` (CLI), `make_site.py` (static site `site/index.html`),
`app.py` (FastAPI + `site/app.html` live frontend), `log_predictions.py`
(prediction log + grading scoreboard).

Auxiliary scrapers: `fetch_upcoming.py` (scheduled cards →
`stats/upcoming.csv`), `fetch_fighter_images.py` (ufc.com/Wikipedia headshots →
`site/img/` + `stats/fighter_images.json`), `fetch_teams.py` ("Trains at" gym →
`stats/fighter_teams.json`), `fetch_market_odds.py` (live Polymarket/Kalshi
probabilities). `ufcstats_parsers.py` is vendored GPL-3 parsing code — do not
edit its parsing logic. Historical odds: `stats/raw/ufc_odds.csv` (manual
download of ufc-master.csv from github.com/shortlikeafox/ultimate_ufc_dataset).

Key invariants that must never break:
- Every training feature is computed ONLY from fights strictly before the
  fight's date (no leakage). `label` correlates with no single feature > ~0.35.
- `META_COLS` in `utils.py` lists all non-feature columns; feature code in
  build/predict/site/app must stay in sync with `FEATURES` in `build_dataset.py`.
- `predict.py::fighter_snapshot` must compute the SAME features as
  `build_dataset.py::build_career_history`, but as-of-today. Any new feature
  must be added in BOTH places (this is the most common source of bugs).

---

## P0 — bugs and correctness (do these first)

### P0.1 Model cache can silently serve uncalibrated fast-fit models
File: `app.py`. `Engine._save_cache()` writes `stats/model_cache.pkl` after
either a fast fit (`UFC_FAST=1`, in-sample, uncalibrated, badly overconfident)
or a full fit. `_load_cache()` cannot tell them apart, so a later
`python -m web.app` (full mode) will happily load a fast-fit cache and serve
overconfident probabilities labeled as calibrated=False but cached forever.
Fix: store `{"mode": "fast"|"full"}` inside the pickle; `_load_cache()` must
reject a cache whose mode doesn't match the requested mode. Acceptance: start
with `UFC_FAST=1`, then start without it → engine refits full and overwrites
cache; assert `engine.ens.calibrator is not None` in full mode.

### P0.2 Fight duration assumes 5-minute rounds for all eras
File: `build_dataset.py::fight_seconds`. Early UFC events had no rounds /
different formats; the `TIME FORMAT` column in `ufc_fight_results.csv` (e.g.
`"3 Rnd (5-5-5)"`, `"No Time Limit"`, `"1 Rnd + OT (12-3)"`) is ignored, so
`total_seconds` is wrong for pre-2001 fights, which distorts SLpM-style rates
for old fighters. Fix: parse `TIME FORMAT` to get per-round minutes; fall back
to 5-5-5. Acceptance: unit test with `("2", "1:30", "1 Rnd + OT (12-3)")` and a
few real old fights; modern fights unchanged.

### P0.3 Prediction-log grading can mis-grade rematches
File: `log_predictions.py::grade`. Results are matched by
`frozenset({fighter1, fighter2})` only — if two fighters fought twice (rematch
in the results CSV), the winner of EITHER fight can be assigned to the logged
prediction. Fix: build the winners map keyed by (frozenset(names), event_date)
and match on name-pair + closest event date within ±14 days of the logged
`event_date`. Acceptance: construct a synthetic results CSV with the same pair
winning once each way on two dates; grading picks the fight nearest the logged
date.

### P0.4 `is_apex` heuristic is unvalidated
File: `build_dataset.py::is_apex_event`. Current rule: Las Vegas + (Fight
Night/TUF, ≥2020-05) plus all Vegas events 2020-05→2021-07. Known misses:
"UFC on ESPN" Vegas cards at the Apex, Noche UFC 2023 (Apex), numbered events
held at the Apex during 2021; false positives possible for Fight Nights at
T-Mobile Arena. Fix: replace heuristic with an explicit allowlist of Apex
event names scraped or curated (a `stats/apex_events.txt` override file that
the heuristic consults first). Acceptance: spot-check 15 known Apex and 15
known non-Apex events, 100% correct.

### P0.5 Vendored parser edits + duplicate method-eval block in train.py
File: `train.py` — the method-of-victory eval block was patched in via string
replacement; verify it appears exactly once and imports are top-of-file (it
currently does `import pandas as pd` inside `main`). Clean up. Acceptance:
`python -m analysis.train` runs end-to-end with a single method-eval printout; flake8
clean.

---

## P1 — architecture and deduplication

### P1.1 Extract a shared prediction engine module
Files: `predict.py`, `make_site.py`, `app.py` all triplicate: model fitting,
`wc_means` computation, history/elo/teams loading, snapshot assembly, and
orientation-averaged prediction. `app.py::Engine` is already 90% of the right
abstraction. Task: create `engine.py` exposing `Engine` (fit/load-cache,
`predict(name_a, name_b, market=False, is_apex=0.0, is_title=0.0)`, `search`,
`fighter_card`, `predict_card(upcoming_df)`); rewrite `predict.py` (CLI thin
wrapper), `make_site.py` (render-only), and `app.py` (HTTP-only) on top of it.
Acceptance: all three entry points produce identical probabilities for the
same matchup; no duplicated `wc_means` code remains (grep).

### P1.2 Single source of truth for feature definitions
Files: `build_dataset.py::FEATURES` and `predict.py::fighter_snapshot`.
Snapshot re-implements ~40 feature formulas by hand. Task: refactor
`build_career_history` so the final per-fighter feature computation is a
function of a fighter's raw prior-fight rows (`compute_features(rows_df,
as_of_date, attrs, extras) -> dict`), used both when building the historical
dataset (vectorized path can remain for speed, but add an equivalence test)
and at predict time. Acceptance: a test picks 20 random (fighter, fight) pairs
and asserts snapshot-at-that-date equals the dataset row's `a_*` values within
1e-6.

### P1.3 Time-aware out-of-fold stacking
File: `strategies.py::StrategyEnsemble.fit`. OOF currently uses
`KFold(shuffle=False)` on a date-sorted frame, so fold 1's strategies train on
FUTURE fights to predict past ones. It's acceptable for weight estimation but
inconsistent with the project's no-future-data principle. Task: switch to
forward-chaining CV (e.g. 5 expanding-window splits; earliest block gets
weights from the first usable fold), keep isotonic calibration on the same OOF
predictions. Acceptance: backtest metrics reported before/after in the PR
description; no fold trains on data dated after its validation block.

### P1.4 Cache upcoming-card predictions in the API
File: `app.py::/api/upcoming`. Recomputes all ~13 predictions (plus optional
live market calls) on every request. Task: cache the response keyed by
(upcoming.csv mtime, market flag) with a 10-minute TTL for market prices.
Acceptance: second request returns in <50ms (log timing).

### P1.5 FastAPI lifespan + robust paths
File: `app.py`. `@app.on_event("startup")` is deprecated → use lifespan
context manager. `FileResponse("site/app.html")` and the `site/img` mount
assume cwd = repo root → resolve relative to `__file__`. Acceptance: `cd / &&
python /path/to/app.py` serves the frontend and images correctly; no
deprecation warnings.

---

## P2 — features, model quality, evaluation

### P2.1 Hyperparameter tuning pass (time-series CV)
Files: `strategies.py`, `train.py`. RF (500 trees, depth 8, sqrt features) and
HGB params were hand-picked before the feature count grew ~4x (now ~177
features). Task: small grid/Optuna search using forward-chaining CV on the
training window only (never the 15% temporal test set); tune RF
(n_estimators, max_depth, max_features, min_samples_leaf), HGB (depth, lr,
iterations, l2), and per-strategy logistic C. Store best params as constants
with a comment noting the search date. Acceptance: temporal-test AUC of
"stats only" improves or ties; params committed, search script included as
`tune.py`.

### P2.2 Feature audit and pruning
Files: `build_dataset.py`, new `feature_audit.py`. 177 features, many nearly
duplicate (career vs r3 vs rel variants). Task: script that reports, on the
temporal train split: permutation importance (RF + LR), pairwise |corr| > 0.95
clusters, and % NaN per feature; propose a drop list. Do NOT auto-drop —
output a report the maintainer reviews. Acceptance: `python feature_audit.py`
writes `stats/feature_audit.md` with the three tables.

### P2.3 Method-of-victory model: evaluate and calibrate properly
Files: `train.py`, `engine.py` (after P1.1). Current 3-class HGB gets 55.3% vs
53.0% majority. Task: (a) report per-class precision/recall + multiclass log
loss vs a class-prior baseline in train.py; (b) probability-calibrate it
(sklearn CalibratedClassifierCV, prefit, isotonic, on a temporal validation
slice); (c) add per-fighter method priors as features are already present —
try a small class-weighted variant. Acceptance: calibrated multiclass log loss
beats the prior baseline; site/app show calibrated method numbers.

### P2.4 Backtest upgrades: calibration curve, Kelly, CLV columns
File: `backtest.py`. Add: (a) reliability table (predicted-prob buckets vs
actual win rate, 10 bins) proving calibration; (b) fractional-Kelly staking
simulation alongside flat stakes (kelly fraction 0.25, cap 5% bankroll);
(c) report model log-loss vs market log-loss per year to show trend.
Acceptance: single `python -m analysis.backtest` prints all three sections; runtime
< 5 min on a laptop.

### P2.5 Round-level features: extend fade curve
File: `build_dataset.py::load_fight_totals`. Only R1/R3 sig strikes are
extracted. Task: also extract R1/R3 absorbed (opponent's round splits are
available by symmetric merge), and R5 output share for fighters with
championship rounds (`had_r5`, `r5_sig`); add features `absorb_fade_ratio`,
`champ_rounds_count`; add both to `fighter_snapshot` (see P1.2) and to the
"cardio & fade" strategy columns. Acceptance: dataset builds; coverage of
`a_absorb_fade_ratio` within 5pp of `a_fade_ratio`; equivalence test from
P1.2 still passes.

### P2.6 Missed-weight history (UFC data only)
New scraper allowed (ufcstats.com only — weigh-in results are NOT on
ufcstats; skip external sites per scope). Instead use what ufcstats has:
`WEIGHTCLASS` containing "Catch Weight" often indicates a miss. Add feature
`catchweight_fights_share` per fighter (prior fights at catchweight / prior
fights). Acceptance: feature present in dataset + snapshot; leak test passes.

### P2.7 Odds auto-refresh
New file `fetch_odds.py`: download
`https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/ufc-master.csv`
to `stats/raw/ufc_odds.csv` (simple requests.get, show bytes written, keep a
`.bak` of the previous file). Wire into `scrape.py` as phase 2.5 with a
try/except that warns but never fails the run. Acceptance: `python
fetch_odds.py` refreshes the file; `scrape.py` health check reports its row
count and max date.

---

## P3 — scrapers, site, and hygiene

### P3.1 Global rate limiter for scrapers
File: `fetch_fights.py`. Per-thread `DELAY_RANGE` means 6 workers ≈ 6 req/s
bursts. Add a shared token-bucket (e.g. `threading.Semaphore` released by a
timer, or simple global min-interval lock) targeting ≤ 3 req/s total,
configurable via `--rps`. Acceptance: log shows steady spacing; scrape of 5
events completes without 429s.

### P3.2 Preserve bout order / main-event flag from upcoming cards
Files: `fetch_upcoming.py`, `make_site.py`, `app.py`. Currently the first row
per event is assumed to be the main event. Verify ufcstats lists bouts
top-down as main→prelims (it does), then store an explicit `BOUT_ORDER`
column and `IS_MAIN` flag in `stats/upcoming.csv`; renderers use the flag
instead of positional assumption. Also pass `is_title` (detectable from
weightclass string containing "Title") and `is_apex` into predictions —
`make_site.py` already passes is_apex; ensure `app.py::/api/upcoming` does too.
Acceptance: title fights on an upcoming card get `is_title=1` in the feature
row (add a debug log line).

### P3.3 Image/team scraper hardening
Files: `fetch_fighter_images.py`, `fetch_teams.py`. (a) Same-name collisions:
ufc.com slugs may resolve to a different fighter (e.g. two "Bruno Silva"s —
slug gets `-1` suffix). Add a sanity check comparing the page's division/
record hints against our tott data when available; on mismatch try
`<slug>-1`. (b) Add `--force NAME` to re-fetch one fighter. (c) `fetch_teams`
should normalize team aliases ("American Top Team" vs "ATT" vs "American Top
Team (ATT)") via a small alias map. Acceptance: both Bruno Silvas resolve to
different images; alias map applied in `load_teams()`.

### P3.4 Static site: add strategy breakdown + calibration badge
File: `make_site.py`. The live app shows per-strategy probabilities; the
static site doesn't. Add a collapsible `<details>` per fight (same table as
app.html: strategy name, P(A), trust bar from `ens.weights`), plus a footer
badge "calibrated" / "fast fit — uncalibrated". Acceptance: renders correctly
with 19 rows; page stays < 150KB for a 14-fight card.

### P3.5 Scoreboard in the web app
Files: `app.py`, `site/app.html`, `log_predictions.py`. Add
`GET /api/scoreboard` returning the graded log as JSON (overall accuracy,
brier, market comparison, last 20 picks) and a third frontend tab rendering
it with a simple cumulative-accuracy line (inline SVG, no chart lib).
Acceptance: tab shows same numbers as `python -m web.log_predictions show`.

### P3.6 Repo cleanup
- Delete or move to `legacy/`: `matchup_builder.py` (deprecated, replaced by
  build_dataset), `random_forest.py`, `decision_tree.py` (custom tree impl,
  unused by pipeline), `Dynamic_stats.py`, `Fetch_stats.py`, `fetch_links.py`
  (legacy fighter-profile JSON pipeline), `stats/matchups.csv` (~1.8M-row
  obsolete synthetic dataset, large), `stats/fighters_flat.csv`,
  `stats/fighter_stats.json`, `stats/failed_links.json`.
- `.gitignore`: ensure `env/`, `__pycache__/`, `stats/raw/*.csv`,
  `stats/model_cache.pkl`, `site/img/` are handled deliberately (raw CSVs are
  ~10MB — decide commit vs ignore; recommend committing `ufc_odds.csv` only).
- Remove zero-byte leftover test images in `site/img/` if present.
- Pin versions in `requirements.txt` (currently unpinned).
Acceptance: fresh clone + `pip install -r requirements.txt` + `python
run.py` works; repo size reduced.

### P3.7 Test suite + CI
New `tests/` with pytest:
- Parser tests using 2-3 saved HTML fixture files (event page, fight page,
  fighter page) — snapshot the current parse output.
- Leak tests on a built dataset: max |corr(feature, label)| < 0.4; every
  `date` in the test split > every date in train split; label balance in
  [0.45, 0.55].
- P1.2 equivalence test (snapshot vs dataset row).
- `log_predictions` round-trip (log → grade → scoreboard) on synthetic data.
- `fetch_market_odds` parsing tests with the recorded Kalshi/Polymarket JSON
  payload shapes (already stubbed once in dev — turn into fixtures).
GitHub Actions: ruff + pytest on push (no network tests in CI).
Acceptance: `pytest -q` green locally; CI badge in README.

### P3.8 Data dictionary
New `docs/FEATURES.md`: table of every column in `train_dataset.csv` — name,
formula in words, source file, leak-safety note (e.g. "shifted cumulative,
excludes current fight"). Generate the skeleton programmatically from
`FEATURES` + META_COLS, then hand-edit descriptions. Acceptance: every one of
the ~184 columns has a row; linked from README.

### P3.9 Logging polish
Replace bare `print` in library-ish modules (`build_dataset.py`,
`strategies.py`, `engine.py`) with the `logging` module (INFO default,
`--quiet` respected by CLIs); keep prints in pure CLI scripts. Acceptance:
`python -c "import build_dataset"` produces no output; CLIs look unchanged.

---

## Explicitly deferred / rejected

- Pre-UFC records from Sherdog/Tapology — **rejected by owner**, do not build.
- Judge/scorecard scraping (mmadecisions.com) — deferred, revisit later.
- Short-notice replacement detection — no reliable free data source.
- Betting line-movement (open vs close) — needs a paid odds-history feed.
- Automated betting of any kind — out of scope permanently.

## Known model context (for anyone evaluating changes)

Temporal test (fights after 2024-03-30, n=965): stats-only RF ≈ 0.64 acc /
0.70 AUC; stats+odds RF ≈ 0.66-0.68 acc / 0.74 AUC; bookmaker favorite ≈ 0.70
acc; market log-loss ≈ 0.58 vs model ≈ 0.64. Backtesting bets against the
market loses 10-22% ROI at every disagreement threshold — the market wins
disagreements. Any "improvement" claim must cite these baselines on the same
temporal split, and anything above ~0.68 stats-only accuracy should be treated
as a leakage red flag and investigated before celebrating.
