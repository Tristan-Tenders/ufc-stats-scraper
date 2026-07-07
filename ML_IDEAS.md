# ML_IDEAS — status

ALL IDEAS BELOW ARE NOW IMPLEMENTED (kept for reference/spec):

1. Market-residual model -> `residual_model.py`. RESULT: blend beats the raw
   market out-of-time (logloss 0.5812 vs 0.5819, acc 0.704 vs 0.696, n=747).
   Small-disagreement bets (|resid|>=0.02, n=270) showed +2.5% ROI; larger
   disagreements remain -ROI. Treat as promising-but-within-noise until the
   prediction log accumulates live evidence.
2. Glicko-1 + MOV-Elo -> `build_dataset.py::compute_elo` (single pass, returns
   (elo, mov_elo, glicko, glicko_rd) tuples). New features + closed-form
   GlickoStrategy. RD inflates with inactivity.
3. Uncertainty meta + abstention -> strategy-disagreement std/spread feed the
   meta-model; `StrategyEnsemble.confidence()` flags too-close-to-call
   (|p-0.5|<0.04 or std>train p80); surfaced in CLI/site/app.
   Venn-Abers still OPEN (optional upgrade over isotonic).
4. Multi-task aux targets -> sig_strike_diff + fight_secs META columns; OOF
   HGB-regressor forecasts appended to the meta-features.
5. Round/method survival -> `round_model.py`. RESULT: hazard logloss 0.581 vs
   0.613 prior; P(decision) calibrated within ~6pp per quintile; ends-in-R1
   24.2% predicted vs 23.2% actual. NOT yet wired into app/site (open task).
6. Meta-learner option -> `StrategyEnsemble(feature_cols, meta="hgb")`
   (monotone-in-strategies HGB). Default stays logistic; compare via backtest.

Still open: Venn-Abers intervals; app/site integration of round_model;
backtest comparison of meta="hgb" vs logistic.

---

## 1. Market-residual model (highest priority)

Reframe: instead of predicting the winner, predict where the MARKET is wrong.
Train on the ~80% of fights that have odds. Target options (try both):
(a) regression on `outcome - market_prob` (a_implied_prob, vig-free);
(b) classification of `outcome` using ONLY features orthogonalized against
the market (fit outcome ~ market first, model the residual with stats
features). Use GBM-mono without the odds columns as the residual learner.
Evaluation is NOT accuracy — it's: does (market_prob + predicted_residual)
beat market_prob on log-loss/Brier on the temporal test set, and does betting
only when |predicted_residual| > threshold produce non-negative ROI at real
odds (reuse backtest.py plumbing)? Acceptance: new `residual_model.py` with a
report section in backtest.py; honest conclusion documented even if negative.

## 2. Glicko-2 / margin-of-victory Elo

Replace/augment `build_dataset.py::compute_elo`. Two independent upgrades:
(a) Glicko-2: adds rating deviation (RD). Implement pure-python (no dep) or
use the `glicko2` pip package. New features: `glicko`, `glicko_rd` per side +
diffs; RD is an uncertainty feature the ensemble can learn to trust less.
(b) MOV-Elo: scale the K-factor update by dominance — finish before final
round → K*1.5; decision with |sig-strike diff| > 40 → K*1.25; split decision
→ K*0.75. Keep classic Elo columns for comparison; add new columns alongside.
Wire into: career history (pre-fight values), `predict.py::fighter_snapshot`,
EloStrategy (switch to Glicko expectation using both ratings + RDs).
Acceptance: leak test passes; temporal AUC of the elo-only strategy improves
vs classic Elo; top-10 current Glicko list is sane (Jones/Makhachev-tier).

## 3. Uncertainty-aware meta-model + abstention

`strategies.py::StrategyEnsemble`. Add to the meta-model's input: the
cross-strategy standard deviation and the min/max spread of the 20 strategy
probabilities (disagreement = uncertainty). Then add an `abstain_threshold`:
if calibrated p in [0.5-t, 0.5+t] OR strategy-stddev above its 80th
percentile, the prediction is flagged "no confident pick" (site/app show a
gray "too close to call" badge instead of a winner tag). Also consider
Venn-Abers calibration (pip `venn-abers`) instead of isotonic — gives
[p_lower, p_upper] intervals; display the interval in the app. Acceptance:
scoreboard tracks accuracy on non-abstained picks separately; abstained set
has measurably lower accuracy (proving the flag works).

## 4. Multi-task auxiliary targets

`build_dataset.py` already stores `method`. Add per-fight auxiliary targets
to the dataset as extra META columns: `sig_strike_diff` (a minus b, landed),
`fight_seconds`. Train a multi-output GBM (or three HGBs sharing nothing —
simpler) and feed their OOF predictions as three additional meta-features
into the stacking layer (predicted strike diff is effectively a learned
"dominance forecast"). Acceptance: stacked ensemble logloss improves on
temporal test; if it doesn't, document and revert.

## 5. Round/method survival model

Replace the 3-class method model with discrete-time hazard: for each round r
(1..5), P(fight ends in round r | reached round r), split by method
(ko/sub), using per-round features (fade curve, output rates). Produces
P(over/under X.5 rounds) and P(method & round) — the softest prop markets.
Training data: one row per (fight, round reached). Acceptance: new
`round_model.py`; predicted round distribution calibrated within 5pp per
bucket on temporal test; app shows "ends R1 18% / R2 22% / R3 25% / dec 35%".

## 6. Fancier stacking meta-learner

Swap logistic meta for a small HGB (depth 2, ~50 trees, monotonic in every
strategy input) OR add pairwise interaction features (elo x ml agreement).
Low expected gain; try only after items 1-3. Acceptance: OOF logloss of meta
improves vs logistic meta; if not, keep logistic.

## Rejected / not worth it (documented so nobody re-litigates)

- Deep learning (fighter embeddings, career-sequence GRU/transformers, GNNs
  over the opponent graph): ~6.4k fights / 17k fighter-fight rows is too
  small; these will memorize. Revisit only if the dataset grows 10x (e.g.
  adding all regional MMA — which is out of scope anyway).
- Bayesian hierarchical per-fighter models (PyMC): elegant, slow, and the
  partial-pooling benefit is already approximated by Elo + shrinkage.
- Label smoothing: tried informally elsewhere, does nothing here; calibration
  layer already handles overconfidence.
