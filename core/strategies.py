"""Strategy ensemble for UFC fight prediction.

Each strategy is an independent "angle" on the fight (form, activity, physical,
striking, grappling, matchup history, Elo, full ML model). A stacked
logistic meta-model learns how much to trust each one from out-of-fold
predictions on historical fights.

IMPORTANT: no strategy uses betting odds — the ensemble probability is
market-independent, so comparing it to the market line is a meaningful
mismatch signal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ODDS_COLS = ["a_implied_prob", "diff_implied_prob"]

# domain priors the data may not violate: +1 = P(A wins) non-decreasing in
# this feature, -1 = non-increasing. Only features we're CONFIDENT about.
MONOTONIC = {
    "diff_elo": +1, "diff_mov_elo": +1, "diff_glicko": +1,
    "diff_win_rate": +1, "diff_implied_prob": +1,
    "diff_recent3_wr": +1, "diff_overperf": +1, "diff_recent3_qualwr": +1,
    "diff_str_def": +1, "diff_slpm": +1, "diff_td_def": +1,
    "diff_team_wr": +1, "diff_h2h_wins": +1, "common_edge": +1,
    "diff_sapm": -1, "diff_age": -1, "diff_ko_loss_share": -1,
}


def monotonic_vector(cols: list[str]) -> list[int]:
    return [MONOTONIC.get(c, 0) for c in cols]


# ---------- training tricks: mirror augmentation + time-decay weights ----------

def mirror_rows(X_df: pd.DataFrame, y: np.ndarray,
                w: np.ndarray | None = None):
    """Every fight twice, corners swapped and label flipped. Doubles data and
    forces the models to learn symmetric decision rules."""
    m = X_df.copy()
    cols = set(X_df.columns)
    swap = {}
    for c in X_df.columns:
        if c.startswith("a_") and ("b_" + c[2:]) in cols:
            swap[c] = "b_" + c[2:]
        elif c.startswith("b_") and ("a_" + c[2:]) in cols:
            swap[c] = "a_" + c[2:]
    m = m.rename(columns=swap)[X_df.columns]
    for c in X_df.columns:
        if c.startswith("diff_") or c == "common_edge":
            m[c] = -m[c]
    if "a_implied_prob" in m.columns:  # vig-free pair sums to 1
        m["a_implied_prob"] = 1 - X_df["a_implied_prob"]
    X_aug = pd.concat([X_df, m], ignore_index=True)
    y_aug = np.concatenate([y, 1 - y])
    w_aug = None if w is None else np.concatenate([w, w])
    return X_aug, y_aug, w_aug


def time_decay_weights(dates, half_life_years: float = 6.0) -> np.ndarray:
    """Recent fights matter more — the sport changes. weight = 0.5^(age/hl)."""
    if dates is None or half_life_years <= 0:
        return None
    age_years = (pd.Timestamp.now() - pd.to_datetime(dates)).dt.days / 365.25
    return np.asarray(0.5 ** (age_years / half_life_years))

# plain-English explanations shown on the website next to each strategy's
# probability — keep these short, concrete, and jargon-free
STRATEGY_EXPLAIN = {
    "form & momentum": "Recent results and win streaks",
    "activity & layoff": "How recently and how often they've fought (ring rust)",
    "physical & age": "Age, height and reach advantages",
    "striking": "Striking volume, accuracy and defense",
    "grappling": "Takedowns, control time and submission attempts",
    "matchup history": "Head-to-head history and shared past opponents",
    "opponent quality": "How strong the opposition they've faced has been",
    "durability & finish": "Finishing power versus ability to absorb damage",
    "camp quality": "How well fighters from their gym have been winning",
    "cardio & fade": "Whether they keep up their pace in later rounds",
    "momentum quality": "How good the opponents in their recent wins were",
    "damage mileage": "Career damage absorbed and recent knockout losses",
    "style matchup": "Each fighter's offense measured against the other's defense",
    "finish threat": "KO and submission power versus the other's record of being finished",
    "experience & pace": "Octagon experience and how long their fights usually last",
    "stance & prime": "Southpaw/orthodox matchup, prime-age window and ring rust",
    "cage & venue": "Small-cage (Apex) effects that help grapplers",
    "elo rating": "Overall skill rating built from every past result",
    "glicko rating": "Skill rating that trusts stale or thin records less",
    "gbm monotonic": "A machine-learning model with common-sense rules built in",
    "ml (all stats)": "The full machine-learning model using every statistic at once",
}

STRATEGY_FEATURES = {
    "form & momentum": ["diff_win_rate", "diff_recent3_wr", "diff_streak",
                        "a_recent3_wr", "b_recent3_wr"],
    "activity & layoff": ["diff_days_since_last", "a_days_since_last",
                          "b_days_since_last", "diff_n_prior"],
    "physical & age": ["diff_age", "diff_height_in", "diff_reach_in",
                       "a_age", "b_age"],
    "striking": ["diff_slpm", "diff_str_acc", "diff_sapm", "diff_str_def",
                 "diff_kd_avg15", "diff_rel_slpm", "diff_rel_sapm",
                 "diff_slpm_r3", "diff_sapm_r3"],
    "grappling": ["diff_td_avg15", "diff_td_acc", "diff_td_def",
                  "diff_sub_avg15", "diff_ctrl_share", "diff_td_avg15_r3"],
    "matchup history": ["diff_h2h_wins", "common_edge", "n_common"],
    "opponent quality": ["diff_opp_elo_avg", "diff_overperf",
                         "a_opp_elo_avg", "b_opp_elo_avg"],
    "durability & finish": ["diff_ko_win_share", "diff_sub_win_share",
                            "diff_ko_loss_share", "diff_sub_loss_share",
                            "diff_avg_fight_mins", "a_ko_loss_share",
                            "b_ko_loss_share"],
    "camp quality": ["diff_team_wr", "a_team_wr", "b_team_wr"],
    "cardio & fade": ["diff_fade_ratio", "a_fade_ratio", "b_fade_ratio",
                      "diff_avg_fight_mins"],
    "momentum quality": ["diff_recent3_qualwr", "diff_recent3_opp_elo",
                         "diff_recent3_overperf", "a_recent3_qualwr",
                         "b_recent3_qualwr"],
    "damage mileage": ["diff_dmg_absorbed", "diff_cage_mins",
                       "a_days_since_ko", "b_days_since_ko",
                       "diff_ko_recent", "a_ko_recent", "b_ko_recent"],
}


class FeatureStrategy:
    """Logistic regression on a small, interpretable feature subset."""

    def __init__(self, name: str, cols: list[str]):
        self.name = name
        self.cols = cols
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000, C=1.0))

    def _matrix(self, X_df):
        # median may itself be NaN if a fold's train slice has an all-NaN col
        return X_df[self.cols].fillna(self.medians).fillna(0.0).to_numpy(dtype=float)

    def fit(self, X_df, y, w=None):
        self.medians = X_df[self.cols].median(numeric_only=True)
        self.model.fit(self._matrix(X_df), y,
                       logisticregression__sample_weight=w)
        return self

    def predict_proba(self, X_df):
        return self.model.predict_proba(self._matrix(X_df))[:, 1]


class DerivedStrategy(FeatureStrategy):
    """Like FeatureStrategy, but on INTERACTION features computed on the fly
    (e.g. my KO power x your KO losses) instead of raw columns."""

    def __init__(self, name: str, builder):
        self.name = name
        self.builder = builder  # X_df -> DataFrame of derived features
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000, C=1.0))

    def _matrix(self, X_df):
        d = self.builder(X_df)
        return d.fillna(self.medians).fillna(0.0).to_numpy(dtype=float)

    def fit(self, X_df, y, w=None):
        self.medians = self.builder(X_df).median(numeric_only=True)
        self.model.fit(self._matrix(X_df), y,
                       logisticregression__sample_weight=w)
        return self


# ---------- derived-feature builders (offense vs the OPPONENT's defense) ----------

def build_style_matchup(X: pd.DataFrame) -> pd.DataFrame:
    """Striker-vs-grappler clash: my output rated against your defense."""
    d = pd.DataFrame(index=X.index)
    d["a_strike_out"] = X["a_slpm"] * (1 - X["b_str_def"])
    d["b_strike_out"] = X["b_slpm"] * (1 - X["a_str_def"])
    d["diff_strike_out"] = d["a_strike_out"] - d["b_strike_out"]
    d["a_grap_press"] = X["a_td_avg15"] * (1 - X["b_td_def"])
    d["b_grap_press"] = X["b_td_avg15"] * (1 - X["a_td_def"])
    d["diff_grap_press"] = d["a_grap_press"] - d["b_grap_press"]
    return d


def build_finish_threat(X: pd.DataFrame) -> pd.DataFrame:
    """My finishing power crossed with your record of being finished."""
    d = pd.DataFrame(index=X.index)
    d["a_ko_threat"] = X["a_ko_win_share"] * X["b_ko_loss_share"]
    d["b_ko_threat"] = X["b_ko_win_share"] * X["a_ko_loss_share"]
    d["diff_ko_threat"] = d["a_ko_threat"] - d["b_ko_threat"]
    d["a_sub_threat"] = X["a_sub_win_share"] * X["b_sub_loss_share"]
    d["b_sub_threat"] = X["b_sub_win_share"] * X["a_sub_loss_share"]
    d["diff_sub_threat"] = d["a_sub_threat"] - d["b_sub_threat"]
    return d


def build_experience_pace(X: pd.DataFrame) -> pd.DataFrame:
    """Octagon mileage and typical fight length."""
    d = pd.DataFrame(index=X.index)
    d["diff_n_prior"] = X["diff_n_prior"]
    d["a_n_prior"] = X["a_n_prior"]
    d["b_n_prior"] = X["b_n_prior"]
    d["diff_avg_fight_mins"] = X["diff_avg_fight_mins"]
    d["is_title"] = X.get("is_title", 0)
    return d


def build_stance_prime(X: pd.DataFrame) -> pd.DataFrame:
    """Southpaw-vs-orthodox edge, distance from prime age, age x layoff rust."""
    d = pd.DataFrame(index=X.index)
    d["southpaw_edge"] = (X["a_stance_southpaw"] * X["b_stance_orthodox"]
                          - X["b_stance_southpaw"] * X["a_stance_orthodox"])
    d["a_prime_dist"] = (X["a_age"] - 27.5).abs()
    d["b_prime_dist"] = (X["b_age"] - 27.5).abs()
    d["diff_prime_dist"] = d["a_prime_dist"] - d["b_prime_dist"]
    d["a_rust"] = X["a_age"] * X["a_days_since_last"] / 1e4
    d["b_rust"] = X["b_age"] * X["b_days_since_last"] / 1e4
    d["diff_rust"] = d["a_rust"] - d["b_rust"]
    return d


def build_cage_venue(X: pd.DataFrame) -> pd.DataFrame:
    """Small-cage (Apex) effects: grapplers benefit when there's less room to
    run, plus each fighter's own small-cage track record."""
    d = pd.DataFrame(index=X.index)
    apex = X.get("is_apex", pd.Series(0, index=X.index)).fillna(0)
    d["apex_grap_edge"] = apex * X["diff_td_avg15"]
    d["apex_ctrl_edge"] = apex * X["diff_ctrl_share"]
    d["apex_pace_edge"] = apex * X["diff_slpm"]
    d["diff_apex_wr"] = X["diff_apex_wr"]
    d["a_n_apex"] = X["a_n_apex"]
    d["b_n_apex"] = X["b_n_apex"]
    return d


DERIVED_STRATEGIES = {
    "style matchup": build_style_matchup,
    "finish threat": build_finish_threat,
    "experience & pace": build_experience_pace,
    "stance & prime": build_stance_prime,
    "cage & venue": build_cage_venue,
}


class GlickoStrategy:
    """Closed-form Glicko expectation — discounts stale/uncertain ratings."""
    name = "glicko rating"

    def fit(self, X_df, y, w=None):
        return self

    def predict_proba(self, X_df):
        import math
        q = math.log(10) / 400
        ra = X_df["a_glicko"].fillna(1500).to_numpy()
        rb = X_df["b_glicko"].fillna(1500).to_numpy()
        da = X_df["a_glicko_rd"].fillna(350).to_numpy()
        db = X_df["b_glicko_rd"].fillna(350).to_numpy()
        g = 1 / np.sqrt(1 + 3 * (q ** 2) * (da ** 2 + db ** 2) / (np.pi ** 2))
        return 1 / (1 + 10 ** (-g * (ra - rb) / 400))


class MonotonicGBMStrategy:
    """Gradient boosting with domain monotonic constraints and NATIVE NaN
    handling — no imputation, because missingness is itself informative
    (e.g. NaN fade_ratio means few long fights)."""
    name = "gbm monotonic"

    def __init__(self, feature_cols):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.cols = [c for c in feature_cols if c not in ODDS_COLS]
        self.model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.04, max_iter=600,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
            monotonic_cst=monotonic_vector(self.cols), random_state=42)

    def _matrix(self, X_df):
        return X_df[self.cols].to_numpy(dtype=float)  # NaN passes through

    def fit(self, X_df, y, w=None):
        self.model.fit(self._matrix(X_df), y, sample_weight=w)
        return self

    def predict_proba(self, X_df):
        return self.model.predict_proba(self._matrix(X_df))[:, 1]


class EloStrategy:
    """Closed-form Elo expectation — no fitting."""
    name = "elo rating"

    def fit(self, X_df, y, w=None):
        return self

    def predict_proba(self, X_df):
        d = (X_df["b_elo"] - X_df["a_elo"]).fillna(0).to_numpy(dtype=float)
        return 1.0 / (1.0 + 10 ** (d / 400.0))


class MLStatsStrategy:
    """Full-feature model (all stats, NO odds): LR + RF average."""
    name = "ml (all stats)"

    def __init__(self, feature_cols):
        self.cols = [c for c in feature_cols if c not in ODDS_COLS]
        self.lr = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.1))
        self.rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42)

    def _matrix(self, X_df):
        return X_df[self.cols].fillna(self.medians).fillna(0.0).to_numpy(dtype=float)

    def fit(self, X_df, y, w=None):
        self.medians = X_df[self.cols].median(numeric_only=True)
        X = self._matrix(X_df)
        self.lr.fit(X, y, logisticregression__sample_weight=w)
        self.rf.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X_df):
        X = self._matrix(X_df)
        return (self.lr.predict_proba(X)[:, 1] + self.rf.predict_proba(X)[:, 1]) / 2


class StrategyEnsemble:
    """Fits all strategies + a stacked meta-model that weighs them."""

    def __init__(self, feature_cols, meta: str = "logistic"):
        self.strategies = [FeatureStrategy(n, c) for n, c in STRATEGY_FEATURES.items()]
        self.strategies += [DerivedStrategy(n, b) for n, b in DERIVED_STRATEGIES.items()]
        self.strategies.append(EloStrategy())
        self.strategies.append(GlickoStrategy())
        self.strategies.append(MonotonicGBMStrategy(feature_cols))
        self.strategies.append(MLStatsStrategy(feature_cols))
        if meta == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            n = len(self.strategies)
            self.meta = HistGradientBoostingClassifier(
                max_depth=2, learning_rate=0.1, max_iter=150, random_state=42,
                monotonic_cst=[1] * n + [0] * 4)  # monotone in each strategy
        else:
            self.meta = LogisticRegression(max_iter=2000)
        self.aux_models = None
        self.unc_p80 = None

    def _aux_regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06,
                                             max_iter=250, random_state=42)

    def _meta_matrix(self, probs: np.ndarray, aux: np.ndarray | None) -> np.ndarray:
        """strategy probs + disagreement (std, spread) + aux forecasts."""
        std = probs.std(axis=1, ddof=0)[:, None]
        spread = (probs.max(axis=1) - probs.min(axis=1))[:, None]
        if aux is None:
            aux = np.zeros((len(probs), 2))
        return np.hstack([probs, std, spread, aux])

    def fit(self, X_df, y, n_folds: int = 5, dates=None,
            half_life_years: float = 6.0, mirror: bool = True,
            aux_targets: pd.DataFrame | None = None):
        w = time_decay_weights(dates, half_life_years)
        # out-of-fold strategy probabilities -> honest meta weights
        oof = np.full((len(X_df), len(self.strategies)), np.nan)
        n_aux = 0 if aux_targets is None else aux_targets.shape[1]
        oof_aux = np.zeros((len(X_df), max(n_aux, 2)))
        kf = KFold(n_splits=n_folds, shuffle=False)  # keeps chronological blocks
        for tr_idx, va_idx in kf.split(X_df):
            Xtr, ytr = X_df.iloc[tr_idx], y[tr_idx]
            wtr = None if w is None else w[tr_idx]
            Xtr_a, ytr_a, wtr_a = mirror_rows(Xtr, ytr, wtr) if mirror \
                else (Xtr, ytr, wtr)
            Xva = X_df.iloc[va_idx]
            for j, s in enumerate(self.strategies):
                s_clone = self._clone(s)
                s_clone.fit(Xtr_a, ytr_a, wtr_a)
                oof[va_idx, j] = s_clone.predict_proba(Xva)
            if n_aux:  # multi-task forecasts (strike diff, duration) as
                Xm = Xtr[[c for c in X_df.columns if c not in ODDS_COLS]]
                Xv = Xva[[c for c in X_df.columns if c not in ODDS_COLS]]
                for k in range(n_aux):
                    t = aux_targets.iloc[tr_idx, k]
                    ok = t.notna().to_numpy()
                    if ok.sum() < 100:
                        continue  # too sparse — leave zeros
                    r = self._aux_regressor()
                    r.fit(Xm[ok].to_numpy(dtype=float), t[ok])
                    oof_aux[va_idx, k] = r.predict(Xv.to_numpy(dtype=float))

        meta_X = self._meta_matrix(oof, oof_aux if n_aux else None)
        self.meta.fit(meta_X, y, sample_weight=w)
        self.unc_p80 = float(np.percentile(oof.std(axis=1, ddof=0), 80))

        # isotonic calibration on out-of-fold ENSEMBLE probabilities,
        # so an output of 0.70 empirically wins ~70% of the time
        oof_ens = cross_val_predict(LogisticRegression(max_iter=2000), meta_X,
                                    y, cv=5, method="predict_proba")[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_ens, y)

        # final fits on ALL data (mirrored + weighted)
        Xf, yf, wf = mirror_rows(X_df, y, w) if mirror else (X_df, y, w)
        for s in self.strategies:
            s.fit(Xf, yf, wf)
        self.aux_models = None
        if n_aux:
            self.aux_models = []
            Xm = X_df[[c for c in X_df.columns if c not in ODDS_COLS]]
            for k in range(n_aux):
                t = aux_targets.iloc[:, k]
                ok = t.notna().to_numpy()
                if ok.sum() < 100:
                    continue
                r = self._aux_regressor()
                r.fit(Xm[ok].to_numpy(dtype=float), t[ok])
                self.aux_models.append(r)
        return self

    def fit_fast(self, X_df, y, dates=None, half_life_years: float = 6.0,
                 mirror: bool = True, aux_targets=None):
        """In-sample fit, no OOF stacking or calibration (quick iterations)."""
        w = time_decay_weights(dates, half_life_years)
        Xf, yf, wf = mirror_rows(X_df, y, w) if mirror else (X_df, y, w)
        for s in self.strategies:
            s.fit(Xf, yf, wf)
        probs = self.strategy_probs(X_df).to_numpy()
        self.aux_models = None
        self.meta.fit(self._meta_matrix(probs, None), y, sample_weight=w)
        self.unc_p80 = float(np.percentile(probs.std(axis=1, ddof=0), 80))
        self.calibrator = None
        return self

    def _clone(self, s):
        if isinstance(s, DerivedStrategy):
            return DerivedStrategy(s.name, s.builder)
        if isinstance(s, FeatureStrategy):
            return FeatureStrategy(s.name, s.cols)
        if isinstance(s, EloStrategy):
            return EloStrategy()
        if isinstance(s, GlickoStrategy):
            return GlickoStrategy()
        if isinstance(s, MonotonicGBMStrategy):
            return MonotonicGBMStrategy(s.cols + ODDS_COLS)
        return MLStatsStrategy(s.cols + ODDS_COLS)

    def strategy_probs(self, X_df) -> pd.DataFrame:
        return pd.DataFrame(
            {s.name: s.predict_proba(X_df) for s in self.strategies},
            index=X_df.index)

    def _aux_predict(self, X_df) -> np.ndarray | None:
        if not self.aux_models:
            return None
        Xm = X_df[[c for c in X_df.columns if c not in ODDS_COLS]] \
            .to_numpy(dtype=float)
        return np.column_stack([m.predict(Xm) for m in self.aux_models])

    def predict_proba(self, X_df) -> np.ndarray:
        probs = self.strategy_probs(X_df).to_numpy()
        p = self.meta.predict_proba(
            self._meta_matrix(probs, self._aux_predict(X_df)))[:, 1]
        if getattr(self, "calibrator", None) is not None:
            p = self.calibrator.transform(p)
        return np.clip(p, 0.02, 0.98)  # never claim certainty

    def confidence(self, X_df, p: np.ndarray) -> np.ndarray:
        """False = too close to call: near-coinflip prob OR the strategies
        disagree more than they did on 80% of training fights."""
        std = self.strategy_probs(X_df).to_numpy().std(axis=1, ddof=0)
        low = (np.abs(np.asarray(p) - 0.5) < 0.04)
        if self.unc_p80 is not None:
            low |= std > self.unc_p80
        return ~low

    @property
    def weights(self) -> dict:
        if hasattr(self.meta, "coef_"):
            return {s.name: float(c) for s, c in
                    zip(self.strategies, self.meta.coef_[0])}
        return {s.name: 1.0 for s in self.strategies}  # hgb meta: no coefs
