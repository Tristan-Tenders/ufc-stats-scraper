"""Build a leak-free training dataset from real UFC fight outcomes.

Replaces matchup_builder.py. Key properties:
  - one row per REAL fight, label = actual winner (not synthetic pairings)
  - every feature is computed from fights strictly BEFORE the fight date
  - one row per fight with deterministic random A/B orientation (no positional bias)
  - diff_* features (a minus b) alongside raw a_*/b_* stats
  - Elo ratings (pre-fight, K=32), weight-class-relative stats,
    and closing betting odds (implied win probability, vig removed)

Inputs (stats/raw/):
  ufc_event_details.csv, ufc_fight_results.csv, ufc_fight_stats.csv,
  ufc_fighter_tott.csv          (from fetch_fights.py or the mirror repo)
  ufc_odds.csv                  (optional; ufc-master.csv from
                                 github.com/shortlikeafox/ultimate_ufc_dataset)

Output: stats/train_dataset.csv
"""

import hashlib
import os
import re

import numpy as np
import pandas as pd

RAW = "stats/raw"
OUT = "stats/train_dataset.csv"
MIN_PRIOR_FIGHTS = 1  # both fighters need at least this many prior UFC fights

ELO_START = 1500.0
ELO_K = 32.0

WEIGHT_LBS = {
    "strawweight": 115, "flyweight": 125, "bantamweight": 135,
    "featherweight": 145, "lightweight": 155, "welterweight": 170,
    "middleweight": 185, "light heavyweight": 205, "heavyweight": 265,
}

# stats compared against the fighter's weight class (expanding, leak-free)
WC_REL_FEATURES = ["slpm", "sapm", "td_avg15", "height_in", "reach_in"]

# features whose MISSINGNESS is informative (never KO'd, few long fights,
# unknown team, no small-cage fights) -> explicit 0/1 "known" flag columns
KNOWN_FLAG_FEATURES = ["days_since_ko", "fade_ratio", "team_wr", "apex_wr"]


def add_known_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a_/b_<feat>_known indicator columns for informative missingness."""
    for feat in KNOWN_FLAG_FEATURES:
        for side in ("a", "b"):
            col = f"{side}_{feat}"
            if col in df.columns:
                df[f"{col}_known"] = df[col].notna().astype(float)
    return df


# ---------- parsing helpers ----------

def parse_of(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'23 of 38' -> (23, 38)"""
    extracted = series.astype(str).str.extract(r"(\d+)\s+of\s+(\d+)")
    return extracted[0].astype(float), extracted[1].astype(float)


def parse_ctrl(series: pd.Series) -> pd.Series:
    """'1:44' -> seconds"""
    def conv(v):
        if not isinstance(v, str) or ":" not in v:
            return np.nan
        m, s = v.split(":")
        try:
            return int(m) * 60 + int(s)
        except ValueError:
            return np.nan
    return series.map(conv)


def parse_height(v) -> float:
    """5' 11" -> inches"""
    if not isinstance(v, str):
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)", v)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def parse_reach(v) -> float:
    if not isinstance(v, str):
        return np.nan
    m = re.match(r"(\d+)", v.strip())
    return float(m.group(1)) if m else np.nan


def fight_seconds(round_num, time_str) -> float:
    try:
        m, s = str(time_str).split(":")
        return (int(round_num) - 1) * 300 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return np.nan


def norm_name(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def parse_weightclass(wc: str) -> tuple[float, float, float]:
    """-> (weight_lbs, is_women, is_title)"""
    if not isinstance(wc, str):
        return np.nan, np.nan, np.nan
    w = wc.lower()
    lbs = next((v for k, v in WEIGHT_LBS.items() if k in w), np.nan)
    return lbs, float("women" in w), float("title" in w)


def american_to_prob(odds: float) -> float:
    """American odds -> implied win probability (with vig)."""
    if pd.isna(odds):
        return np.nan
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


# ---------- load & assemble fight-level table ----------

def load_fight_totals() -> pd.DataFrame:
    """Per (fight, fighter) totals summed across rounds."""
    s = pd.read_csv(f"{RAW}/ufc_fight_stats.csv")
    s["sig_l"], s["sig_a"] = parse_of(s["SIG.STR."])
    s["tot_l"], s["tot_a"] = parse_of(s["TOTAL STR."])
    s["td_l"], s["td_a"] = parse_of(s["TD"])
    s["head_l"], _ = parse_of(s["HEAD"])
    s["body_l"], _ = parse_of(s["BODY"])
    s["leg_l"], _ = parse_of(s["LEG"])
    s["dist_l"], _ = parse_of(s["DISTANCE"])
    s["clinch_l"], _ = parse_of(s["CLINCH"])
    s["ground_l"], _ = parse_of(s["GROUND"])
    s["ctrl_s"] = parse_ctrl(s["CTRL"])
    s["kd"] = pd.to_numeric(s["KD"], errors="coerce")
    s["sub_att"] = pd.to_numeric(s["SUB.ATT"], errors="coerce")

    s["EVENT"] = s["EVENT"].str.strip()
    s["BOUT"] = s["BOUT"].str.strip()
    s["fighter_key"] = s["FIGHTER"].map(norm_name)

    cols = ["sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a", "head_l", "body_l",
            "leg_l", "dist_l", "clinch_l", "ground_l", "ctrl_s", "kd", "sub_att"]
    out = s.groupby(["EVENT", "BOUT", "fighter_key"], as_index=False)[cols].sum(min_count=1)

    # round-level output for the cardio/fade curve (R3 output vs R1 output)
    rounds = s["ROUND"].astype(str).str.strip()
    keys = ["EVENT", "BOUT", "fighter_key"]
    r1 = (s[rounds == "Round 1"][keys + ["sig_l"]]
          .drop_duplicates(keys).rename(columns={"sig_l": "r1_sig"}))
    r3 = (s[rounds == "Round 3"][keys + ["sig_l"]]
          .drop_duplicates(keys).rename(columns={"sig_l": "r3_sig"}))
    out = out.merge(r1, on=keys, how="left").merge(r3, on=keys, how="left")
    out["had_r3"] = out["r3_sig"].notna().astype(float)
    out["r3_sig"] = out["r3_sig"].fillna(0)
    out["r1_when3"] = out["r1_sig"].fillna(0) * out["had_r3"]
    return out


def is_apex_event(event: str, location: str, date) -> float:
    """Heuristic for small-cage UFC Apex events: Vegas Fight Nights since
    mid-2020, plus all Vegas events during the covid window."""
    loc = str(location).lower()
    if "las vegas" not in loc or pd.isna(date):
        return 0.0
    if pd.Timestamp("2020-05-01") <= date <= pd.Timestamp("2021-07-01"):
        return 1.0
    if date >= pd.Timestamp("2020-05-01") and (
            "fight night" in str(event).lower() or "tuf" in str(event).lower()):
        return 1.0
    return 0.0


def load_fights() -> pd.DataFrame:
    events = pd.read_csv(f"{RAW}/ufc_event_details.csv")
    events["EVENT"] = events["EVENT"].str.strip()
    events["date"] = pd.to_datetime(events["DATE"], format="%B %d, %Y", errors="coerce")
    events["is_apex"] = [is_apex_event(e, l, d) for e, l, d in
                         zip(events["EVENT"], events.get("LOCATION", ""), events["date"])]

    r = pd.read_csv(f"{RAW}/ufc_fight_results.csv")
    r["EVENT"] = r["EVENT"].str.strip()
    r["BOUT"] = r["BOUT"].str.strip()
    r = r[r["OUTCOME"].isin(["W/L", "L/W"])].copy()  # drop draws / no-contests

    split = r["BOUT"].str.split(" vs. ", n=1, expand=True)
    r["f1"], r["f2"] = split[0].str.strip(), split[1].str.strip()
    r["winner"] = np.where(r["OUTCOME"] == "W/L", r["f1"], r["f2"])
    r["total_seconds"] = [fight_seconds(rd, t) for rd, t in zip(r["ROUND"], r["TIME"])]

    r = r.merge(events[["EVENT", "date", "is_apex"]], on="EVENT", how="left")
    r = r.dropna(subset=["date", "f1", "f2"])
    r["fight_id"] = r["URL"].fillna(r["EVENT"] + "|" + r["BOUT"])
    return r


def load_fighter_attrs() -> pd.DataFrame:
    t = pd.read_csv(f"{RAW}/ufc_fighter_tott.csv")
    t["fighter_key"] = t["FIGHTER"].map(norm_name)
    t["height_in"] = t["HEIGHT"].map(parse_height)
    t["reach_in"] = t["REACH"].map(parse_reach)
    t["dob"] = pd.to_datetime(t["DOB"], format="%b %d, %Y", errors="coerce")
    t["stance_orthodox"] = (t["STANCE"].str.strip() == "Orthodox").astype(float)
    t["stance_southpaw"] = (t["STANCE"].str.strip() == "Southpaw").astype(float)
    t.loc[t["STANCE"].isna(), ["stance_orthodox", "stance_southpaw"]] = np.nan
    return t.drop_duplicates("fighter_key")[
        ["fighter_key", "height_in", "reach_in", "dob", "stance_orthodox", "stance_southpaw"]
    ]


def load_teams() -> dict:
    """fighter_key -> team name, from stats/fighter_teams.json (fetch_teams.py)."""
    path = "stats/fighter_teams.json"
    if not os.path.exists(path):
        print("no fighter_teams.json — team features will be NaN "
              "(run fetch_teams.py to enable)")
        return {}
    import json
    with open(path) as f:
        raw = json.load(f)
    return {norm_name(k): v for k, v in raw.items() if v}


def load_odds() -> dict:
    """(date, frozenset{name1, name2}) -> {name1: vig-free implied prob, name2: ...}"""
    path = f"{RAW}/ufc_odds.csv"
    if not os.path.exists(path):
        print("no ufc_odds.csv found - skipping odds features")
        return {}
    rename = {"RedFighter": "red", "BlueFighter": "blue", "Date": "date",
              "RedOdds": "r_odds", "BlueOdds": "b_odds",
              "R_fighter": "red", "B_fighter": "blue",
              "R_odds": "r_odds", "B_odds": "b_odds"}
    o = pd.read_csv(path, low_memory=False).rename(columns=rename)
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    o["r_odds"] = pd.to_numeric(o["r_odds"], errors="coerce")
    o["b_odds"] = pd.to_numeric(o["b_odds"], errors="coerce")
    o["rk"], o["bk"] = o["red"].map(norm_name), o["blue"].map(norm_name)
    o["rp"] = o["r_odds"].map(american_to_prob)
    o["bp"] = o["b_odds"].map(american_to_prob)

    book = {}
    for _, row in o.dropna(subset=["date", "rp", "bp"]).iterrows():
        total = row["rp"] + row["bp"]  # remove vig
        book[(row["date"], frozenset((row["rk"], row["bk"])))] = {
            row["rk"]: (row["rp"] / total, row["r_odds"]),
            row["bk"]: (row["bp"] / total, row["b_odds"])}
    print(f"odds loaded: {len(book)} fights")
    return book


# ---------- head-to-head & common opponents (chronological, leak-free) ----------

def compute_matchup_history(fights: pd.DataFrame) -> dict:
    """fight_id -> prior h2h record and common-opponent edge (f1's perspective)."""
    from collections import defaultdict
    past = defaultdict(lambda: defaultdict(list))  # past[k][opp] = [1/0 results]
    out = {}
    for _, f in fights.sort_values("date").iterrows():
        k1, k2 = norm_name(f["f1"]), norm_name(f["f2"])
        common = (set(past[k1]) & set(past[k2])) - {k1, k2}
        edges = [np.mean(past[k1][o]) - np.mean(past[k2][o]) for o in common]
        out[f["fight_id"]] = {
            "h2h_wins_f1": float(sum(past[k1][k2])),
            "h2h_wins_f2": float(sum(past[k2][k1])),
            "n_common": float(len(common)),
            "common_edge_f1": float(np.mean(edges)) if edges else np.nan,
        }
        s1 = 1.0 if norm_name(f["winner"]) == k1 else 0.0
        past[k1][k2].append(s1)
        past[k2][k1].append(1 - s1)
    return out


# ---------- Elo (pre-fight, chronological) ----------

def compute_elo(fights: pd.DataFrame) -> tuple[dict, dict]:
    """Three rating systems in one chronological pass. Returns (pre, now):
    pre[(fight_id, fighter_key)] = (elo, mov_elo, glicko, glicko_rd) BEFORE
    that fight; now[fighter_key] = same tuple after all fights to date.

    elo       classic K=32
    mov_elo   margin-of-victory K: finishes move ratings 1.4x, split/majority
              decisions only 0.75x — dominance counts
    glicko    Glicko-1: rating + RD (uncertainty). RD inflates with
              inactivity and shrinks with fights — a stale rating is
              explicitly less trustworthy.
    """
    import math
    Q = math.log(10) / 400
    RD_START, RD_MIN, RD_MAX, RD_C = 350.0, 30.0, 350.0, 70.0

    elo, mov, gl, rd, last = {}, {}, {}, {}, {}
    pre = {}

    def gfun(x):
        return 1 / math.sqrt(1 + 3 * (Q ** 2) * (x ** 2) / (math.pi ** 2))

    for _, f in fights.sort_values("date").iterrows():
        k1, k2 = norm_name(f["f1"]), norm_name(f["f2"])
        date = f["date"]
        for k in (k1, k2):  # inactivity inflates uncertainty
            if k in rd and k in last and pd.notna(date):
                years = max((date - last[k]).days, 0) / 365.25
                rd[k] = min(math.sqrt(rd[k] ** 2 + (RD_C ** 2) * years), RD_MAX)

        r1, r2 = elo.get(k1, ELO_START), elo.get(k2, ELO_START)
        m1, m2 = mov.get(k1, ELO_START), mov.get(k2, ELO_START)
        g1, g2 = gl.get(k1, ELO_START), gl.get(k2, ELO_START)
        d1, d2 = rd.get(k1, RD_START), rd.get(k2, RD_START)
        pre[(f["fight_id"], k1)] = (r1, m1, g1, d1)
        pre[(f["fight_id"], k2)] = (r2, m2, g2, d2)

        s1 = 1.0 if norm_name(f["winner"]) == k1 else 0.0

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        elo[k1] = r1 + ELO_K * (s1 - exp1)
        elo[k2] = r2 + ELO_K * ((1 - s1) - (1 - exp1))

        meth = str(f.get("METHOD", "")).lower()
        mult = 1.4 if ("ko" in meth or "submission" in meth) else \
            0.75 if ("split" in meth or "majority" in meth) else 1.0
        em1 = 1 / (1 + 10 ** ((m2 - m1) / 400))
        mov[k1] = m1 + ELO_K * mult * (s1 - em1)
        mov[k2] = m2 + ELO_K * mult * ((1 - s1) - (1 - em1))

        for ka, ra, rb, da, db, sa in ((k1, g1, g2, d1, d2, s1),
                                       (k2, g2, g1, d2, d1, 1 - s1)):
            g_ = gfun(db)
            e = 1 / (1 + 10 ** (-g_ * (ra - rb) / 400))
            dsq = 1 / ((Q ** 2) * (g_ ** 2) * e * (1 - e))
            gl[ka] = ra + Q / (1 / da ** 2 + 1 / dsq) * g_ * (sa - e)
            rd[ka] = max(math.sqrt(1 / (1 / da ** 2 + 1 / dsq)), RD_MIN)

        if pd.notna(date):
            last[k1] = last[k2] = date

    now = {k: (elo[k], mov.get(k, ELO_START), gl.get(k, ELO_START),
               rd.get(k, RD_START)) for k in elo}
    return pre, now


# ---------- as-of-date career features ----------

def parse_method(method: str) -> str:
    m = str(method).lower()
    if "ko" in m:          # KO/TKO
        return "ko"
    if "submission" in m:
        return "sub"
    return "dec"


def build_career_history(fights: pd.DataFrame, totals: pd.DataFrame,
                         elo_pre: dict | None = None,
                         teams: dict | None = None) -> pd.DataFrame:
    """One row per (fighter, fight) with cumulative stats over strictly PRIOR fights."""
    teams = teams or {}

    def one_side(me_col: str, opp_col: str) -> pd.DataFrame:
        method = (fights["METHOD"].map(parse_method) if "METHOD" in fights
                  else pd.Series("dec", index=fights.index))
        me = fights[me_col].map(norm_name)
        d = pd.DataFrame({
            "fight_id": fights["fight_id"],
            "date": fights["date"],
            "fighter_key": me,
            "opp_key": fights[opp_col].map(norm_name),
            "won": (fights["winner"].map(norm_name) == me).astype(float),
            "seconds": fights["total_seconds"],
            "EVENT": fights["EVENT"],
            "BOUT": fights["BOUT"],
            "is_apex_fight": fights["is_apex"].astype(float)
                if "is_apex" in fights else 0.0,
            "team": me.map(lambda k: teams.get(k)),
        })
        d["ko_win"] = d["won"] * (method == "ko")
        d["sub_win"] = d["won"] * (method == "sub")
        d["ko_loss"] = (1 - d["won"]) * (method == "ko")
        d["sub_loss"] = (1 - d["won"]) * (method == "sub")
        return d

    h = pd.concat([one_side("f1", "f2"), one_side("f2", "f1")], ignore_index=True)

    # camp quality: expanding win rate of the fighter's TEAM before this fight
    # (chronological, shifted -> leak-free; NaN when team unknown)
    h = h.sort_values("date", kind="stable").reset_index(drop=True)
    h["team_wr"] = (h.groupby("team", dropna=True)["won"]
                    .transform(lambda x: x.shift(1).expanding(min_periods=3).mean()))

    DEFAULT_R = (ELO_START, ELO_START, ELO_START, 350.0)
    if elo_pre:
        own = [elo_pre.get((fid, k), DEFAULT_R)
               for fid, k in zip(h["fight_id"], h["fighter_key"])]
        opp = [elo_pre.get((fid, k), DEFAULT_R)
               for fid, k in zip(h["fight_id"], h["opp_key"])]
        h["elo_pre"] = [v[0] for v in own]
        h["mov_elo"] = [v[1] for v in own]
        h["glicko"] = [v[2] for v in own]
        h["glicko_rd"] = [v[3] for v in own]
        h["opp_elo_pre"] = [v[0] for v in opp]
    else:
        for c in ["elo_pre", "mov_elo", "glicko", "glicko_rd", "opp_elo_pre"]:
            h[c] = np.nan
    h["elo_expected"] = 1 / (1 + 10 ** ((h["opp_elo_pre"] - h["elo_pre"]) / 400))

    h = h.merge(totals, on=["EVENT", "BOUT", "fighter_key"], how="left")
    opp_stats = totals.rename(columns={c: f"opp_{c}" for c in totals.columns
                                       if c not in ("EVENT", "BOUT", "fighter_key")})
    opp_stats = opp_stats.rename(columns={"fighter_key": "opp_key"})
    h = h.merge(opp_stats, on=["EVENT", "BOUT", "opp_key"], how="left")

    h = h.sort_values(["fighter_key", "date"]).reset_index(drop=True)
    g = h.groupby("fighter_key", sort=False)
    fk = h["fighter_key"]

    # ---- vectorized group helpers (no python lambdas: ~10x faster) ----
    def prior_cum(col):
        filled = h[col].fillna(0)
        return filled.groupby(fk, sort=False).cumsum() - filled

    def roll3(col, how):
        shifted = g[col].shift(1)
        r = shifted.groupby(fk, sort=False).rolling(3, min_periods=1)
        return (r.mean() if how == "mean" else r.sum()).reset_index(level=0, drop=True)

    h["n_prior"] = g.cumcount()
    h["prior_wins"] = prior_cum("won")
    h["prior_losses"] = h["n_prior"] - h["prior_wins"]
    n_pri = h["n_prior"].replace(0, np.nan)
    h["win_rate"] = h["prior_wins"] / n_pri
    h["recent3_wr"] = roll3("won", "mean")

    # streak: consecutive same results entering this fight (+wins / -losses)
    def calc_streak(results):
        out, streak = [], 0
        for r in results:
            out.append(streak)
            streak = streak + 1 if r == 1 else -1 if streak > 0 else streak - 1
        return out
    h["streak"] = g["won"].transform(
        lambda x: pd.Series(calc_streak(x.tolist()), index=x.index))

    h["days_since_last"] = (h["date"] - g["date"].shift(1)).dt.days

    # finishing power & durability (shares of prior fights)
    for col in ["ko_win", "sub_win", "ko_loss", "sub_loss"]:
        h[f"{col}_share"] = prior_cum(col) / n_pri

    # small-cage (Apex) record entering this fight
    h["apex_won"] = h["won"] * h["is_apex_fight"]
    h["n_apex"] = prior_cum("is_apex_fight")
    h["apex_wr"] = prior_cum("apex_won") / h["n_apex"].replace(0, np.nan)

    # opponent quality: avg Elo faced + Elo-expected overperformance
    h["opp_elo_avg"] = prior_cum("opp_elo_pre") / n_pri
    h["overperf_pf"] = h["won"] - h["elo_expected"]
    h["overperf"] = prior_cum("overperf_pf") / n_pri

    # recent form: same rate stats over the LAST 3 fights only
    mins3 = roll3("seconds", "sum") / 60.0
    h["slpm_r3"] = roll3("sig_l", "sum") / mins3
    h["sapm_r3"] = roll3("opp_sig_l", "sum") / mins3
    h["td_avg15_r3"] = roll3("td_l", "sum") / mins3 * 15

    mins = prior_cum("seconds") / 60.0
    h["avg_fight_mins"] = mins / n_pri

    # cardio / fade curve: career R3 output as a share of R1 output,
    # over prior fights that actually reached round 3
    h["fade_ratio"] = prior_cum("r3_sig") / prior_cum("r1_when3").replace(0, np.nan)

    # quality-weighted momentum: recent results weighted by opponent Elo
    h["recent3_opp_elo"] = roll3("opp_elo_pre", "mean")
    h["qual_win_pf"] = h["won"] * h["opp_elo_pre"] / ELO_START
    h["recent3_qualwr"] = roll3("qual_win_pf", "mean")
    h["recent3_overperf"] = roll3("overperf_pf", "mean")

    # damage mileage: absolute absorbed strikes, cage time, recency of last KO loss
    h["dmg_absorbed"] = prior_cum("opp_sig_l")
    h["cage_mins"] = mins
    ko_date = h["date"].where(h["ko_loss"] == 1)
    h["last_ko_date"] = ko_date.groupby(fk, sort=False).shift(1) \
                               .groupby(fk, sort=False).cummax()
    h["days_since_ko"] = (h["date"] - h["last_ko_date"]).dt.days
    h["ko_recent"] = (h["days_since_ko"] < 540).astype(float).fillna(0.0)

    for col in ["sig_l", "sig_a", "tot_l", "td_l", "td_a", "kd", "sub_att", "ctrl_s",
                "head_l", "body_l", "leg_l", "dist_l", "clinch_l", "ground_l",
                "opp_sig_l", "opp_sig_a", "opp_td_l", "opp_td_a", "seconds"]:
        h[f"c_{col}"] = prior_cum(col)

    h["slpm"] = h["c_sig_l"] / mins
    h["str_acc"] = h["c_sig_l"] / h["c_sig_a"].replace(0, np.nan)
    h["sapm"] = h["c_opp_sig_l"] / mins
    h["str_def"] = 1 - h["c_opp_sig_l"] / h["c_opp_sig_a"].replace(0, np.nan)
    h["td_avg15"] = h["c_td_l"] / mins * 15
    h["td_acc"] = h["c_td_l"] / h["c_td_a"].replace(0, np.nan)
    h["td_def"] = 1 - h["c_opp_td_l"] / h["c_opp_td_a"].replace(0, np.nan)
    h["sub_avg15"] = h["c_sub_att"] / mins * 15
    h["kd_avg15"] = h["c_kd"] / mins * 15
    h["ctrl_share"] = h["c_ctrl_s"] / h["c_seconds"].replace(0, np.nan)
    sig = h["c_sig_l"].replace(0, np.nan)
    h["head_share"] = h["c_head_l"] / sig
    h["body_share"] = h["c_body_l"] / sig
    h["leg_share"] = h["c_leg_l"] / sig
    h["dist_share"] = h["c_dist_l"] / sig
    h["clinch_share"] = h["c_clinch_l"] / sig
    h["ground_share"] = h["c_ground_l"] / sig

    return h


FEATURES = [
    "n_prior", "prior_wins", "prior_losses", "win_rate", "recent3_wr", "streak",
    "days_since_last", "slpm", "str_acc", "sapm", "str_def", "td_avg15", "td_acc",
    "td_def", "sub_avg15", "kd_avg15", "ctrl_share", "head_share", "body_share",
    "leg_share", "dist_share", "clinch_share", "ground_share",
    "height_in", "reach_in", "age", "stance_orthodox", "stance_southpaw", "elo",
    "h2h_wins",
    # recent form (last 3 fights)
    "slpm_r3", "sapm_r3", "td_avg15_r3",
    # opponent quality
    "opp_elo_avg", "overperf",
    # finishing power & durability
    "ko_win_share", "sub_win_share", "ko_loss_share", "sub_loss_share",
    "avg_fight_mins",
    # venue & camp
    "apex_wr", "n_apex", "team_wr",
    # cardio, momentum quality, damage mileage
    "fade_ratio", "recent3_opp_elo", "recent3_qualwr", "recent3_overperf",
    "dmg_absorbed", "cage_mins", "days_since_ko", "ko_recent",
    # rating-system upgrades (pre-fight values)
    "mov_elo", "glicko", "glicko_rd",
]


def deterministic_flip(fight_id: str) -> bool:
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0


def build_rows(fights, history, attrs, elo, odds, matchup_hist) -> pd.DataFrame:
    hist = history.set_index(["fight_id", "fighter_key"])
    attrs = attrs.set_index("fighter_key")

    def side_features(fight_id, key, date):
        try:
            hrow = hist.loc[(fight_id, key)]
        except KeyError:
            return None
        feats = {f: hrow.get(f, np.nan) for f in FEATURES if f in hrow.index}
        feats["elo"] = elo.get((fight_id, key), (np.nan,))[0]
        if key in attrs.index:
            a = attrs.loc[key]
            feats["height_in"] = a["height_in"]
            feats["reach_in"] = a["reach_in"]
            feats["age"] = (date - a["dob"]).days / 365.25 if pd.notna(a["dob"]) else np.nan
            feats["stance_orthodox"] = a["stance_orthodox"]
            feats["stance_southpaw"] = a["stance_southpaw"]
        else:
            feats.update({k: np.nan for k in
                          ["height_in", "reach_in", "age", "stance_orthodox", "stance_southpaw"]})
        return feats

    out = []
    for _, f in fights.iterrows():
        k1, k2 = norm_name(f["f1"]), norm_name(f["f2"])
        s1 = side_features(f["fight_id"], k1, f["date"])
        s2 = side_features(f["fight_id"], k2, f["date"])
        if s1 is None or s2 is None:
            continue
        if s1["n_prior"] < MIN_PRIOR_FIGHTS or s2["n_prior"] < MIN_PRIOR_FIGHTS:
            continue

        mh = matchup_hist.get(f["fight_id"], {})
        s1["h2h_wins"] = mh.get("h2h_wins_f1", 0.0)
        s2["h2h_wins"] = mh.get("h2h_wins_f2", 0.0)

        # deterministic orientation flip -> ~50/50 labels, no positional bias
        if deterministic_flip(f["fight_id"]):
            a, b, a_name, b_name = s2, s1, f["f2"], f["f1"]
            a_key, b_key = k2, k1
            label = int(norm_name(f["winner"]) == k2)
            common_edge = -mh.get("common_edge_f1", np.nan) \
                if pd.notna(mh.get("common_edge_f1", np.nan)) else np.nan
        else:
            a, b, a_name, b_name = s1, s2, f["f1"], f["f2"]
            a_key, b_key = k1, k2
            label = int(norm_name(f["winner"]) == k1)
            common_edge = mh.get("common_edge_f1", np.nan)

        wc_lbs, is_women, is_title = parse_weightclass(f.get("WEIGHTCLASS", ""))
        row = {"label": label, "date": f["date"], "weightclass": f.get("WEIGHTCLASS", ""),
               "method": parse_method(f.get("METHOD", "")),
               "wc_lbs": wc_lbs, "is_women": is_women, "is_title": is_title,
               "is_apex": float(f.get("is_apex", 0.0)),
               "n_common": mh.get("n_common", 0.0), "common_edge": common_edge}

        for feat in FEATURES:
            row[f"a_{feat}"] = a.get(feat, np.nan)
            row[f"b_{feat}"] = b.get(feat, np.nan)
            row[f"diff_{feat}"] = row[f"a_{feat}"] - row[f"b_{feat}"] \
                if pd.notna(row[f"a_{feat}"]) and pd.notna(row[f"b_{feat}"]) else np.nan

        # auxiliary multi-task targets (META, not features): what actually
        # happened in the fight — used by the stacker's dominance forecasts
        try:
            a_sig = hist.loc[(f["fight_id"], a_key)]["sig_l"]
            b_sig = hist.loc[(f["fight_id"], b_key)]["sig_l"]
            row["sig_strike_diff"] = float(a_sig) - float(b_sig)
        except (KeyError, TypeError, ValueError):
            row["sig_strike_diff"] = np.nan
        row["fight_secs"] = f["total_seconds"]

        # betting odds: vig-free implied prob (feature) + raw american (backtest meta)
        market = odds.get((f["date"], frozenset((k1, k2))), {})
        pa, oa = market.get(a_key, (np.nan, np.nan))
        pb, ob = market.get(b_key, (np.nan, np.nan))
        row["a_implied_prob"] = pa
        row["diff_implied_prob"] = pa - pb if market else np.nan
        row["a_odds_am"], row["b_odds_am"] = oa, ob

        row["a_name"], row["b_name"] = a_name, b_name
        out.append(row)

    return pd.DataFrame(out)


def add_weightclass_relative(df: pd.DataFrame) -> pd.DataFrame:
    """stat minus the expanding mean of that stat in the same weight class,
    using only rows from strictly earlier fights (leak-free)."""
    df = df.sort_values("date").reset_index(drop=True)
    for feat in WC_REL_FEATURES:
        a = df[["date", "wc_lbs"]].assign(v=df[f"a_{feat}"], side="a", ridx=df.index)
        b = df[["date", "wc_lbs"]].assign(v=df[f"b_{feat}"], side="b", ridx=df.index)
        pooled = pd.concat([a, b], ignore_index=True).sort_values(
            ["date", "ridx"], kind="stable")
        pooled["wc_mean"] = (pooled.groupby("wc_lbs")["v"]
                             .transform(lambda x: x.expanding().mean().shift(1)))
        am = pooled[pooled.side == "a"].set_index("ridx")["wc_mean"]
        bm = pooled[pooled.side == "b"].set_index("ridx")["wc_mean"]
        m = (am.reindex(df.index) + bm.reindex(df.index)) / 2
        df[f"a_rel_{feat}"] = df[f"a_{feat}"] - m
        df[f"b_rel_{feat}"] = df[f"b_{feat}"] - m
        df[f"diff_rel_{feat}"] = df[f"a_rel_{feat}"] - df[f"b_rel_{feat}"]
    return df


def main():
    print("loading raw data...")
    fights = load_fights()
    totals = load_fight_totals()
    attrs = load_fighter_attrs()
    odds = load_odds()
    print(f"{len(fights)} decisive fights, {len(attrs)} fighters")

    print("computing Elo ratings...")
    elo, _ = compute_elo(fights)

    print("computing h2h / common-opponent history...")
    matchup_hist = compute_matchup_history(fights)

    print("building career histories (as-of-date)...")
    history = build_career_history(fights, totals, elo_pre=elo, teams=load_teams())

    print("building matchup rows...")
    df = build_rows(fights, history, attrs, elo, odds, matchup_hist)
    df = add_weightclass_relative(df)
    df = add_known_flags(df)

    os.makedirs("stats", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows x {df.shape[1]} cols -> {OUT}")
    print(f"label balance: {df.label.mean():.3f}")
    print(f"odds coverage: {df.a_implied_prob.notna().mean():.1%}")
    print(f"date range: {df.date.min().date()} .. {df.date.max().date()}")


if __name__ == "__main__":
    main()
