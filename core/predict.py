"""Predict a UFC fight by weighing multiple strategies, then compare to market.

Usage:
  python -m core.predict "Islam Makhachev" "Charles Oliveira"
  python -m core.predict "Fighter A" "Fighter B" --odds -180 150
  python -m core.predict --card card.txt     # 'Fighter A vs Fighter B[, -180, 150]' per line
  python -m core.predict "A" "B" --title

Output per fight:
  - each strategy's independent probability (form, activity, physical,
    striking, grappling, matchup history, Elo, full ML model)
  - the learned weight of each strategy (stacked meta-model)
  - the ensemble probability + fair line
  - market comparison: vig-free implied probs, model-vs-market edge, EV per side

The ensemble never sees the odds, so the edge is a real disagreement signal.
Backtest note (see backtest.py): historically the closing line wins these
disagreements — treat 'edges' as research leads, not bets.
"""

import argparse
import difflib
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from core import build_dataset as bd
from core.strategies import StrategyEnsemble
from core.utils import META_COLS

EDGE_NOTE_THRESHOLD = 0.08


# ---------- current (as-of-today) fighter features ----------

def fighter_snapshot(h, attrs_idx, elo_now, key, today):
    rows = h[h["fighter_key"] == key].sort_values("date")
    if rows.empty:
        return None

    n = len(rows)
    wins = rows["won"].sum()
    mins = rows["seconds"].sum() / 60.0
    results = rows["won"].tolist()

    streak = 0
    for r in results:
        streak = streak + 1 if r == 1 else (-1 if streak > 0 else streak - 1)

    def safe_div(a, b):
        return a / b if b and not pd.isna(b) else np.nan

    sig_l = rows["sig_l"].sum()
    f = {
        "n_prior": n,
        "prior_wins": wins,
        "prior_losses": n - wins,
        "win_rate": wins / n,
        "recent3_wr": np.mean(results[-3:]),
        "streak": streak,
        "days_since_last": (today - rows["date"].max()).days,
        "slpm": safe_div(sig_l, mins),
        "str_acc": safe_div(sig_l, rows["sig_a"].sum()),
        "sapm": safe_div(rows["opp_sig_l"].sum(), mins),
        "str_def": 1 - safe_div(rows["opp_sig_l"].sum(), rows["opp_sig_a"].sum()),
        "td_avg15": safe_div(rows["td_l"].sum(), mins) * 15,
        "td_acc": safe_div(rows["td_l"].sum(), rows["td_a"].sum()),
        "td_def": 1 - safe_div(rows["opp_td_l"].sum(), rows["opp_td_a"].sum()),
        "sub_avg15": safe_div(rows["sub_att"].sum(), mins) * 15,
        "kd_avg15": safe_div(rows["kd"].sum(), mins) * 15,
        "ctrl_share": safe_div(rows["ctrl_s"].sum(), rows["seconds"].sum()),
        "avg_fight_mins": mins / n,
        "ko_win_share": rows["ko_win"].sum() / n,
        "sub_win_share": rows["sub_win"].sum() / n,
        "ko_loss_share": rows["ko_loss"].sum() / n,
        "sub_loss_share": rows["sub_loss"].sum() / n,
        "opp_elo_avg": rows["opp_elo_pre"].mean(),
        "overperf": (rows["won"] - rows["elo_expected"]).mean(),
        "n_apex": rows["is_apex_fight"].sum(),
        "apex_wr": safe_div(rows.loc[rows["is_apex_fight"] > 0, "won"].sum(),
                            rows["is_apex_fight"].sum()),
        "team_wr": rows["team_wr"].iloc[-1] if "team_wr" in rows else np.nan,
        "head_share": safe_div(rows["head_l"].sum(), sig_l),
        "body_share": safe_div(rows["body_l"].sum(), sig_l),
        "leg_share": safe_div(rows["leg_l"].sum(), sig_l),
        "dist_share": safe_div(rows["dist_l"].sum(), sig_l),
        "clinch_share": safe_div(rows["clinch_l"].sum(), sig_l),
        "ground_share": safe_div(rows["ground_l"].sum(), sig_l),
        "elo": elo_now.get(key, (bd.ELO_START,) * 3 + (350.0,))[0],
        "mov_elo": elo_now.get(key, (bd.ELO_START,) * 3 + (350.0,))[1],
        "glicko": elo_now.get(key, (bd.ELO_START,) * 3 + (350.0,))[2],
        "glicko_rd": elo_now.get(key, (bd.ELO_START,) * 3 + (350.0,))[3],
    }

    last3 = rows.tail(3)
    mins3 = last3["seconds"].sum() / 60.0
    f["slpm_r3"] = safe_div(last3["sig_l"].sum(), mins3)
    f["sapm_r3"] = safe_div(last3["opp_sig_l"].sum(), mins3)
    f["td_avg15_r3"] = safe_div(last3["td_l"].sum(), mins3) * 15 if mins3 else np.nan

    # cardio / fade
    f["fade_ratio"] = safe_div(rows["r3_sig"].sum(), rows["r1_when3"].sum())
    # quality-weighted momentum
    f["recent3_opp_elo"] = last3["opp_elo_pre"].mean()
    f["recent3_qualwr"] = (last3["won"] * last3["opp_elo_pre"] / bd.ELO_START).mean()
    f["recent3_overperf"] = (last3["won"] - last3["elo_expected"]).mean()
    # damage mileage
    f["dmg_absorbed"] = rows["opp_sig_l"].sum()
    f["cage_mins"] = mins
    ko_dates = rows.loc[rows["ko_loss"] == 1, "date"]
    f["days_since_ko"] = (today - ko_dates.max()).days if len(ko_dates) else np.nan
    f["ko_recent"] = float(f["days_since_ko"] < 540) if pd.notna(f["days_since_ko"]) else 0.0

    if key in attrs_idx.index:
        a = attrs_idx.loc[key]
        f["height_in"] = a["height_in"]
        f["reach_in"] = a["reach_in"]
        f["age"] = (today - a["dob"]).days / 365.25 if pd.notna(a["dob"]) else np.nan
        f["stance_orthodox"] = a["stance_orthodox"]
        f["stance_southpaw"] = a["stance_southpaw"]
    else:
        f.update({k: np.nan for k in
                  ["height_in", "reach_in", "age", "stance_orthodox", "stance_southpaw"]})
    return f


def build_past_results(history):
    """past[k][opp] = [1/0 results] over all fights to date."""
    past = defaultdict(lambda: defaultdict(list))
    for _, r in history.sort_values("date").iterrows():
        past[r["fighter_key"]][r["opp_key"]].append(r["won"])
    return past


def matchup_features(past, ka, kb):
    h2h_a = float(sum(past[ka][kb])) if past[ka][kb] else 0.0
    h2h_b = float(sum(past[kb][ka])) if past[kb][ka] else 0.0
    common = (set(past[ka]) & set(past[kb])) - {ka, kb}
    edges = [np.mean(past[ka][o]) - np.mean(past[kb][o]) for o in common]
    edge = float(np.mean(edges)) if edges else np.nan
    return h2h_a, h2h_b, float(len(common)), edge


def resolve_name(name, all_keys):
    key = bd.norm_name(name)
    if key in all_keys:
        return key
    matches = difflib.get_close_matches(key, all_keys, n=3, cutoff=0.6)
    if not matches:
        sys.exit(f"fighter not found: '{name}' (no close matches)")
    print(f"  note: '{name}' matched to '{matches[0]}'")
    return matches[0]


def latest_weightclass(fights, key):
    mask = (fights["f1"].map(bd.norm_name) == key) | (fights["f2"].map(bd.norm_name) == key)
    rows = fights[mask].sort_values("date")
    return rows.iloc[-1].get("WEIGHTCLASS", "") if len(rows) else ""


# ---------- row assembly & prediction ----------

def make_row(sa, sb, wc_row, wc_means):
    row = dict(wc_row)
    for feat in bd.FEATURES:
        row[f"a_{feat}"] = sa.get(feat, np.nan)
        row[f"b_{feat}"] = sb.get(feat, np.nan)
        av, bv = row[f"a_{feat}"], row[f"b_{feat}"]
        row[f"diff_{feat}"] = av - bv if pd.notna(av) and pd.notna(bv) else np.nan
    for feat in bd.WC_REL_FEATURES:
        m = wc_means.get((wc_row["wc_lbs"], feat), np.nan)
        row[f"a_rel_{feat}"] = sa.get(feat, np.nan) - m
        row[f"b_rel_{feat}"] = sb.get(feat, np.nan) - m
        row[f"diff_rel_{feat}"] = row[f"a_rel_{feat}"] - row[f"b_rel_{feat}"]
    row["a_implied_prob"] = np.nan
    row["diff_implied_prob"] = np.nan
    return row


def predict_fight(ens, feature_cols, sa, sb, wc_row_a, wc_row_b, wc_means,
                  method_model=None):
    """Per-strategy and ensemble P(a wins), averaged over both orientations."""
    X = pd.DataFrame([make_row(sa, sb, wc_row_a, wc_means),
                      make_row(sb, sa, wc_row_b, wc_means)])
    X = bd.add_known_flags(X)
    X = X.reindex(columns=feature_cols)
    sp = ens.strategy_probs(X)
    per_strategy = {c: (sp[c].iloc[0] + (1 - sp[c].iloc[1])) / 2 for c in sp.columns}
    p = ens.predict_proba(X)
    p_avg = (p[0] + (1 - p[1])) / 2
    confident = bool(ens.confidence(X.iloc[[0]], np.array([p_avg]))[0]) \
        if hasattr(ens, "confidence") else True
    method_probs = None
    if method_model is not None:
        m = method_model.predict_proba(X.to_numpy(dtype=float))
        method_probs = dict(zip(method_model.classes_, (m[0] + m[1]) / 2))
    return p_avg, per_strategy, method_probs, confident


def fair_line(p):
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def ev_per_unit(p_win, american):
    payout = american / 100 if american > 0 else 100 / -american
    return p_win * payout - (1 - p_win)


def parse_card_line(line):
    parts = [p.strip() for p in line.strip().split(",")]
    lower = parts[0].lower()
    if " vs " not in lower:
        return None
    idx = lower.index(" vs ")
    a, b = parts[0][:idx].strip(), parts[0][idx + 4:].strip()
    odds = (float(parts[1]), float(parts[2])) if len(parts) >= 3 else None
    return a, b, odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fighters", nargs="*", help="two fighter names")
    ap.add_argument("--card", help="file: 'Fighter A vs Fighter B[, -180, 150]' per line")
    ap.add_argument("--odds", nargs=2, type=float, metavar=("A", "B"),
                    help="American odds, e.g. --odds -180 150")
    ap.add_argument("--market", action="store_true",
                    help="fetch live probabilities from Polymarket + Kalshi")
    ap.add_argument("--title", action="store_true", help="title fight")
    args = ap.parse_args()

    if args.card:
        pairs = [p for p in (parse_card_line(l) for l in open(args.card)) if p]
        if not pairs:
            sys.exit("no 'A vs B' lines found in card file")
    elif len(args.fighters) == 2:
        pairs = [(args.fighters[0], args.fighters[1], tuple(args.odds) if args.odds else None)]
    else:
        sys.exit("give two fighter names, or --card file")

    print("loading data & fitting strategy ensemble...")
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in META_COLS]
    y = df["label"].to_numpy(dtype=int)
    ens = StrategyEnsemble(feature_cols).fit(df[feature_cols], y)

    # method-of-victory model (decision / ko / sub) — HGB handles NaNs natively
    from sklearn.ensemble import HistGradientBoostingClassifier
    method_model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    method_model.fit(df[feature_cols].to_numpy(dtype=float),
                     df["method"].to_numpy())

    fights = bd.load_fights()
    totals = bd.load_fight_totals()
    attrs_idx = bd.load_fighter_attrs().set_index("fighter_key")
    elo_pre, elo_now = bd.compute_elo(fights)
    history = bd.build_career_history(fights, totals, elo_pre=elo_pre,
                                      teams=bd.load_teams())
    past = build_past_results(history)
    all_keys = set(history["fighter_key"])
    today = pd.Timestamp(datetime.now().date())

    wc_means = {}
    for feat in bd.WC_REL_FEATURES:
        pooled = pd.concat([df[["wc_lbs"]].assign(v=df[f"a_{feat}"]),
                            df[["wc_lbs"]].assign(v=df[f"b_{feat}"])])
        for wc, m in pooled.groupby("wc_lbs")["v"].mean().items():
            wc_means[(wc, feat)] = m

    weights = ens.weights
    max_w = max(abs(w) for w in weights.values())

    for name_a, name_b, odds in pairs:
        print("\n" + "=" * 62)
        ka, kb = resolve_name(name_a, all_keys), resolve_name(name_b, all_keys)
        sa = fighter_snapshot(history, attrs_idx, elo_now, ka, today)
        sb = fighter_snapshot(history, attrs_idx, elo_now, kb, today)

        h2h_a, h2h_b, n_common, edge_ab = matchup_features(past, ka, kb)
        sa["h2h_wins"], sb["h2h_wins"] = h2h_a, h2h_b

        wc_str = latest_weightclass(fights, ka) or latest_weightclass(fights, kb)
        wc_lbs, is_women, _ = bd.parse_weightclass(wc_str)
        base = {"wc_lbs": wc_lbs, "is_women": is_women, "is_title": float(args.title),
                "is_apex": 0.0, "n_common": n_common}
        wc_row_a = dict(base, common_edge=edge_ab)
        wc_row_b = dict(base, common_edge=-edge_ab if pd.notna(edge_ab) else np.nan)

        p, per_strategy, method_probs, confident = predict_fight(
            ens, feature_cols, sa, sb, wc_row_a, wc_row_b, wc_means, method_model)

        print(f"{name_a} ({int(sa['prior_wins'])}-{int(sa['prior_losses'])}, "
              f"elo {sa['elo']:.0f})  vs  "
              f"{name_b} ({int(sb['prior_wins'])}-{int(sb['prior_losses'])}, "
              f"elo {sb['elo']:.0f})\n")
        print(f"  {'strategy':20s} {'P(' + name_a.split()[-1] + ')':>10s}   weight")
        for sname, sp in per_strategy.items():
            bar = "#" * int(round(abs(weights[sname]) / max_w * 10))
            print(f"  {sname:20s} {sp:9.1%}   {weights[sname]:+.2f} {bar}")
        tag = "" if confident else "   << TOO CLOSE TO CALL"
        print(f"\n  {'ENSEMBLE':20s} {p:9.1%}   fair line {fair_line(p)}{tag}")
        if method_probs:
            labels = {"dec": "decision", "ko": "KO/TKO", "sub": "submission"}
            parts = " / ".join(f"{labels.get(k, k)} {v:.0%}" for k, v in
                               sorted(method_probs.items(), key=lambda x: -x[1]))
            print(f"  {'how it ends':20s} {parts}")

        mkt_a = None
        if odds:
            pa_raw = bd.american_to_prob(odds[0])
            pb_raw = bd.american_to_prob(odds[1])
            mkt_a = pa_raw / (pa_raw + pb_raw)
            print(f"\n  sportsbook (vig-free){mkt_a:9.1%} / {1 - mkt_a:.1%}")
            print(f"  EV per 1u:  {name_a} {ev_per_unit(p, odds[0]):+.2f}   "
                  f"{name_b} {ev_per_unit(1 - p, odds[1]):+.2f}")
        elif args.market:
            from scrapers.fetch_market_odds import get_market_probs
            sources = get_market_probs(name_a, name_b)
            if not sources:
                print("\n  no open Polymarket/Kalshi market found for this fight")
            else:
                print()
                for src, (pa_m, pb_m, note) in sources.items():
                    print(f"  {src:12s}       {pa_m:9.1%} / {pb_m:.1%}   [{note}]")
                mkt_a = sum(v[0] for v in sources.values()) / len(sources)
                # EV of buying a $1-payout contract at the market price
                if 0 < mkt_a < 1:
                    print(f"  EV per 1u:  {name_a} {p / mkt_a - 1:+.2f}   "
                          f"{name_b} {(1 - p) / (1 - mkt_a) - 1:+.2f}")

        if mkt_a is not None:
            edge = p - mkt_a
            print(f"  model vs market edge {edge:+9.1%} on {name_a}")
            if abs(edge) >= EDGE_NOTE_THRESHOLD:
                side = name_a if edge > 0 else name_b
                print(f"  >> mismatch flagged: model likes {side} much more than "
                      f"the market.")
                print(f"     caution: in backtests the market usually wins "
                      f"disagreements this large.")
            else:
                print("  no meaningful mismatch — model and market agree.")

    print()


if __name__ == "__main__":
    main()
