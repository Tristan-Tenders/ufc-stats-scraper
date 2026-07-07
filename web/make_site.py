"""Generate a static prediction website for upcoming UFC cards.

  python -m scrapers.fetch_upcoming          # scrape scheduled fights first
  python -m web.make_site               # -> site/index.html
  python -m web.make_site --market      # include live Polymarket/Kalshi prices
  python -m web.make_site --fast        # quicker ensemble fit (skips OOF stacking)

Open site/index.html in any browser — it's fully self-contained.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from core import build_dataset as bd
from core import predict as P
from core.strategies import STRATEGY_EXPLAIN, StrategyEnsemble
from core.utils import META_COLS


def setup_models(fast: bool):
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in META_COLS]
    y = df["label"].to_numpy(dtype=int)

    ens = StrategyEnsemble(feature_cols)
    aux = df[["sig_strike_diff", "fight_secs"]] \
        if "sig_strike_diff" in df.columns else None
    if fast:
        ens.fit_fast(df[feature_cols], y, dates=df["date"], aux_targets=aux)
    else:
        ens.fit(df[feature_cols], y, dates=df["date"], aux_targets=aux)

    method_model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    method_model.fit(df[feature_cols].to_numpy(dtype=float), df["method"].to_numpy())

    wc_means = {}
    for feat in bd.WC_REL_FEATURES:
        pooled = pd.concat([df[["wc_lbs"]].assign(v=df[f"a_{feat}"]),
                            df[["wc_lbs"]].assign(v=df[f"b_{feat}"])])
        for wc, m in pooled.groupby("wc_lbs")["v"].mean().items():
            wc_means[(wc, feat)] = m

    # share the fitted models with the web app (same cache format as app.py)
    import pickle
    with open("stats/model_cache.pkl", "wb") as cf:
        pickle.dump((ens, method_model, feature_cols, wc_means), cf)
    print("model cache saved — app.py will start instantly")
    return df, feature_cols, ens, method_model, wc_means


def setup_data():
    fights = bd.load_fights()
    totals = bd.load_fight_totals()
    attrs_idx = bd.load_fighter_attrs().set_index("fighter_key")
    elo_pre, elo_now = bd.compute_elo(fights)
    history = bd.build_career_history(fights, totals, elo_pre=elo_pre,
                                      teams=bd.load_teams())
    past = P.build_past_results(history)
    return fights, attrs_idx, elo_now, history, past


def predict_upcoming(upcoming, feature_cols, ens, method_model, wc_means,
                     fights, attrs_idx, elo_now, history, past,
                     use_market: bool):
    all_keys = set(history["fighter_key"])
    today = pd.Timestamp(datetime.now().date())
    results = []
    for _, f in upcoming.iterrows():
        ka, kb = bd.norm_name(f["F1"]), bd.norm_name(f["F2"])
        row = {"event": f["EVENT"], "date": f["DATE"], "location": f.get("LOCATION", ""),
               "f1": f["F1"], "f2": f["F2"], "weightclass": f.get("WEIGHTCLASS", "")}
        if ka not in all_keys or kb not in all_keys:
            row["status"] = "no data (debut or unmatched name)"
            results.append(row)
            continue

        sa = P.fighter_snapshot(history, attrs_idx, elo_now, ka, today)
        sb = P.fighter_snapshot(history, attrs_idx, elo_now, kb, today)
        h2h_a, h2h_b, n_common, edge_ab = P.matchup_features(past, ka, kb)
        sa["h2h_wins"], sb["h2h_wins"] = h2h_a, h2h_b

        wc_lbs, is_women, _ = bd.parse_weightclass(row["weightclass"])
        is_apex = bd.is_apex_event(f["EVENT"], f.get("LOCATION", ""),
                                   pd.to_datetime(f["DATE"], errors="coerce"))
        base = {"wc_lbs": wc_lbs, "is_women": is_women, "is_title": 0.0,
                "is_apex": is_apex, "n_common": n_common}
        p, per_strategy, method_probs, confident = P.predict_fight(
            ens, feature_cols, sa, sb,
            dict(base, common_edge=edge_ab),
            dict(base, common_edge=-edge_ab if pd.notna(edge_ab) else np.nan),
            wc_means, method_model)

        row.update({
            "status": "ok", "p1": p, "confident": confident,
            "elo1": sa["elo"], "elo2": sb["elo"],
            "rec1": f"{int(sa['prior_wins'])}-{int(sa['prior_losses'])}",
            "rec2": f"{int(sb['prior_wins'])}-{int(sb['prior_losses'])}",
            "method": method_probs,
            "strategies": per_strategy,
        })
        if use_market:
            from scrapers.fetch_market_odds import get_market_probs
            sources = get_market_probs(f["F1"], f["F2"])
            if sources:
                row["market"] = sum(v[0] for v in sources.values()) / len(sources)
                # keep each source separately for the frontend
                row["market_sources"] = {src: pa for src, (pa, pb, note)
                                         in sources.items()}
        results.append(row)
        winner = f["F1"] if p >= 0.5 else f["F2"]
        print(f"  {f['F1']} vs {f['F2']}: {winner} ({max(p, 1 - p):.0%})")
    return results


# ---------- html ----------

PLACEHOLDER = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
               "viewBox='0 0 100 100'><rect width='100' height='100' fill='%23181818'/>"
               "<circle cx='50' cy='38' r='18' fill='%23333'/>"
               "<ellipse cx='50' cy='85' rx='30' ry='22' fill='%23333'/></svg>")

CSS = """
:root{--red:#ff5c69;--blue:#4d9fff;--green:#2ee6a8;--bg:#0b0e14;--ink:#e8ecf4}
*{box-sizing:border-box}
body{font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;color:var(--ink);
margin:0;padding:0;background:#0b0e14;
background:radial-gradient(1100px 500px at 50% -10%,#182135 0%,#0b0e14 60%) fixed}
.topbar{background:rgba(11,14,20,.72);backdrop-filter:blur(14px);
-webkit-backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,.07);
padding:16px 24px;position:sticky;top:0;z-index:10;display:flex;
align-items:baseline;gap:14px}
.topbar h1{margin:0;font-size:20px;font-weight:700;letter-spacing:.3px}
.topbar h1 span{background:linear-gradient(90deg,var(--red),var(--blue));
-webkit-background-clip:text;background-clip:text;color:transparent}
.topbar .gen{color:#5d6a80;font-size:12px}
.wrap{max-width:920px;margin:auto;padding:24px 16px 48px}
.event{margin-top:36px}
.eventhead h2{margin:0;font-size:26px;font-weight:800;letter-spacing:-.3px}
.eventsub{color:#7d8aa0;font-size:13px;margin:6px 0 18px;display:flex;gap:8px;
flex-wrap:wrap}
.eventsub span{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
padding:3px 12px;border-radius:20px}
.fight{background:rgba(255,255,255,.038);border:1px solid rgba(255,255,255,.08);
border-radius:18px;margin:16px 0;padding:22px 24px 18px;position:relative;
backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
box-shadow:0 10px 34px rgba(0,0,0,.35);
transition:transform .18s,border-color .18s,box-shadow .18s}
.fight:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.17);
box-shadow:0 16px 44px rgba(0,0,0,.5)}
.fight.main{border-color:rgba(46,230,168,.35);
box-shadow:0 0 0 1px rgba(46,230,168,.18),0 14px 44px rgba(0,0,0,.45)}
.mainevent-tag{position:absolute;top:-11px;left:22px;
background:linear-gradient(90deg,var(--green),#1fb885);color:#04110c;
font-weight:700;font-size:11px;letter-spacing:1.6px;text-transform:uppercase;
padding:3px 14px;border-radius:20px}
.wc{color:#7d8aa0;font-size:12px;text-transform:uppercase;letter-spacing:2.5px;
text-align:center;margin-bottom:14px}
.matchup{display:flex;align-items:center;justify-content:space-between;gap:10px}
.corner{flex:1;display:flex;flex-direction:column;align-items:center;text-align:center}
.disc{width:84px;height:84px;border-radius:24px;background:#161b26;
display:flex;align-items:center;justify-content:center;font-size:28px;
font-weight:800;color:#7d8aa0;border:2px solid rgba(255,255,255,.12)}
.main .disc{width:104px;height:104px;font-size:34px}
.corner.r .disc{box-shadow:0 0 0 2px rgba(255,92,105,.5);color:var(--red)}
.corner.b .disc{box-shadow:0 0 0 2px rgba(77,159,255,.5);color:var(--blue)}
.corner.winner .disc{box-shadow:0 0 0 2px var(--green),
0 0 26px rgba(46,230,168,.35);color:var(--green)}
details.why{margin-top:14px}
details.why summary{color:#7d8aa0;font-size:13px;cursor:pointer;
text-align:center}
.whyrow{display:flex;gap:10px;align-items:baseline;padding:7px 4px;
border-bottom:1px solid rgba(255,255,255,.05);font-size:13.5px}
.whyrow .expl{flex:1;color:#aab6c8;line-height:1.45}
.whyrow .who{white-space:nowrap;font-weight:600;
font-variant-numeric:tabular-nums}
.whyrow .who.r{color:var(--red)}.whyrow .who.b{color:var(--blue)}
.whyrow .trust{flex:0 0 56px;height:5px;border-radius:3px;
background:rgba(255,255,255,.08);align-self:center;overflow:hidden}
.whyrow .trust i{display:block;height:100%;background:var(--green)}
.fname{font-size:17px;font-weight:700;line-height:1.15;margin-top:12px;
letter-spacing:-.2px}
.main .fname{font-size:20px}
.frec{color:#7d8aa0;font-size:12px;margin-top:3px}
.wtag{color:var(--green);font-weight:600;font-size:11px;letter-spacing:1.4px;
text-transform:uppercase;margin-top:5px;min-height:14px}
.center{flex:0 0 150px;text-align:center}
.vs{font-size:12px;font-weight:600;color:#5d6a80;letter-spacing:3px}
.bigpct{font-size:36px;font-weight:800;line-height:1;margin-top:6px;
font-variant-numeric:tabular-nums;letter-spacing:-1px}
.bigpct .r{color:var(--red)}.bigpct .b{color:var(--blue)}
.bigpct .sep{color:#3a4356;font-size:22px;padding:0 5px}
.barwrap{background:rgba(255,255,255,.06);height:8px;margin-top:18px;display:flex;
overflow:hidden;border-radius:8px}
.bar{background:linear-gradient(90deg,#c2374a,var(--red));height:100%;
animation:grow .9s cubic-bezier(.2,.8,.2,1)}
.bar2{background:linear-gradient(90deg,var(--blue),#2b6fd4);height:100%;
animation:grow .9s cubic-bezier(.2,.8,.2,1)}
@keyframes grow{from{flex-basis:0}}
.chips{display:flex;justify-content:center;gap:8px;margin-top:14px;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
color:#aab6c8;font-size:12px;padding:4px 14px;border-radius:20px}
.chip b{color:var(--ink);font-weight:600}
.chip.market{border-color:rgba(46,230,168,.3);color:var(--green)}
.nodata{color:#5d6a80;font-style:italic;text-align:center;padding:8px}
footer{color:#5d6a80;font-size:12.5px;margin-top:48px;line-height:1.65;
max-width:680px}
@media(max-width:640px){.headshot{width:76px;height:76px;border-radius:20px}
.main .headshot{width:96px;height:96px}.fname{font-size:14px}
.main .fname{font-size:16px}.center{flex-basis:96px}.bigpct{font-size:26px}}
"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_images() -> dict:
    path = "stats/fighter_images.json"
    if os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    return {}


def initials(name: str) -> str:
    parts = str(name).split()
    return (parts[0][:1] + (parts[-1][:1] if len(parts) > 1 else "")).upper()


def corner_html(name, rec, elo, img, side, is_winner, wp):
    wtag = f"predicted winner &middot; {wp:.0%}" if is_winner else ""
    return (f"<div class='corner {side}{' winner' if is_winner else ''}'>"
            f"<div class='disc'>{esc(initials(name))}</div>"
            f"<div class='fname'>{esc(name)}</div>"
            f"<div class='frec'>{esc(rec)} &middot; ELO {elo:.0f}</div>"
            f"<div class='wtag'>{wtag}</div></div>")


def render_breakdown(r: dict, weights: dict) -> str:
    """Plain-English 'why these odds': one sentence per angle the model
    looked at, who that angle favors, and how much the model trusts it."""
    strategies = r.get("strategies") or {}
    if not strategies:
        return ""
    last1 = r["f1"].split()[-1]
    last2 = r["f2"].split()[-1]
    max_w = max((abs(w) for w in weights.values()), default=1) or 1
    rows = []
    order = sorted(strategies.items(),
                   key=lambda kv: -abs(weights.get(kv[0], 0)))
    for name, p in order:
        expl = STRATEGY_EXPLAIN.get(name, name)
        if abs(p - 0.5) < 0.015:
            who = "<span class='who'>even</span>"
        elif p >= 0.5:
            who = f"<span class='who r'>{esc(last1)} {p:.0%}</span>"
        else:
            who = f"<span class='who b'>{esc(last2)} {1 - p:.0%}</span>"
        trust = abs(weights.get(name, 0)) / max_w * 100
        rows.append(
            f"<div class='whyrow'><span class='expl'>{esc(expl)}</span>"
            f"{who}<span class='trust'><i style='width:{trust:.0f}%'></i></span></div>")
    return (f"<details class='why'><summary>why these odds &mdash; "
            f"{len(rows)} angles, sorted by how much the model trusts each"
            f"</summary>{''.join(rows)}</details>")


def render_html(results, weights=None) -> str:
    weights = weights or {}
    parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Fight Predictions</title>"
             f"<link href='https://fonts.googleapis.com/css2?family=Barlow+Condensed:"
             f"ital,wght@0,500;0,700;0,800;1,800&display=swap' rel='stylesheet'>"
             f"<style>{CSS}</style></head><body>"
             f"<div class='topbar'><h1>Fight <span>Predictions</span></h1>"
             f"<span class='gen'>model build {datetime.now():%b %d, %Y}</span></div>"
             f"<div class='wrap'>"]

    for event in dict.fromkeys(r["event"] for r in results):
        ev_rows = [r for r in results if r["event"] == event]
        parts.append(f"<div class='event'><div class='eventhead'>"
                     f"<h2>{esc(event)}</h2></div>"
                     f"<div class='eventsub'><span>{esc(ev_rows[0]['date'])}</span>"
                     f"<span>{esc(ev_rows[0]['location'])}</span></div>")
        for i, r in enumerate(ev_rows):
            main = " main" if i == 0 else ""
            parts.append(f"<div class='fight{main}'>")
            if main:
                parts.append("<div class='mainevent-tag'>Main Event</div>")
            parts.append(f"<div class='wc'>{esc(r.get('weightclass', '') or 'bout')}</div>")
            if r["status"] != "ok":
                parts.append(
                    f"<div class='matchup'>"
                    + corner_html(r['f1'], "", 0, None, "r", False, 0)
                    + "<div class='center'><div class='vs'>VS</div></div>"
                    + corner_html(r['f2'], "", 0, None, "b", False, 0)
                    + f"</div><div class='nodata'>{esc(r['status'])}</div></div>")
                continue
            p1 = r["p1"]
            w1 = p1 >= 0.5
            if not r.get("confident", True):
                w1 = None  # too close to call — no winner highlight
            parts.append(
                "<div class='matchup'>"
                + corner_html(r["f1"], r["rec1"], r["elo1"], None,
                              "r", w1 is True, p1)
                + f"<div class='center'><div class='vs'>VS</div>"
                  f"<div class='bigpct'><span class='r'>{p1 * 100:.0f}</span>"
                  f"<span class='sep'>&ndash;</span>"
                  f"<span class='b'>{(1 - p1) * 100:.0f}</span></div></div>"
                + corner_html(r["f2"], r["rec2"], r["elo2"], None,
                              "b", w1 is False, 1 - p1)
                + "</div>")
            parts.append(
                f"<div class='barwrap'>"
                f"<div class='bar' style='flex-basis:{p1 * 100:.1f}%'></div>"
                f"<div class='bar2' style='flex-basis:{(1 - p1) * 100:.1f}%'></div></div>")
            chips = []
            if not r.get("confident", True):
                chips.append("<span class='chip'>&#9878;&#65039; too close to call</span>")
            if r.get("method"):
                labels = {"dec": "Decision", "ko": "KO/TKO", "sub": "Submission"}
                top = max(r["method"].items(), key=lambda x: x[1])
                chips.append(f"<span class='chip'>likely finish: "
                             f"<b>{labels.get(top[0], top[0])} {top[1]:.0%}</b></span>")
            for src, pa_m in (r.get("market_sources") or {}).items():
                icon = "&#128200;" if src == "polymarket" else "&#127919;"
                chips.append(f"<span class='chip market'>{icon} {esc(src)}: "
                             f"<b>{pa_m:.0%}</b> {esc(r['f1'].split()[-1])}</span>")
            if r.get("market") is not None:
                edge = p1 - r["market"]
                chips.append(f"<span class='chip market'>model edge "
                             f"<b>{edge:+.1%}</b> on {esc(r['f1'].split()[-1])}</span>")
            if chips:
                parts.append(f"<div class='chips'>{''.join(chips)}</div>")
            parts.append(render_breakdown(r, weights))
            parts.append("</div>")
        parts.append("</div>")

    parts.append(
        "<footer>Unofficial fan project — statistical estimates from public fight "
        "data, not betting advice. Fighter images &copy; UFC/Zuffa, loaded from "
        "ufc.com for personal use. Close fights are coin flips; the betting market "
        "has historically been more accurate than this model when they disagree."
        "</footer></div></body></html>")
    return "".join(parts)


def run(market: bool = False, fast: bool = False,
        out: str = "site/index.html", all_events: bool = False):
    """Full site build — callable from pipeline.py without a subprocess."""
    class args:  # keep the body below unchanged
        pass
    args.market, args.fast, args.out, args.all = market, fast, out, all_events

    if not os.path.exists("stats/upcoming.csv"):
        raise SystemExit("no stats/upcoming.csv — run: python -m scrapers.fetch_upcoming")
    upcoming = pd.read_csv("stats/upcoming.csv")
    if not args.all:
        dates = pd.to_datetime(upcoming["DATE"], errors="coerce")
        next_event = upcoming.loc[dates.idxmin(), "EVENT"]
        upcoming = upcoming[upcoming["EVENT"] == next_event]
        print(f"next card only: {next_event} ({len(upcoming)} fights) — use --all for every event")
    else:
        print(f"{len(upcoming)} scheduled fights on {upcoming['EVENT'].nunique()} cards")

    print("fitting models...")
    _, feature_cols, ens, method_model, wc_means = setup_models(args.fast)
    print("building fighter histories...")
    fights, attrs_idx, elo_now, history, past = setup_data()

    print("predicting...")
    results = predict_upcoming(upcoming, feature_cols, ens, method_model, wc_means,
                               fights, attrs_idx, elo_now, history, past, args.market)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render_html(results, ens.weights))

    from web.log_predictions import log_card
    log_card(results)   # scoreboard entry BEFORE the fights happen

    # machine-readable card for the web app (served instantly, no recompute)
    import json
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    with open("stats/upcoming_predictions.json", "w") as jf:
        json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
                   "results": [clean(r) for r in results]}, jf)
    print("saved stats/upcoming_predictions.json (used by the web app)")
    print(f"\nwrote {args.out} — open it in your browser")
    print("after the event: python -m scrapers.scrape && python -m web.log_predictions grade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    run(a.market, a.fast, a.out, a.all)


if __name__ == "__main__":
    main()
