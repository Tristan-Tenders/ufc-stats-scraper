"""Live prediction web app — API + interactive frontend.

  pip install fastapi uvicorn
  python -m web.app                # http://localhost:8000
  UFC_FAST=1 python -m web.app     # quicker startup (skips OOF calibration)

Models fit once at startup and are cached to stats/model_cache.pkl
(invalidated automatically when train_dataset.csv changes), so restarts
are instant. Endpoints:

  GET /api/fighters?q=mak          fuzzy name search
  GET /api/fighter/Islam Makhachev current stats snapshot
  GET /api/predict?a=..&b=..[&market=1]
  GET /api/upcoming[?market=1]     next card with predictions
"""

import difflib
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import HistGradientBoostingClassifier

from core import build_dataset as bd
from core import predict as P
from core.strategies import StrategyEnsemble
from core.utils import META_COLS

CACHE = "stats/model_cache.pkl"
DATASET = "stats/train_dataset.csv"


class Engine:
    def __init__(self):
        fast = bool(os.environ.get("UFC_FAST"))
        if self._load_cache():
            print("engine: loaded from cache")
        else:
            print(f"engine: fitting models ({'fast' if fast else 'full OOF + calibration'})...")
            self._fit(fast)
            self._save_cache()

        print("engine: building fighter histories...")
        fights = bd.load_fights()
        totals = bd.load_fight_totals()
        self.fights = fights
        self.attrs_idx = bd.load_fighter_attrs().set_index("fighter_key")
        elo_pre, self.elo_now = bd.compute_elo(fights)
        self.history = bd.build_career_history(fights, totals, elo_pre=elo_pre,
                                                teams=bd.load_teams())
        self.past = P.build_past_results(self.history)
        self.all_keys = sorted(set(self.history["fighter_key"]))
        names = pd.concat([fights["f1"], fights["f2"]]).dropna().unique()
        self.display_names = {bd.norm_name(n): n for n in sorted(names)}
        self.images = self._load_json("stats/fighter_images.json")
        print("engine ready.")

    # ---------- model cache ----------
    def _cache_valid(self):
        return (os.path.exists(CACHE) and os.path.exists(DATASET)
                and os.path.getmtime(CACHE) > os.path.getmtime(DATASET))

    def _load_cache(self):
        if not self._cache_valid():
            return False
        try:
            with open(CACHE, "rb") as f:
                (self.ens, self.method_model, self.feature_cols,
                 self.wc_means) = pickle.load(f)
            return True
        except Exception:
            return False

    def _save_cache(self):
        with open(CACHE, "wb") as f:
            pickle.dump((self.ens, self.method_model, self.feature_cols,
                         self.wc_means), f)

    def _fit(self, fast: bool):
        df = pd.read_csv(DATASET, parse_dates=["date"])
        self.feature_cols = [c for c in df.columns if c not in META_COLS]
        y = df["label"].to_numpy(dtype=int)
        self.ens = StrategyEnsemble(self.feature_cols)
        aux = df[["sig_strike_diff", "fight_secs"]] \
            if "sig_strike_diff" in df.columns else None
        (self.ens.fit_fast if fast else self.ens.fit)(
            df[self.feature_cols], y, dates=df["date"], aux_targets=aux)
        self.method_model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
        self.method_model.fit(df[self.feature_cols].to_numpy(dtype=float),
                              df["method"].to_numpy())
        self.wc_means = {}
        for feat in bd.WC_REL_FEATURES:
            pooled = pd.concat([df[["wc_lbs"]].assign(v=df[f"a_{feat}"]),
                                df[["wc_lbs"]].assign(v=df[f"b_{feat}"])])
            for wc, m in pooled.groupby("wc_lbs")["v"].mean().items():
                self.wc_means[(wc, feat)] = m

    @staticmethod
    def _load_json(path):
        if os.path.exists(path):
            import json
            with open(path) as f:
                return json.load(f)
        return {}

    # ---------- queries ----------
    def resolve(self, name: str) -> str:
        key = bd.norm_name(name)
        if key in self.display_names:
            return key
        m = difflib.get_close_matches(key, self.all_keys, n=1, cutoff=0.6)
        if not m:
            raise HTTPException(404, f"fighter not found: {name}")
        return m[0]

    def search(self, q: str, limit=8) -> list[dict]:
        qn = bd.norm_name(q)
        hits = [k for k in self.all_keys if qn in k][:limit]
        if len(hits) < limit:
            hits += [k for k in difflib.get_close_matches(qn, self.all_keys, n=limit)
                     if k not in hits]
        return [self.fighter_card(k) for k in hits[:limit]]

    def snapshot(self, key: str) -> dict:
        today = pd.Timestamp(datetime.now().date())
        s = P.fighter_snapshot(self.history, self.attrs_idx, self.elo_now, key, today)
        if s is None:
            raise HTTPException(404, "no fight history")
        return s

    def fighter_card(self, key: str) -> dict:
        s = self.snapshot(key)
        name = self.display_names.get(key, key.title())
        clean = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else
                     round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in s.items()}
        return {"name": name, "key": key,
                "record": f"{int(s['prior_wins'])}-{int(s['prior_losses'])}",
                "elo": round(float(s["elo"])),
                "image": self.images.get(name), "stats": clean}

    def predict(self, name_a: str, name_b: str, market: bool) -> dict:
        ka, kb = self.resolve(name_a), self.resolve(name_b)
        sa, sb = self.snapshot(ka), self.snapshot(kb)
        h2h_a, h2h_b, n_common, edge_ab = P.matchup_features(self.past, ka, kb)
        sa["h2h_wins"], sb["h2h_wins"] = h2h_a, h2h_b
        wc_str = P.latest_weightclass(self.fights, ka) or \
            P.latest_weightclass(self.fights, kb)
        wc_lbs, is_women, _ = bd.parse_weightclass(wc_str)
        base = {"wc_lbs": wc_lbs, "is_women": is_women, "is_title": 0.0,
                "is_apex": 0.0, "n_common": n_common}
        p, per_strategy, method_probs, confident = P.predict_fight(
            self.ens, self.feature_cols, sa, sb,
            dict(base, common_edge=edge_ab),
            dict(base, common_edge=-edge_ab if pd.notna(edge_ab) else np.nan),
            self.wc_means, self.method_model)

        out = {"a": self.fighter_card(ka), "b": self.fighter_card(kb),
               "p_a": round(float(p), 4),
               "weightclass": wc_str,
               "strategies": {k: round(float(v), 4) for k, v in per_strategy.items()},
               "weights": {k: round(float(v), 3) for k, v in self.ens.weights.items()},
               "method": {k: round(float(v), 3) for k, v in (method_probs or {}).items()},
               "confident": confident,
               "calibrated": getattr(self.ens, "calibrator", None) is not None}
        if market:
            from scrapers.fetch_market_odds import get_market_probs
            srcs = get_market_probs(out["a"]["name"], out["b"]["name"])
            out["market"] = {s: round(v[0], 4) for s, v in srcs.items()} if srcs else {}
        return out


app = FastAPI(title="UFC Fight Predictions")
engine: Engine | None = None


@app.on_event("startup")
def _startup():
    global engine
    engine = Engine()


@app.get("/api/fighters")
def fighters(q: str):
    return engine.search(q)


@app.get("/api/fighter/{name}")
def fighter(name: str):
    return engine.fighter_card(engine.resolve(name))


@app.get("/api/predict")
def api_predict(a: str, b: str, market: int = 0):
    return engine.predict(a, b, bool(market))


CARD_JSON = "stats/upcoming_predictions.json"


def _row_to_api(r: dict) -> dict:
    """Convert a saved make_site prediction row into the /api/predict shape."""
    def card(name, rec, elo):
        return {"name": name, "record": rec or "",
                "elo": round(elo) if elo else None,
                "image": engine.images.get(name) if engine else None}
    if r.get("status") != "ok":
        return {"a": {"name": r["f1"]}, "b": {"name": r["f2"]},
                "p_a": None, "error": r.get("status", "no data")}
    out = {"a": card(r["f1"], r.get("rec1"), r.get("elo1")),
           "b": card(r["f2"], r.get("rec2"), r.get("elo2")),
           "p_a": r["p1"], "weightclass": r.get("weightclass", ""),
           "strategies": r.get("strategies", {}),
           "weights": engine.ens.weights if engine else {},
           "method": r.get("method") or {},
           "confident": r.get("confident", True),
           "calibrated": engine is not None and
           getattr(engine.ens, "calibrator", None) is not None}
    if r.get("market") is not None:
        out["market"] = r.get("market_sources") or {"card": r["market"]}
    return out


@app.get("/api/upcoming")
def upcoming(market: int = 0):
    """Serve the PRECOMPUTED card (from pipeline.py / make_site.py) when
    fresh — instant, consistent with the static site, and already logged to
    the prediction log. Falls back to live computation if it's stale."""
    if not os.path.exists("stats/upcoming.csv"):
        raise HTTPException(404, "run fetch_upcoming.py (or pipeline.py) first")

    if (os.path.exists(CARD_JSON) and
            os.path.getmtime(CARD_JSON) >= os.path.getmtime("stats/upcoming.csv")):
        import json
        with open(CARD_JSON) as f:
            saved = json.load(f)
        rows = saved.get("results", [])
        if rows:
            return {"event": rows[0]["event"], "date": rows[0]["date"],
                    "location": rows[0].get("location", ""),
                    "precomputed": saved.get("generated_at"),
                    "fights": [_row_to_api(r) for r in rows]}

    # stale or missing -> live compute (custom matchups always use the API)
    up = pd.read_csv("stats/upcoming.csv")
    dates = pd.to_datetime(up["DATE"], errors="coerce")
    next_event = up.loc[dates.idxmin(), "EVENT"]
    up = up[up["EVENT"] == next_event]
    fights = []
    for _, f in up.iterrows():
        try:
            fights.append(engine.predict(f["F1"], f["F2"], bool(market)))
        except HTTPException:
            fights.append({"a": {"name": f["F1"]}, "b": {"name": f["F2"]},
                           "p_a": None, "error": "no data (debut?)"})
    return {"event": next_event, "date": up.iloc[0]["DATE"],
            "location": up.iloc[0].get("LOCATION", ""), "fights": fights}


@app.get("/")
def index():
    return FileResponse("site/app.html")


if os.path.isdir("site/img"):
    app.mount("/img", StaticFiles(directory="site/img"), name="img")


if __name__ == "__main__":
    import socket
    import uvicorn
    port = int(os.environ.get("UFC_PORT", 8000))
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", candidate)) != 0:
                if candidate != port:
                    print(f"[~] port {port} busy — using {candidate} "
                          f"(free it with: lsof -ti :{port} | xargs kill)")
                port = candidate
                break
    uvicorn.run(app, host="127.0.0.1", port=port)
