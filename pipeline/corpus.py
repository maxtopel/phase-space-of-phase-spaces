#!/usr/bin/env python3
r"""
Mechanical, reproducible construction of the universal NON-EUROSTAT MACRO corpus.

This is the only bespoke step in the universal-PSoPS pipeline (everything downstream
applies the V3 functions via the v3 bridge). The rules are deterministic so the
corpus is a referee-rerunnable artifact, not hand-curation:

  R1  source: non-eurostat ECONOMIC sources only. The non-economic series
      (who/noaa/rivers) are set aside as OUT-OF-DOMAIN FALSIFICATION CONTROLS.
      (Eurostat -- 88% of the cache, granular regional/sectoral -- is excluded from
       the universal core and re-attached later as a Nystrom extension: a feature.)
  R2  quality: n_points >= MIN_POINTS and a non-degenerate cost matrix.
  R3  frequency: macro frequencies only (monthly/quarterly/annual; drop daily/weekly
      market microstructure). Then ONE record per underlying series key -- the
      best-resolved (most observations) macro-frequency variant -- removing the
      timescale-variant axis that otherwise inflates intrinsic dimension.
  R4  indicator: KEEP 'other' (ECB monetary aggregates + OECD composite indicators
      are the macro connective tissue; excluding them leaves a finance-only corpus).

Reads the per-series 40x40 GW cost-matrix cache + a compact labels file; writes
artifacts/corpus.npz (indices into the cache + per-series labels) and
corpus_report.json (the composition table).
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np
from collections import Counter

import os as _os
from pathlib import Path as _Path
# stage-1 cost-matrix cache: private, env-configurable; not needed for any
# paper figure (see README, "The PSoPS construction chain")
DM = _os.environ.get("AOA_DM_CACHE", str(_Path.home() / "Desktop" / "atlas" /
     "research" / "core" / "AoA" / "checkpoints" / "family_discovery" / "dm_cache"))
COSTS = os.path.join(DM, "dist_matrices_453k.npy")


def require_artifact(fname):
    """Fail with a Zenodo pointer instead of a bare traceback for the big plan binaries."""
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} is a transport-plan binary deposited on Zenodo (DOI in the README); "
            f"download it into {os.path.basename(OUT)}/ to run this script.")
    return p


def require_cache():
    """Fail with a pointer instead of a bare traceback when the stage-1 cache is absent."""
    if not os.path.exists(COSTS):
        raise FileNotFoundError(
            f"{COSTS} is the stage-1 cost-matrix cache, which is not distributed "
            "(available on reasonable request; see README, 'The PSoPS construction chain').")
    return COSTS
LABELS = os.path.join(DM, "labels.npz")
OUT = os.path.join(_ROOT, "artifacts")
os.makedirs(OUT, exist_ok=True)

MIN_POINTS = 200
MIN_COSTMAX = 1e-3
MACRO_FREQ = ("annual", "quarterly", "monthly")
CONTROL_SRC = ("who", "noaa", "rivers")

# --- distributed-corpus variant (resolves the ECB/rates domination) ---
# n>=60 admits the annual country-level macro (World Bank GDP/labour/inflation, Penn World
# Table TFP/capital) that n>=200 deleted; a per-concept cap then prevents any one family
# (ECB rates/aggregates) from defining the manifold -- they become a degenerate cluster, not
# the geometry. This is the referee-defensible "geometry of the macroeconomy" corpus.
MIN_POINTS_DIST = 60
CONCEPT_CAP_DIST = 0.12        # no concept exceeds this fraction of the distributed corpus


def eurostat_substream(key):
    body = key.split("/", 1)[-1].replace("eurostat_", "", 1)
    toks = body.split("_")
    return "_".join(toks[:2]) if len(toks) >= 2 else (toks[0] if toks else "?")


def concept_of(src_i, ind_i):
    """Named indicator for labeled series; source-level pseudo-concept (ecb-other,
    oecd-other, ...) for unlabeled 'other' so monetary/composite families are
    treated as their own capped concepts rather than one blob."""
    return ind_i if ind_i != "other" else f"{src_i}-other"


def cost_maxes():
    cache = os.path.join(OUT, "cost_max.npy")
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.load(COSTS, mmap_mode="r")
    n = arr.shape[0]
    mx = np.empty(n, np.float32)
    for s in range(0, n, 20000):
        e = min(s + 20000, n)
        mx[s:e] = np.asarray(arr[s:e]).reshape(e - s, -1).max(1)
    np.save(cache, mx)
    return mx


def build_corpus():
    z = np.load(LABELS, allow_pickle=True)
    src, ind, ts = z["source"], z["indicator"], z["timescale"]
    npts, keys = z["n_points"], z["key"]
    mx = cost_maxes()
    N = len(src)

    econ = ~np.isin(src, np.array(CONTROL_SRC + ("eurostat",)))
    qual = (npts >= MIN_POINTS) & (mx > MIN_COSTMAX)
    macro = np.isin(ts, np.array(MACRO_FREQ))
    base = econ & qual & macro
    print(f"R1-R3 (non-eurostat econ, n>={MIN_POINTS}, non-degenerate, macro-freq): {base.sum()}", flush=True)

    best = {}
    for i in np.where(base)[0]:
        k = keys[i]
        if k not in best or npts[i] > npts[best[k]]:
            best[k] = i
    chosen = np.array(sorted(best.values()))
    print(f"one-frequency-per-series dedup: {len(chosen)} distinct series", flush=True)

    concept = np.array([concept_of(src[i], ind[i]) for i in chosen])
    comp_src = Counter(src[chosen].tolist())
    comp_con = Counter(concept.tolist())
    comp_ts = Counter(ts[chosen].tolist())
    print("  by source:", dict(comp_src.most_common(12)), flush=True)
    print("  by concept:", dict(comp_con.most_common(14)), flush=True)
    print("  by timescale:", dict(comp_ts.most_common()), flush=True)

    # falsification controls (one freq per key)
    cmask = np.isin(src, np.array(CONTROL_SRC)) & qual & macro
    cbest = {}
    for i in np.where(cmask)[0]:
        k = keys[i]
        if k not in cbest or npts[i] > npts[cbest[k]]:
            cbest[k] = i
    ctrl = np.array(sorted(cbest.values()))
    print(f"controls (who/noaa/rivers): {len(ctrl)}", flush=True)

    # all-frequency non-eurostat (kept WITH timescale variants) for the timescale analysis
    allfreq = econ & qual & np.isin(ts, np.array(MACRO_FREQ))
    af_idx = np.where(allfreq)[0]

    np.savez(os.path.join(OUT, "corpus.npz"),
             idx=chosen, source=src[chosen], indicator=ind[chosen],
             concept=concept, timescale=ts[chosen], n_points=npts[chosen],
             ctrl_idx=ctrl, ctrl_source=src[ctrl],
             allfreq_idx=af_idx, allfreq_timescale=ts[af_idx],
             allfreq_source=src[af_idx], allfreq_concept=np.array([concept_of(src[i], ind[i]) for i in af_idx]))
    json.dump({"n": int(len(chosen)), "n_controls": int(len(ctrl)), "n_allfreq": int(len(af_idx)),
               "by_source": dict(comp_src), "by_concept": dict(comp_con),
               "by_timescale": dict(comp_ts), "min_points": MIN_POINTS,
               "macro_freq": list(MACRO_FREQ)},
              open(os.path.join(OUT, "corpus_report.json"), "w"), indent=2)
    return chosen


def build_corpus_distributed():
    """Distributed macro corpus: n>=60 (admits annual country macro), concept-capped so
    rates/ECB cannot dominate. Writes corpus_dist.npz / corpus_dist_report.json."""
    z = np.load(LABELS, allow_pickle=True)
    src, ind, ts = z["source"], z["indicator"], z["timescale"]
    npts, keys = z["n_points"], z["key"]
    mx = cost_maxes()
    econ = ~np.isin(src, np.array(CONTROL_SRC + ("eurostat",)))
    qual = (npts >= MIN_POINTS_DIST) & (mx > MIN_COSTMAX)
    macro = np.isin(ts, np.array(MACRO_FREQ))
    base = econ & qual & macro
    best = {}
    for i in np.where(base)[0]:
        k = keys[i]
        if k not in best or npts[i] > npts[best[k]]:
            best[k] = i
    pool = np.array(sorted(best.values()))
    concept = np.array([concept_of(src[i], ind[i]) for i in pool])
    # per-concept cap
    rng = np.random.default_rng(7)
    cap = int(len(pool) * CONCEPT_CAP_DIST)
    keep = []
    g = {}
    for p, c in zip(pool, concept):
        g.setdefault(c, []).append(p)
    for c, members in g.items():
        m = np.array(members)
        keep.extend((rng.choice(m, cap, replace=False) if len(m) > cap else m).tolist())
    chosen = np.array(sorted(keep))
    concept = np.array([concept_of(src[i], ind[i]) for i in chosen])
    print(f"distributed corpus: {len(chosen)} series (n>={MIN_POINTS_DIST}, concept cap {CONCEPT_CAP_DIST:.0%})", flush=True)
    print("  by source:", dict(Counter(src[chosen].tolist()).most_common(12)), flush=True)
    print("  by indicator:", dict(Counter(ind[chosen].tolist()).most_common()), flush=True)
    print("  real-activity (gdp+labor+inflation+unemployment+capital+tfp+trade): "
          f"{np.isin(ind[chosen], ['gdp','labor','inflation','unemployment','capital','tfp','trade']).sum()}", flush=True)
    np.savez(os.path.join(OUT, "corpus_dist.npz"),
             idx=chosen, source=src[chosen], indicator=ind[chosen],
             concept=concept, timescale=ts[chosen], n_points=npts[chosen])
    json.dump({"n": int(len(chosen)), "by_source": dict(Counter(src[chosen].tolist())),
               "by_indicator": dict(Counter(ind[chosen].tolist())), "min_points": MIN_POINTS_DIST,
               "concept_cap": CONCEPT_CAP_DIST}, open(os.path.join(OUT, "corpus_dist_report.json"), "w"), indent=2)
    return chosen


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "distributed":
        build_corpus_distributed()
    else:
        build_corpus()
