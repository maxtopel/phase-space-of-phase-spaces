#!/usr/bin/env python3
r"""
Distributed-corpus PSoPS: single vs multi barycenter + the rate-degeneracy test.

Builds the PSoPS on the balanced distributed macro corpus (the distributed corpus builder in the corpus module:
n>=60 admits World Bank / Penn World Table country-level GDP/labour/inflation/TFP, concept-
capped so ECB rates can't dominate). Then:

  SINGLE   one universal barycenter -> PSoPS -> robust dimension (Coifman/spectral).
  MULTI    per-concept barycenters -> coupling structure (which concepts couple).
  DEGENERACY TEST  the hypothesis: on a DISTRIBUTED corpus the European
           interest-rate data should be DEGENERATE -- it projects onto only a few PSoPS
           dimensions (low local intrinsic dim) rather than defining the manifold, while
           the diverse real-activity data spans the rest.

Writes artifacts/distributed_results.json.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
import pipeline.bridge as V3
import pipeline.dimension as AD
import pipeline.corpus as C

OUT = C.OUT
N_S = 40
SEED = 7


def _plan_chunk(args):
    idx_chunk, Bflat = args
    arr = np.load(C.COSTS, mmap_mode="r")
    B = Bflat.reshape(N_S, N_S)
    out = np.zeros((len(idx_chunk), N_S * N_S), np.float32)
    ok = np.zeros(len(idx_chunk), bool)
    costs, loc = [], []
    for j, i in enumerate(idx_chunk):
        M = np.asarray(arr[i], np.float64)
        if np.isfinite(M).all() and M.max() > C.MIN_COSTMAX:
            costs.append(V3.normalize_cost(M)); loc.append(j)
    if costs:
        for j, T in zip(loc, V3.plans_to(costs, B)):
            T = np.asarray(T, np.float64)
            if np.isfinite(T).all() and T.sum() > 0:
                out[j] = (T / T.sum()).ravel().astype(np.float32); ok[j] = True
    return idx_chunk, out, ok


def build_plans():
    bal = np.load(os.path.join(OUT, "corpus_dist.npz"))
    idx = bal["idx"]; n = len(idx)
    arr = np.load(C.COSTS, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    # barycenter on a CONCEPT-BALANCED landmark sample (so it reflects diverse macro, not rates)
    con = bal["concept"]; land = []
    for c in np.unique(con):
        ix = idx[con == c]
        land.extend(rng.choice(ix, min(120, len(ix)), replace=False).tolist())
    lc = [V3.normalize_cost(np.asarray(arr[i], float)) for i in land
          if np.isfinite(arr[i]).all() and np.asarray(arr[i]).max() > C.MIN_COSTMAX]
    B = V3.gw_barycenter_nonentropic(lc[:1500])
    np.save(os.path.join(OUT, "barycenter_dist.npy"), B)
    print(f"distributed barycenter on {len(lc[:1500])} concept-balanced series", flush=True)
    mm = np.memmap(os.path.join(OUT, "plans_dist.dat"), dtype=np.float32, mode="w+", shape=(n, N_S * N_S))
    done = np.zeros(n, bool)
    chunks = [np.arange(s, min(s + 200, n)) for s in range(0, n, 200)]
    t0 = time.time(); comp = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_plan_chunk, (idx[c].tolist(), B.ravel())): c for c in chunks}
        for fut in futs:
            c = futs[fut]; _, out, ok = fut.result()
            mm[c] = out; done[c] = ok; comp += len(c)
            if comp % 4000 < 200:
                r = comp / max(time.time() - t0, 1e-9)
                print(f"  plans {comp}/{n} ETA {(n-comp)/max(r,1e-9)/60:.0f}m", flush=True)
    mm.flush()
    valid = (np.abs(mm).sum(1) > 0)
    np.save(os.path.join(OUT, "plans_valid_dist.npy"), valid)
    print(f"DONE distributed plans: {valid.sum()}/{n} valid", flush=True)


def analyse():
    bal = np.load(os.path.join(OUT, "corpus_dist.npz"))
    n = len(bal["idx"])
    mm = np.memmap(os.path.join(OUT, "plans_dist.dat"), dtype=np.float32, mode="r", shape=(n, N_S * N_S))
    valid = np.load(os.path.join(OUT, "plans_valid_dist.npy"))
    con = bal["concept"]; ind = bal["indicator"]
    v = np.where(valid)[0]
    X = np.asarray(mm[v]) / np.sqrt(2.0)
    rng = np.random.default_rng(SEED)

    # dedup near-duplicates (distinct series) for a fair global dimension
    tree = cKDTree(X); dk, _ = tree.query(X, k=2, workers=-1)
    keep = []
    # greedy covering at the typical-noise scale
    order = rng.permutation(len(X)); kept, pts, t2, rec = [], [], None, []
    for i in order:
        x = X[i]; ok = True
        if t2 is not None and t2.query(x, k=1)[0] <= 0.03: ok = False
        if ok and rec and np.min(np.linalg.norm(np.array(rec) - x, axis=1)) <= 0.03: ok = False
        if ok:
            kept.append(i); pts.append(x); rec.append(x)
            if len(rec) >= 256: t2 = cKDTree(np.array(pts)); rec = []
    distinct = np.array(kept)
    print(f"distinct (deduped) distributed series: {len(distinct)}", flush=True)

    # SINGLE global dimension
    s = distinct if len(distinct) <= 1800 else rng.choice(distinct, 1800, replace=False)
    res_glob, _ = AD.analyse(X[s], label="distributed_global")
    print(f"[SINGLE] global Coifman dim={res_glob['coifman_dim']:.2f}  spectral95={res_glob['spectral_dim_95var']}  "
          f"L-knee={res_glob['lmethod_knee']}", flush=True)

    # DEGENERACY TEST: local dimension of the rates cluster vs the real-activity cluster
    def local_dim(mask_name, members):
        m = members[np.isin(members, distinct)]
        if len(m) < 80: return None
        m = m if len(m) <= 1200 else rng.choice(m, 1200, replace=False)
        r, _ = AD.analyse(X[m], label=mask_name)
        return r["coifman_dim"]
    rates = v[np.isin(con[v], ["interest_rate"])]
    realact = v[np.isin(ind[v], ["gdp", "labor", "inflation", "unemployment", "tfp", "capital"])]
    fx = v[np.isin(con[v], ["exchange_rate"])]
    d_rates = local_dim("rates", rates)
    d_real = local_dim("real_activity", realact)
    d_fx = local_dim("fx", fx)
    print(f"[DEGENERACY] interest-rate cluster Coifman dim = {d_rates}  (LOW => degenerate, as hypothesised)", flush=True)
    print(f"             real-activity cluster   Coifman dim = {d_real}", flush=True)
    print(f"             exchange-rate cluster   Coifman dim = {d_fx}", flush=True)

    # MULTI: per-concept sub-manifold dimension (which concepts are simple vs rich)
    per = {}
    for c in np.unique(con):
        m = v[con[v] == c]; m = m[np.isin(m, distinct)]
        if len(m) < 80: continue
        m = m if len(m) <= 1000 else rng.choice(m, 1000, replace=False)
        r, _ = AD.analyse(X[m], label=c)
        per[str(c)] = round(r["coifman_dim"], 2)
    print(f"[MULTI] per-concept Coifman dim: {per}", flush=True)

    out = {"n_distinct": int(len(distinct)),
           "single_global": {k: res_glob[k] for k in ["coifman_dim", "spectral_dim_95var", "lmethod_knee", "eigenvalues"]},
           "degeneracy": {"rates_dim": d_rates, "real_activity_dim": d_real, "fx_dim": d_fx},
           "per_concept_dim": per}
    json.dump(out, open(os.path.join(OUT, "distributed_results.json"), "w"), indent=2)
    print("wrote distributed_results.json", flush=True)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("plans", "all"):
        build_plans()
    if stage in ("analyse", "all"):
        analyse()
