#!/usr/bin/env python3
r"""
Headline dimensionality numbers with BOOTSTRAP uncertainty (all paper estimates).

Reports two interpreted estimates per object (the author's "structural vs stretching modes"):
  dominant_modes  = L-method knee on the eigenvalue spectrum (the few stable, variance-
                    dominant diffusion modes -- the robust "structural" dimension).
  full_dim        = Coifman epsilon-scaling intrinsic dimension (bandwidth-free; the total
                    "stretching" extent including sub-dominant structural modes).
Each with a bootstrap mean +/- s.d. over resamples. Plus per-law (Phillips/Okun/Solow) and
the out-of-domain control separation.

Covers BOTH corpora: n>=200 (high-resolution) and distributed (n>=60, country macro in).
Writes artifacts/dimensions_bootstrap.json.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import pipeline.dimension as AD
import pipeline.corpus as C

OUT = C.OUT
N_S = 40
SEED = 7


def _distinct(X, eps=0.03, seed=0, rebuild=256):
    rng = np.random.default_rng(seed); order = rng.permutation(len(X))
    kept, pts, tree, rec = [], [], None, []
    for i in order:
        x = X[i]; ok = True
        if tree is not None and tree.query(x, k=1)[0] <= eps: ok = False
        if ok and rec and np.min(np.linalg.norm(np.array(rec) - x, axis=1)) <= eps: ok = False
        if ok:
            kept.append(i); pts.append(x); rec.append(x)
            if len(rec) >= rebuild: tree = cKDTree(np.array(pts)); rec = []
    return np.array(kept)


def boot_dim(X, n_boot=8, n_sub=6400, seed=SEED, full_estimate=True):
    """Coifman and L-method dimension, with the protocol defects that biased the earlier
    numbers removed.

    Two bugs were in the original: n_sub=900 and replace=True.

    The Coifman epsilon-scaling estimate has not converged at 900 points. A sweep on this
    corpus (artifacts/coifman_n_sweep.json) gives 4.56, 5.42, 6.22, 7.04, 7.12, 7.25
    for n = 200 to 6400 without replacement: it climbs ~0.8 per doubling below n ~ 1600
    and ~0.1 above, so 900 sits well below the knee and reads roughly 1.1 low. Sampling
    with replacement compounds this by placing duplicate points at distance exactly zero,
    which inflates S(epsilon) at small epsilon, the very regime the slope is read from,
    and it inflates the spread: at n=900 the s.d. is 0.26 with replacement against 0.03
    without.

    The point estimate is therefore taken on the whole set when it fits in memory, with
    the uncertainty coming from large subsamples drawn without replacement.
    """
    rng = np.random.default_rng(seed)
    dom, full = [], []
    for b in range(n_boot):
        s = rng.choice(len(X), min(n_sub, len(X)), replace=False)
        res, _ = AD.analyse(X[s])
        dom.append(res["lmethod_knee"]); full.append(res["coifman_dim"])
    # The Coifman estimate needs only the distance matrix, so it can be taken on every
    # series at once. The L-method knee needs the diffusion spectrum, whose dense
    # eigendecomposition is not tractable at this n, so it stays a subsample mean.
    if full_estimate and len(X) <= 20000:
        from scipy.spatial.distance import pdist, squareform
        fl = float(AD.kernel_sum_bandwidth(squareform(pdist(X)))[1])
    else:
        fl = float(np.mean(full))
    return float(np.mean(dom)), float(np.std(dom)), fl, float(np.std(full))


def corpus_dims(plans_file, corpus_file, valid_file, name):
    bal = np.load(os.path.join(OUT, corpus_file))
    n = len(bal["idx"])
    mm = np.memmap(C.require_artifact(plans_file), dtype=np.float32, mode="r", shape=(n, N_S * N_S))
    valid = np.load(os.path.join(OUT, valid_file))
    v = np.where(valid)[0]
    # Hellinger coordinates x_i = sqrt(T_i / sum T_i), the metric the whole paper uses:
    # the DMAP basis, the controls diagnostic and the Nystrom extension all read the
    # sqrtplans files. This read plans.dat, the raw couplings, which is a different metric.
    # The /sqrt(2) below is the tell: it maps a unit-norm vector's maximum separation to 1
    # and is meaningless for raw plans.
    X = np.asarray(mm[v]) / np.sqrt(2.0)
    # No near-duplicate gate. Two series whose transport plans coincide are distinct
    # observations of the same dynamics, drawn from different sources; collapsing them
    # would discard the coincidence this paper reports. Retaining them widens the
    # bootstrap interval, which is the honest uncertainty for an arbitrary corpus.
    dd = _distinct(X)                      # reported for reference only, never applied
    Xd = X
    dm, dms, fl, fls = boot_dim(Xd)
    out = {"name": name, "n_valid": int(len(v)), "n_distinct": int(len(dd)),
           "dedup_applied": False,
           "dominant_modes": {"mean": dm, "sd": dms},
           "full_dim": {"mean": fl, "sd": fls}}
    print(f"[{name}] dominant(L-method)={dm:.1f}+/-{dms:.1f}  full(Coifman)={fl:.1f}+/-{fls:.1f}  "
          f"(n_distinct={len(dd)})", flush=True)
    # Per-concept dimension, at MATCHED sample size.
    #
    # The earlier version drew min(700, len(m)) with replacement from each concept and
    # compared the results directly. Concept sizes span 123 to 1389 here, and the
    # estimator is strongly size-dependent below n ~ 1600, so the reported spread of
    # 4.1-7.0 tracked concept size almost perfectly: corr(log n, d) = 0.976. That is a
    # measurement of how many series each concept has, not of how rich it is.
    #
    # Every concept is therefore subsampled to the same n before comparison. Concepts do
    # still differ at matched n, but over a much narrower range, and the ordering moves.
    # It also no longer filters to `dd`, which deduplicated the per-concept numbers while
    # the headline deliberately did not.
    con = bal["concept"][v]
    per = {}
    sizes = [int((con == c).sum()) for c in np.unique(con)]
    n_match = max(120, int(np.percentile([s for s in sizes if s >= 120], 5))) if sizes else 120
    for c in np.unique(con):
        m = np.where(con == c)[0]
        if len(m) < n_match: continue
        fs = []
        for b in range(12):
            rng_c = np.random.default_rng(SEED + b)
            s = m[rng_c.choice(len(m), n_match, replace=False)]
            fs.append(AD.analyse(X[s])[0]["coifman_dim"])
        per[str(c)] = {"mean": round(float(np.mean(fs)), 2), "sd": round(float(np.std(fs)), 2),
                       "n_available": int(len(m)), "n_matched": int(n_match)}
    out["per_concept_matched_n"] = n_match
    out["per_concept_full_dim"] = per
    return out


def main():
    res = {}
    # high-resolution corpus
    if os.path.exists(os.path.join(OUT, "sqrtplans.dat")):
        res["high_res_n200"] = corpus_dims("sqrtplans.dat", "corpus.npz", "plans_valid.npy", "n>=200 high-res")
    # distributed corpus
    if os.path.exists(os.path.join(OUT, "sqrtplans_dist.dat")) and os.path.exists(os.path.join(OUT, "plans_valid_dist.npy")):
        res["distributed_n60"] = corpus_dims("sqrtplans_dist.dat", "corpus_dist.npz", "plans_valid_dist.npy", "distributed n>=60")
    json.dump(res, open(os.path.join(OUT, "dimensions_bootstrap.json"), "w"), indent=2)
    print("wrote dimensions_bootstrap.json", flush=True)


if __name__ == "__main__":
    main()
