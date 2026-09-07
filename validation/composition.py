#!/usr/bin/env python3
r"""Is the PSoPS dimension actually invariant to corpus composition?

An earlier draft claimed rate-only, monetary, real-activity and balanced sub-corpora all
return d_eff ~ 6. Two problems with the evidence as it stood. The stored artifact
(distributed_results.json) actually reports 7.80, 7.68 and 7.39 for rates, real activity
and FX, not ~6, so the quoted number does not match its own source. And those values came
through the same defects as the headline: raw-plan coordinates instead of Hellinger, and
a silent filter to the deduplicated index set.

More basic than either: the Coifman estimate is strongly sample-size dependent below
n ~ 1600 (artifacts/convergence_study.json), and the sub-corpora have very different
sizes. Comparing them at their natural sizes measures how many series each contains. Any
honest invariance test has to hold n fixed.

This re-runs the comparison in Hellinger coordinates, without deduplication, with every
sub-corpus subsampled to a common n, against the full corpus drawn at the same n.

Writes artifacts/composition_invariance.json
Run    python3 validation/composition.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import gc
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform

import pipeline.dimension as AD
import pipeline.corpus as C

OUT = C.OUT
NS, SEED, REPS = 40, 0, 6

# the groupings the paper names, defined exactly as pipeline/barycenter_plans.py defines them
REAL_ACTIVITY = ("gdp", "labor", "inflation", "unemployment", "tfp", "capital")
MONETARY = ("interest_rate", "credit")


def coifman(X):
    return float(AD.kernel_sum_bandwidth(squareform(pdist(X)))[1])


def main():
    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    v = np.sort(np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0])
    S = np.memmap(C.require_artifact("sqrtplans_dist.dat"), dtype=np.float32, mode="r",
                  shape=(len(z["idx"]), NS * NS))
    con = np.asarray(z["concept"])[v]
    ind = np.asarray(z["indicator"])[v]

    groups = {
        "Full corpus": np.arange(len(v)),
        "Rate-only": np.where(con == "interest_rate")[0],
        "Monetary": np.where(np.isin(con, MONETARY))[0],
        "Real activity": np.where(np.isin(ind, REAL_ACTIVITY))[0],
        "Exchange rate": np.where(con == "exchange_rate")[0],
    }

    # concept-balanced: equal draw from every concept large enough to supply one
    rng = np.random.default_rng(SEED)
    n_match = min(len(g) for k, g in groups.items() if k != "Full corpus")
    n_match = int(min(2000, n_match))
    concepts = [c for c in np.unique(con) if (con == c).sum() >= 40]
    # draw equally from every concept, topping up from the largest ones if the equal
    # share leaves the group short of n_match
    per_c = max(1, int(np.ceil(n_match / len(concepts))))
    bal = np.concatenate([rng.choice(np.where(con == c)[0], min(per_c, int((con == c).sum())),
                                     replace=False) for c in concepts])
    if len(bal) < n_match:
        spare = np.setdiff1d(np.arange(len(v)), bal)
        bal = np.concatenate([bal, rng.choice(spare, n_match - len(bal), replace=False)])
    groups["Concept-balanced"] = bal

    print(f"  every group subsampled to n={n_match}, {REPS} draws, no replacement,")
    print(f"  Hellinger coordinates, no deduplication\n")
    print(f"  {'sub-corpus':18s} {'available':>9}   d_eff at matched n")
    out = {"n_matched": n_match, "reps": REPS, "groups": {}}
    for name, g in groups.items():
        if len(g) < n_match:
            print(f"  {name:18s} {len(g):9d}   too small, skipped")
            continue
        vals = []
        for r in range(REPS):
            rr = np.random.default_rng(SEED + r)
            X = np.asarray(S[v[np.sort(rr.choice(g, n_match, replace=False))]], np.float64)
            vals.append(coifman(X))
            del X
            gc.collect()
        out["groups"][name] = {"n_available": int(len(g)), "mean": float(np.mean(vals)),
                               "sd": float(np.std(vals))}
        print(f"  {name:18s} {len(g):9d}   {np.mean(vals):5.2f} +/- {np.std(vals):4.2f}",
              flush=True)

    ds = np.array([r["mean"] for r in out["groups"].values()])
    out["spread"] = float(ds.max() - ds.min())
    out["relative_spread"] = float((ds.max() - ds.min()) / ds.mean())
    print(f"\n  spread across sub-corpora: {ds.min():.2f} to {ds.max():.2f}  "
          f"({100 * out['relative_spread']:.0f}% of the mean)")
    print("  for reference, changing n from 800 to 1600 on the full corpus moves the "
          "estimate by 0.84,")
    print("  so a spread below that is not distinguishable from sampling.")

    p = os.path.join(OUT, "composition_invariance.json")
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
