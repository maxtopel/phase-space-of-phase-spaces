#!/usr/bin/env python3
"""Split-half stability of the universal barycenter.

The corpus is split into two independent random halves. Each half fits its own
barycenter by the production rule (concept-balanced seeded draw, at most 120
series per concept, truncated to 1500, non-entropic GW). The two reference
geometries are then compared by Procrustes disparity on the barycenter distance
matrices (after optimal alignment) and by the relative difference of their
kernel eigenvalue spectra. Writes artifacts/split_half.json.

Run: python3 validation/split_half.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
import os
import numpy as np
from scipy.spatial import procrustes

import pipeline.bridge as V3
import pipeline.corpus as C

OUT = C.OUT
SEED = 7


def half_barycenter(idx, con, arr, rng):
    land = []
    for c in np.unique(con):
        ix = idx[con == c]
        land.extend(rng.choice(ix, min(120, len(ix)), replace=False).tolist())
    lc = [V3.normalize_cost(np.asarray(arr[i], float)) for i in land
          if np.isfinite(arr[i]).all() and np.asarray(arr[i]).max() > C.MIN_COSTMAX]
    return V3.gw_barycenter_nonentropic(lc[:1500])


def main():
    bal = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    idx, con = bal["idx"], bal["concept"]
    arr = np.load(C.require_cache(), mmap_mode="r")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(idx))
    h1, h2 = perm[: len(idx) // 2], perm[len(idx) // 2:]

    B1 = half_barycenter(idx[h1], con[h1], arr, np.random.default_rng(SEED + 1))
    B2 = half_barycenter(idx[h2], con[h2], arr, np.random.default_rng(SEED + 2))
    print("both half-barycenters fitted", flush=True)

    _, _, disparity = procrustes(B1, B2)
    e1 = np.sort(np.linalg.eigvalsh(np.exp(-B1 ** 2 / np.median(B1) ** 2)))[::-1]
    e2 = np.sort(np.linalg.eigvalsh(np.exp(-B2 ** 2 / np.median(B2) ** 2)))[::-1]
    k = 10
    spec = float(np.max(np.abs(e1[:k] - e2[:k]) / np.maximum(np.abs(e1[:k]), 1e-12)))

    res = {"procrustes_disparity": float(disparity),
           "spectral_max_reldiff_top10": spec,
           "n_half": int(len(h1)), "seed": SEED}
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(os.path.join(OUT, "split_half.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
