#!/usr/bin/env python3
r"""Is the Nystrom extension measuring the held-out series, or measuring its own shrinkage?

The full run (validation/nystrom_281k.py) reports that 99.6% of the 381,677 extended series
land inside the exhaustive corpus's support. Taken alone that statistic is close to
vacuous, because the extended cloud is also far more concentrated than the exhaustive
one: extended Psi_2 spans [-0.054, +0.042] against [-0.23, +0.17] for the exhaustive
corpus. Everything crammed into the core trivially has a near neighbour.

There are two readings and they have opposite consequences for the paper.

  (a) SUBSTANTIVE. Eurostat is granular regional and sectoral data. Those series are
      dynamically similar to one another, so they genuinely occupy a sub-region of the
      macroeconomic geometry rather than exploring it. The extension is informative.

  (b) ARTEFACT. A Nystrom extension with a Gaussian kernel shrinks toward the origin
      when a new point is far from every landmark: the kernel row goes near-uniform,
      so the projection loses contrast regardless of where the point actually sits.
      Concentration would then say nothing about the held-out series at all.

These are distinguishable. Under (b) the extended series must be systematically farther
from the landmark set than the exhaustive series are, and coordinate norm must fall off
with landmark distance. Under (a) the landmark distances overlap and the concentration
survives.

Writes artifacts/nystrom_diagnostic.json
Run    python3 validation/nystrom_diagnostic.py [n_sample]
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import sys
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import pdist, squareform

import pipeline.corpus as C
import pipeline.dimension as AD
from validation.nystrom_281k import fps, _solve_chunk, K_LAND, Q, NS, SEED, BATCH, WORKERS

def _require_dm_cache(path):
    import os as _os
    if not _os.path.exists(str(path)):
        raise FileNotFoundError(
            f"{path} is the stage-1 cost-matrix cache, which is not distributed "
            "(available on reasonable request; see README, 'The PSoPS "
            "construction chain'). All published results derived from it ship "
            "in artifacts/.")
    return path


OUT, DM = C.OUT, os.path.dirname(_require_dm_cache(C.COSTS))


def main(n_sample=6144):
    n_sample = int(n_sample)
    rng = np.random.default_rng(SEED)

    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    v = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]
    S = np.memmap(os.path.join(OUT, "sqrtplans_dist.dat"), dtype=np.float32,
                  mode="r", shape=(len(z["idx"]), NS * NS))
    Xex = np.asarray(S[np.sort(v)], dtype=np.float64)
    XL = Xex[fps(Xex, K_LAND, seed=SEED)]

    def to_landmarks(A):
        d2 = ((A ** 2).sum(1)[:, None] + (XL ** 2).sum(1)[None, :] - 2 * A @ XL.T)
        return np.sqrt(np.maximum(d2, 0))

    # --- re-solve a sample of held-out plans so we can see them in plan space ----
    L = np.load(os.path.join(DM, "labels.npz"), allow_pickle=True)
    macro = np.isin(L["timescale"], ("annual", "quarterly", "monthly"))
    held = np.where(macro & (L["n_points"] >= C.MIN_POINTS_DIST)
                    & (L["source"] == "eurostat"))[0]
    pick = np.sort(rng.choice(held, min(n_sample, len(held)), replace=False))
    B = np.load(os.path.join(OUT, "barycenter_dist.npy"))
    chunks = [(C.COSTS, B, pick[i:i + BATCH]) for i in range(0, len(pick), BATCH)]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        Xnew = np.vstack([c for c in ex.map(_solve_chunk, chunks) if len(c)])
    print(f"solved {len(Xnew)} held-out plans", flush=True)

    Dex, Dnew = to_landmarks(Xex).min(1), to_landmarks(Xnew).min(1)

    # --- the eigenbasis, so we can relate coordinate norm to landmark distance ---
    D = squareform(pdist(XL))
    sig, _ = AD.kernel_sum_bandwidth(D)
    KL = np.exp(-(D ** 2) / (2 * sig ** 2))
    dL = KL.sum(1)
    w, U = np.linalg.eigh(KL / np.sqrt(np.outer(dL, dL)))
    order = np.argsort(w)[::-1][1:Q + 1]
    lam, Phi = w[order], U[:, order]

    def nystrom(A):
        d2 = ((A ** 2).sum(1)[:, None] + (XL ** 2).sum(1)[None, :] - 2 * A @ XL.T)
        Kn = np.exp(-np.maximum(d2, 0) / (2 * sig ** 2))
        return ((Kn / np.sqrt(np.outer(Kn.sum(1), dL))) @ Phi) / lam

    Cex, Cnew = nystrom(Xex), nystrom(Xnew)
    Nex, Nnew = np.linalg.norm(Cex, axis=1), np.linalg.norm(Cnew, axis=1)

    # Under the artefact reading, coordinate norm falls as landmark distance grows.
    # Measure that slope on the exhaustive corpus, where we know the coordinates are
    # trustworthy, then ask whether the held-out shrinkage is what that slope predicts.
    rho = float(np.corrcoef(Dex, np.log(Nex + 1e-12))[0, 1])

    res = {
        "n_sample_heldout": int(len(Xnew)), "n_exhaustive": int(len(Xex)),
        "sigma": float(sig),
        "landmark_dist_exhaustive": {
            "p5": float(np.percentile(Dex, 5)), "median": float(np.median(Dex)),
            "p95": float(np.percentile(Dex, 95))},
        "landmark_dist_heldout": {
            "p5": float(np.percentile(Dnew, 5)), "median": float(np.median(Dnew)),
            "p95": float(np.percentile(Dnew, 95))},
        "frac_heldout_beyond_exhaustive_p95":
            float((Dnew > np.percentile(Dex, 95)).mean()),
        "coord_norm_median_exhaustive": float(np.median(Nex)),
        "coord_norm_median_heldout": float(np.median(Nnew)),
        "corr_landmarkdist_vs_lognorm_exhaustive": rho,
        # spread of the two clouds in the shared coordinates, outlier-robust
        "iqr_exhaustive": [float(x) for x in
                           (np.percentile(Cex, 75, 0) - np.percentile(Cex, 25, 0))[:3]],
        "iqr_heldout": [float(x) for x in
                        (np.percentile(Cnew, 75, 0) - np.percentile(Cnew, 25, 0))[:3]],
    }
    print(json.dumps(res, indent=2), flush=True)
    verdict = ("ARTEFACT-CONSISTENT: held-out series sit outside the landmark cover"
               if res["frac_heldout_beyond_exhaustive_p95"] > 0.5 else
               "SUBSTANTIVE: held-out series are inside the landmark cover")
    print("\n" + verdict, flush=True)
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(OUT, "nystrom_diagnostic.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else 6144))
