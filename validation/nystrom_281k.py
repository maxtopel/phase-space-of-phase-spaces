#!/usr/bin/env python3
r"""Nystrom extension of the universal PSoPS from the 17,025-series exhaustive corpus
to the held-out remainder of the cached corpus.

WHY THIS EXISTS
    pipeline/corpus.py rule R1 excludes Eurostat (88% of the 453,907-attractor cache,
    granular regional and sectoral rather than macroeconomic) from the universal
    core, and states that it is "re-attached later as a Nystrom extension: a
    feature". Until that extension is actually run, the paper can only claim the
    manifold describes the 17,025 series embedded exhaustively, not the full
    corpus it ingests. This script performs the extension and measures whether the
    held-out series land on the same object.

METHOD
    The Nystrom formula needs the kernel between a new point and the landmarks.
    Here that kernel is a Hellinger distance between transport plans, so every new
    series still needs its own entropic-GW solve against the universal barycenter.
    What Nystrom avoids is the dense N x N eigendecomposition, not the transport.

      1. landmarks   K farthest-point-sampled landmarks from the exhaustive plans
      2. eigenbasis  dense Hellinger-DMAP on the landmark plans (alpha = 1)
      3. extension   solve each held-out plan, embed by the Nystrom formula
      4. comparison  do the extended points occupy the same manifold, at the same
                     effective dimension, as the exhaustive corpus?

OUTPUT
    artifacts/nystrom_281k.json

RUN
    python3 validation/nystrom_281k.py [n_extend]      # n_extend omitted = all held-out
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

import pipeline.corpus as C
import pipeline.dimension as AD

OUT = C.OUT
DM = os.path.dirname(C.COSTS)
NS, SEED = 40, 7
K_LAND = 800          # matches the landmark count used in pipeline/coordinates.py
Q = 10                # PSoPS coordinates retained
BATCH = 2048
WORKERS = 8


def _solve_chunk(args):
    """Solve entropic-GW plans for a chunk of cost-matrix indices. Module level so
    ProcessPoolExecutor can pickle it. Returns sqrt-normalised plans."""
    costs_path, bary, idx = args
    import numpy as _np
    import pipeline.bridge as _V3
    import pipeline.corpus as _C
    a = _np.load(costs_path, mmap_mode="r")
    costs = []
    for i in idx:
        M = _np.asarray(a[i], _np.float64)
        if _np.isfinite(M).all() and M.max() > _C.MIN_COSTMAX:
            costs.append(_V3.normalize_cost(M))
    if not costs:
        return _np.empty((0, 1600))
    return _np.array([_np.sqrt(_np.asarray(T, float) / _np.asarray(T, float).sum()).ravel()
                      for T in _V3.plans_to(costs, bary)])


def fps(X, k, seed=0):
    """Farthest point sampling in the Hellinger (sqrt-plan) space."""
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(sel)


def main(n_extend=None):
    C.require_cache()
    rng = np.random.default_rng(SEED)
    t_all = time.time()

    # ---- exhaustive corpus: the series the PSoPS is built on -------------------
    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    n_ex = len(z["idx"])
    v = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]
    S = np.memmap(os.path.join(OUT, "sqrtplans_dist.dat"), dtype=np.float32,
                  mode="r", shape=(n_ex, NS * NS))
    Xex = np.asarray(S[np.sort(v)], dtype=np.float64)
    print(f"exhaustive corpus: {Xex.shape[0]} series", flush=True)

    # ---- landmarks and the dense eigenbasis ---------------------------------
    XL = Xex[fps(Xex, K_LAND, seed=SEED)]
    D = squareform(pdist(XL))
    sig, _ = AD.kernel_sum_bandwidth(D)
    KL = np.exp(-(D ** 2) / (2 * sig ** 2))
    dL = KL.sum(1)
    w, U = np.linalg.eigh(KL / np.sqrt(np.outer(dL, dL)))
    order = np.argsort(w)[::-1][1:Q + 1]          # drop the trivial constant mode
    lam, Phi = w[order], U[:, order]
    print(f"landmarks {K_LAND}, sigma {sig:.4f}, leading eigenvalues "
          f"{np.round(lam[:4], 4).tolist()}", flush=True)

    def nystrom(Xnew):
        d2 = ((Xnew ** 2).sum(1)[:, None] + (XL ** 2).sum(1)[None, :]
              - 2 * Xnew @ XL.T)
        Kn = np.exp(-np.maximum(d2, 0) / (2 * sig ** 2))
        Kn = Kn / np.sqrt(np.outer(Kn.sum(1), dL))
        return (Kn @ Phi) / lam

    Cex = nystrom(Xex)

    # ---- held-out population: what rule R1 excluded -------------------------
    L = np.load(os.path.join(DM, "labels.npz"), allow_pickle=True)
    macro = np.isin(L["timescale"], ("annual", "quarterly", "monthly"))
    held = np.where(macro & (L["n_points"] >= C.MIN_POINTS_DIST)
                    & (L["source"] == "eurostat"))[0]
    if n_extend:
        held = np.sort(rng.choice(held, min(int(n_extend), len(held)), replace=False))
    print(f"held-out (eurostat, macro-frequency, n>={C.MIN_POINTS_DIST}): "
          f"{len(held)} attractors", flush=True)

    B = np.load(os.path.join(OUT, "barycenter_dist.npy"))
    chunks = [(C.COSTS, B, held[i:i + BATCH]) for i in range(0, len(held), BATCH)]
    # checkpoint every 10 chunks so a long run survives interruption
    ck = os.path.join(OUT, "nystrom_281k_partial.npy")
    parts, kept, done, t0 = [], 0, 0, time.time()
    if os.path.exists(ck):
        prev = np.load(ck)
        parts.append(prev); kept = done = len(prev)
        print(f"  resuming from checkpoint: {kept} already embedded", flush=True)
        chunks = chunks[kept // BATCH:]   # resume assumes no dropped series so far
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for c, Xn in enumerate(ex.map(_solve_chunk, chunks), 1):
            done += BATCH
            if len(Xn):
                parts.append(nystrom(Xn))
                kept += len(Xn)
            el = time.time() - t0
            d = min(done, len(held))
            print(f"  {d}/{len(held)} kept={kept} {el:.0f}s "
                  f"eta={el / max(c * BATCH, 1) * (len(held) - d) / 3600:.2f}h", flush=True)
            if c % 10 == 0:
                np.save(ck, np.vstack(parts))
    Cnew = np.vstack(parts)
    if os.path.exists(ck):
        os.remove(ck)

    # ---- do the extended points lie on the same object? ---------------------
    tree = cKDTree(Cex)
    d_new = tree.query(Cnew, k=1)[0]
    q = rng.choice(len(Cex), min(4000, len(Cex)), replace=False)
    d_self = tree.query(Cex[q], k=2)[0][:, 1]
    sub = rng.choice(len(Cnew), min(4000, len(Cnew)), replace=False)

    res = {
        "n_exhaustive": int(len(Cex)),
        "n_extended": int(kept),
        "K_landmarks": K_LAND,
        "sigma": float(sig),
        "d_eff_exhaustive": float(AD.kernel_sum_bandwidth(squareform(pdist(Cex[q])))[1]),
        "d_eff_extended": float(AD.kernel_sum_bandwidth(squareform(pdist(Cnew[sub])))[1]),
        "median_dist_extended_to_manifold": float(np.median(d_new)),
        "median_within_manifold_nn": float(np.median(d_self)),
        "frac_extended_within_manifold_support":
            float((d_new <= np.percentile(d_self, 95)).mean()),
        "elapsed_s": float(time.time() - t_all),
    }
    print("\nRESULT " + json.dumps(res, indent=2), flush=True)
    json.dump(res, open(os.path.join(OUT, "nystrom_281k.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "nystrom_281k_coords.npz"),
                        exhaustive=Cex.astype(np.float32),
                        extended=Cnew.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
