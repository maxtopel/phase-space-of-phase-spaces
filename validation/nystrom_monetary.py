#!/usr/bin/env python3
r"""
Macro-corpus validation of the plan-space Nystrom PSoPS extension (V3-faithful).

Uses the precomputed per-series 40x40 GW cost matrices for the real macro corpus
(atlas dm_cache/dist_matrices_453k.npy, mmap'd; domain/indicator labels from
metadata_453k.pkl). For a given (domain[,indicator]) group it builds the PSoPS two
ways and reports their held-out agreement:

  GOLD   : per-series entropic-GW plan to a local barycenter for EVERY series,
           then the full dense Hellinger-DMAP on all plans.
  NYSTROM: Hellinger-DMAP on K landmark plans only, every other series placed by
           extending that eigenbasis with its OWN plan's Hellinger distances to
           the K landmarks (no dense N^2 step).

This is the mechanism the universal PSoPS uses at 281K scale (per-series GW for all;
Nystrom avoids the intractable all-pairs spectral embedding, not the GW). RAM-safe:
cost matrices are memory-mapped and read in batches.

Run: python3 validation/nystrom_monetary.py [domain] [max_n]
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, sys, pickle
import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

def _require_dm_cache(path):
    import os as _os
    if not _os.path.exists(str(path)):
        raise FileNotFoundError(
            f"{path} is the stage-1 cost-matrix cache, which is not distributed "
            "(available on reasonable request; see README, 'The PSoPS "
            "construction chain'). All published results derived from it ship "
            "in artifacts/.")
    return path


import os as _os
DM = _require_dm_cache(_os.environ.get(
    "AOA_DM_CACHE",
    str(__import__("pathlib").Path.home() / "Desktop/atlas/research/core/AoA/checkpoints/family_discovery/dm_cache")))
COSTS = os.path.join(DM, "dist_matrices_453k.npy")
META  = os.path.join(DM, "metadata_453k.pkl")

N_S      = 40
EPS      = 0.05
SCALE    = 2.0          # graded-similarity regime for macro (null Hellinger ~0.5; see SI)
MAX_ITER = 1500
TOL      = 1e-7
K_LAND   = 150          # landmark plans for the DMAP eigenbasis
Q_DIM    = 6
K_DMAP   = 10
BATCH    = 256          # plan-solve batch (RAM-safe)
SEED     = 7
P        = np.full(N_S, 1.0 / N_S)


def norm_scale(C):
    """Normalize a cost matrix to unit max, then scale into the structured
    entropic regime (same convention as the dysts validation)."""
    m = C.max()
    return SCALE * (C / m) if m > 0 else C


def gw_plan(C, B):
    return np.asarray(ot.gromov.entropic_gromov_wasserstein(
        C, B, P, P, loss_fun="square_loss",
        epsilon=EPS, max_iter=MAX_ITER, tol=TOL), float)


def dmap_fit(D, q=Q_DIM, k=K_DMAP, alpha=1.0):
    n = len(D); kk = min(k, n - 1)
    bw = np.sort(D, 1)[:, kk] + 1e-12
    K = np.exp(-(D ** 2) / (bw[:, None] * bw[None, :]))
    da = K.sum(1); Ka = K / ((da[:, None] * da[None, :]) ** alpha)
    rs = Ka.sum(1) + 1e-12; s = np.sqrt(rs)
    Ms = Ka / (s[:, None] * s[None, :])
    w, V = np.linalg.eigh(Ms); o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]; phi = V / s[:, None]
    return phi[:, 1:q + 1] * w[1:q + 1], dict(bw=bw, da=da, phi=phi, w=w,
                                              k=kk, alpha=alpha, q=q)


def dmap_nystrom(Dod, fit):
    bw, da, phi, w = fit["bw"], fit["da"], fit["phi"], fit["w"]
    k, q, a = fit["k"], fit["q"], fit["alpha"]
    out = np.zeros((Dod.shape[0], q))
    for j in range(Dod.shape[0]):
        dd = Dod[j]; bwj = np.sort(dd)[k] + 1e-12
        kk = np.exp(-(dd ** 2) / (bwj * bw)); daj = kk.sum()
        ka = kk / ((daj ** a) * (da ** a)); p = ka / (ka.sum() + 1e-12)
        for c in range(q):
            out[j, c] = (1.0 / w[c + 1]) * (p @ phi[:, c + 1])
    return out


def plans_batched(arr, idx, B, label=""):
    """Per-series GW plans to B for all idx, read from the mmap in batches."""
    sqp = np.empty((len(idx), N_S * N_S), np.float64)
    for s in range(0, len(idx), BATCH):
        chunk = idx[s:s + BATCH]
        C = np.asarray(arr[chunk], np.float64)        # (b,40,40) into RAM only
        for j, Cj in enumerate(C):
            sqp[s + j] = np.sqrt(gw_plan(norm_scale(Cj), B)).ravel()
        print(f"    {label} plans {min(s+BATCH,len(idx))}/{len(idx)}", flush=True)
    return sqp


def fps(X, k, seed=0):
    """Farthest-point sampling on rows of X (Euclidean = Hellinger on sqrt-plans).
    O(n*k): only ever measures distance to already-selected landmarks."""
    n = len(X)
    sel = [int(np.random.default_rng(seed).integers(n))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax()); sel.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return sel


def run_group(arr, idx, name):
    rng = np.random.default_rng(SEED)
    if len(idx) > 1:
        idx = list(rng.permutation(idx))
    n = len(idx)
    L = min(K_LAND, n // 2)
    print(f"\n[{name}] n={n} -> {L} FPS landmark + {n-L} held-out", flush=True)

    # common GW target = Frobenius-medoid of a random sample of cost matrices
    samp = idx[:min(300, n)]
    Cs = np.stack([norm_scale(np.asarray(arr[i], np.float64)) for i in samp])
    fro = cdist(Cs.reshape(len(samp), -1), Cs.reshape(len(samp), -1))
    B = Cs[int(fro.sum(1).argmin())]
    del Cs, fro

    # per-series GW plan for EVERY series, then FPS landmarks in plan space
    X = plans_batched(arr, idx, B, name)
    landpos = fps(X, L, seed=SEED)
    lset = set(landpos)
    heldpos = [i for i in range(n) if i not in lset]
    Xl, Xh = X[landpos], X[heldpos]

    # GOLD dense DMAP on all plans; NYSTROM landmark DMAP + plan-space extension
    coords_full, _ = dmap_fit(cdist(X, X))
    coords_land, fitL = dmap_fit(cdist(Xl, Xl))
    coords_ny = dmap_nystrom(cdist(Xh, Xl), fitL)
    coords_self = dmap_nystrom(cdist(Xl, Xl), fitL)

    R, _ = orthogonal_procrustes(coords_land - coords_land.mean(0),
                                 coords_full[landpos] - coords_full[landpos].mean(0))
    al = lambda C: (C - coords_land.mean(0)) @ R + coords_full[landpos].mean(0)
    rho = lambda a, b: np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1]
                                for c in range(Q_DIM)])
    r_self = rho(al(coords_self), al(coords_land))
    r_ny = rho(al(coords_ny), coords_full[heldpos])
    per_mode = [float(np.corrcoef(al(coords_ny)[:, c], coords_full[heldpos][:, c])[0, 1])
                for c in range(Q_DIM)]
    print(f"[{name}] per-mode Nystrom-vs-full rho: "
          + ", ".join(f"rho_{c+1}={v:.2f}" for c, v in enumerate(per_mode)), flush=True)
    print(f"[{name}] RESULT self-consistency rho={r_self:.3f} | "
          f"held-out Nystrom-vs-full rho={r_ny:.3f}  (L={L}, H={n-L})", flush=True)
    print(f"LATEX_MACRO group={name} rho={r_ny:.3f} self={r_self:.3f} "
          f"L={L} H={n-L}", flush=True)
    return r_ny


def main():
    import warnings; warnings.filterwarnings("ignore")
    domain = sys.argv[1] if len(sys.argv) > 1 else "monetary"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    gi = np.load(os.path.join(DM, "group_indices.npz"))         # tiny index cache
    idx = list(gi[f"dom::{domain}"])
    arr = np.load(COSTS, mmap_mode="r")                          # mmap, not in RAM
    # drop degenerate cost matrices (max ~ 0: collapsed/constant series, no
    # attractor); their unit-normalization blows up and crashes the GW solver.
    idx = [int(i) for i in idx if float(np.asarray(arr[i], float).max()) > 1e-6]
    if len(idx) > max_n:
        idx = list(np.random.default_rng(SEED).choice(idx, max_n, replace=False))
    print(f"domain={domain}: {len(idx)} series (cap {max_n})", flush=True)
    run_group(arr, idx, domain)


if __name__ == "__main__":
    main()
