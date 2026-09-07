#!/usr/bin/env python3
r"""
Plan-space Nystrom extension of the PSoPS embedding (V3-faithful), + the
entropic-vs-exact GW regularization regime cited in the SI.

WHAT V3 DOES (and what this validates). The PSoPS coordinates are a Hellinger-DMAP
on per-series Gromov--Wasserstein (GW) transport plans: every series gets ONE
cheap GW plan to a (local) barycenter -- per-series GW is run for ALL series,
O(N), ~30 min for 281K. The expensive step is the DENSE N x N Hellinger-DMAP
(all-pairs plan distances + eigendecomposition), which is O(N^2) and intractable
at 281K. The Nystrom extension avoids THAT, not the GW: build the diffusion-map
eigenbasis on K landmark plans, then place every other series by extending that
eigenbasis using the series' OWN plan and its Hellinger distances to the K
landmark plans. No descriptors are involved (descriptors were a v1 device and
appear nowhere in the reported method).

This script validates, on the local chaotic-systems validation corpus (the Fig.-2
systems; 96 per-series 40x40 GW cost matrices from validation/dysts_families.py), that
the plan-space Nystrom extension reproduces the full dense Hellinger-DMAP:
  GOLD  : Hellinger-DMAP on ALL plans (the dense embedding).
  NYSTROM: Hellinger-DMAP on landmark plans, held-out series extended via the
           plan-Hellinger kernel to the landmarks (no dense N^2 step).
Agreement is the Procrustes-aligned coordinate correlation rho on held-out
series, with a landmark self-consistency check (should be ~1).

It also computes the entropic-vs-exact GW regime result cited in SI
('Entropic vs. exact GW'): exact GW gives near-orthogonal plans; entropic GW at
fixed epsilon collapses to the uniform coupling unless cost/epsilon is large.

Run:     python3 validation/nystrom_equivalence.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

COSTS_NPZ = os.path.join(_ROOT, "caches", "cache_dysts_costs.npz")

# ----------------------------------------------------------------- parameters
N_S       = 40        # support size
EPS       = 0.05      # entropic-GW regularization (POT-stable regime)
SCALE     = 6.0       # cost scaling: eps/SCALE ~ 0.008 effective (paper regime)
MAX_ITER  = 2000
TOL       = 1e-8
FRAC_LAND = 0.67      # landmark fraction (rest held out)
Q_DIM     = 6         # PSoPS diffusion coordinates compared
K_DMAP    = 10        # adaptive-bandwidth kNN for the Hellinger-DMAP kernel
SEED      = 7
P         = np.full(N_S, 1.0 / N_S)


# ----------------------------------------------------------------- GW + metrics
def gw_plan(C, B):
    """Entropic GW transport plan from cost matrix C to reference B."""
    return ot.gromov.entropic_gromov_wasserstein(
        C, B, P, P, loss_fun="square_loss",
        epsilon=EPS, max_iter=MAX_ITER, tol=TOL)


def hellinger(Ta, Tb):
    return float(np.linalg.norm(np.sqrt(Ta) - np.sqrt(Tb)) / np.sqrt(2.0))


# ----------------------------------------------------------------- Hellinger-DMAP
def dmap_fit(D, q=Q_DIM, k=K_DMAP, alpha=1.0):
    """Diffusion map on a precomputed distance matrix D (here Hellinger between
    plans). Returns (coords, fit) where fit lets us Nystrom-extend out-of-sample."""
    n = len(D); kk = min(k, n - 1)
    bw = np.sort(D, 1)[:, kk] + 1e-12
    K = np.exp(-(D ** 2) / (bw[:, None] * bw[None, :]))
    da = K.sum(1); Ka = K / ((da[:, None] * da[None, :]) ** alpha)
    rs = Ka.sum(1) + 1e-12; s = np.sqrt(rs)
    Ms = Ka / (s[:, None] * s[None, :])
    w, V = np.linalg.eigh(Ms); o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]; phi = V / s[:, None]
    coords = phi[:, 1:q + 1] * w[1:q + 1]
    return coords, dict(bw=bw, da=da, phi=phi, w=w, k=kk, alpha=alpha, q=q)


def dmap_nystrom(Dod, fit):
    """Coifman--Lafon Nystrom extension: place out-of-sample rows (distance matrix
    Dod, shape (M, L), to the L fitted landmarks) into the landmark eigenbasis."""
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


# ----------------------------------------------------------------- SI regime test
def exact_vs_entropic(costs_unit, ref, m=14):
    """SI ('Entropic vs.\\ exact GW'): exact GW gives near-orthogonal plans;
    entropic GW at fixed epsilon collapses to the uniform coupling unless the
    cost/epsilon ratio is large (swept by scaling the unit costs)."""
    B0 = costs_unit[ref]; sub = costs_unit[:m]
    ex = [ot.gromov.gromov_wasserstein(c, B0, P, P, loss_fun="square_loss")
          for c in sub]
    nx = [hellinger(ex[i], ex[j])
          for i in range(len(ex)) for j in range(i + 1, len(ex))]
    print("\n========= ENTROPIC vs EXACT GW (SI regularization regime) =========")
    print(f"  exact GW          : mean pairwise Hellinger = {np.mean(nx):.3f} "
          f"(range {np.min(nx):.2f}-{np.max(nx):.2f})")
    for s in (1, 3, 6):
        en = [ot.gromov.entropic_gromov_wasserstein(
                  s * c, s * B0, P, P, loss_fun="square_loss",
                  epsilon=0.05, max_iter=1000, tol=1e-7) for c in sub]
        ne = [hellinger(en[i], en[j])
              for i in range(len(en)) for j in range(i + 1, len(en))]
        print(f"  entropic eps=0.05 scale x{s}: mean pairwise Hellinger = "
              f"{np.mean(ne):.3f}")


# ----------------------------------------------------------------- run
def main():
    import warnings
    warnings.filterwarnings("ignore")
    print("loading local dysts cost-matrix corpus ...", flush=True)
    d = np.load(COSTS_NPZ, allow_pickle=True)
    costs = [SCALE * np.asarray(C, float) for C in d["costs"]]
    n = len(costs)
    L = int(round(FRAC_LAND * n)); H = n - L
    print(f"  {n} series -> {L} landmark + {H} held-out (eps={EPS}, scale={SCALE})",
          flush=True)

    # fixed common GW target: Frobenius-medoid cost matrix (stand-in for a local
    # barycenter; the Nystrom mechanism is identical for a true barycenter).
    M = np.array([[np.linalg.norm(a - b) for b in costs] for a in costs])
    ref = int(M.sum(1).argmin()); B = costs[ref]
    print(f"  reference = corpus medoid (series {ref})", flush=True)

    rng = np.random.default_rng(SEED)
    perm = [i for i in rng.permutation(n) if i != ref]
    land_idx, held_idx = perm[:L], perm[L:L + H]

    print("per-series GW plans (entropic) for ALL series ...", flush=True)
    T_land = [np.asarray(gw_plan(costs[i], B), float) for i in land_idx]
    T_held = [np.asarray(gw_plan(costs[i], B), float) for i in held_idx]

    # sqrt-plan vectors: Euclidean distance between them IS the Hellinger distance
    Xl = np.array([np.sqrt(T).ravel() for T in T_land])
    Xh = np.array([np.sqrt(T).ravel() for T in T_held])

    print("GOLD: dense Hellinger-DMAP on ALL plans ...", flush=True)
    coords_full, _ = dmap_fit(cdist(np.vstack([Xl, Xh]), np.vstack([Xl, Xh])))

    print("NYSTROM: Hellinger-DMAP on landmark plans + plan-space extension ...",
          flush=True)
    coords_land, fitL = dmap_fit(cdist(Xl, Xl))
    coords_ny = dmap_nystrom(cdist(Xh, Xl), fitL)          # held-out (plan kernel)
    coords_self = dmap_nystrom(cdist(Xl, Xl), fitL)        # self-consistency

    # align the landmark eigenbasis to the gold (shared landmark points) and apply
    R, _ = orthogonal_procrustes(coords_land - coords_land.mean(0),
                                 coords_full[:L] - coords_full[:L].mean(0))
    def algn(C):
        return (C - coords_land.mean(0)) @ R + coords_full[:L].mean(0)

    def agree(a, b):
        r = np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(Q_DIM)])
        nr = np.linalg.norm(a - b) / (np.linalg.norm(b - b.mean(0)) + 1e-12)
        return r, nr

    r_self, _ = agree(algn(coords_self), algn(coords_land))
    r_ny, nr_ny = agree(algn(coords_ny), coords_full[L:])

    print("\n========= PLAN-SPACE NYSTROM vs FULL DENSE Hellinger-DMAP =========")
    print(f"  landmark self-consistency (extend landmark by its own plan): "
          f"rho={r_self:.3f}  (should be ~1)")
    print(f"  held-out PSoPS coords: Nystrom vs full dense  rho={r_ny:.3f}  "
          f"norm.RMSE={nr_ny:.3f}  (q={Q_DIM}, L={L}, H={H})")
    print("LATEX_NYSTROM rho=%.3f nrmse=%.3f self=%.3f q=%d L=%d H=%d"
          % (r_ny, nr_ny, r_self, Q_DIM, L, H))

    exact_vs_entropic([C / SCALE for C in costs], ref)


if __name__ == "__main__":
    main()
