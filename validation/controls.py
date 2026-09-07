#!/usr/bin/env python3
r"""Out-of-domain controls under matched solver settings: a NEGATIVE result.

An earlier version of this analysis reported that the 156 held-out controls
(WHO health, river gauges, NOAA weather) fail to resolve in the PSoPS basis:
control plans at 99.98% of maximum entropy against structured macro plans,
ratio 42x, AUC 1.000, complete separation. That comparison solved the two
sides at different settings: macro plans were read from plans_dist.dat
(eps = 0.008, distributed barycenter) while control plans were solved fresh
at eps = 0.05 against the n>=200 barycenter. The structure statistic
||T - U||_F is monotone in eps for EVERY series, so the published separation
measured the eps mismatch, not the domain.

Under matched settings the separation does not exist:

    both at eps = 0.008:  ratio 1.10, AUC 0.59
    both at eps = 0.02:   ratio 1.46, AUC 0.58
    both at eps = 0.05:   ratio 1.09, AUC 0.43

and the Nystrom placement test (same machinery as validation/nystrom_281k.py)
puts 98.7% of controls within the manifold support, against 99.6% for the
held-out Eurostat reference. The basis places out-of-domain attractors
rather than rejecting them: the PSoPS is a representation of attractor
geometry, not of domain membership. The paper states this and claims no
separation.

This script reproduces the matched comparison at eps = 0.008 and the
placement test, and writes artifacts/controls_matched.json.
Run: python3 validation/controls.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

import pipeline.bridge as V3
import pipeline.corpus as C
import pipeline.dimension as AD
import validation.nystrom_281k as NY

OUT = C.OUT
N_S = 40
EPS = 0.008                       # the production eps (placement test)
EPS_GRID = (0.008, 0.02, 0.05)    # matched comparison at every operating point
U = np.full(N_S * N_S, 1.0 / (N_S * N_S))


def _row_normalise(T):
    T = np.asarray(T, np.float64)
    return T / T.sum(axis=1, keepdims=True)


def _load_control_costs():
    arr = np.load(C.require_cache(), mmap_mode="r")
    zc = np.load(os.path.join(OUT, "corpus.npz"), allow_pickle=True)
    costs, src = [], []
    for i, s in zip(zc["ctrl_idx"], zc["ctrl_source"]):
        M = np.asarray(arr[i], np.float64)
        if np.isfinite(M).all() and M.max() > C.MIN_COSTMAX:
            costs.append(V3.normalize_cost(M))
            src.append(str(s))
    return costs, np.array(src)


def main():
    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    n = len(z["idx"])
    valid = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]

    # macro plans: the stored production solves (eps = 0.008, barycenter_dist)
    P = np.memmap(C.require_artifact("plans_dist.dat"), dtype=np.float32, mode="r",
                  shape=(n, N_S * N_S))
    Tm = _row_normalise(np.asarray(P[np.sort(valid)], dtype=np.float64))
    d_macro = np.linalg.norm(Tm - U, axis=1)

    # control plans: same barycenter, same eps, at every operating point. The
    # macro side at eps != 0.008 uses a 1000-series matched subsample solved live.
    B = np.load(os.path.join(OUT, "barycenter_dist.npy"))
    costs, src = _load_control_costs()
    arr = np.load(C.require_cache(), mmap_mode="r")
    rng = np.random.default_rng(0)
    sub = rng.choice(z["idx"], 1000, replace=False)
    mc = [V3.normalize_cost(np.asarray(arr[i], np.float64)) for i in sub
          if np.isfinite(arr[i]).all() and np.asarray(arr[i]).max() > C.MIN_COSTMAX]
    by_eps = {}
    plans = None
    for eps in EPS_GRID:
        cp = [np.asarray(T, float) for T in V3.plans_to(costs, B, epsilon=eps)]
        Tc = _row_normalise(np.array([T.ravel() for T in cp]))
        d_c = np.linalg.norm(Tc - U, axis=1)
        if eps == EPS:
            plans = cp
            d_m = d_macro
        else:
            Tms = _row_normalise(np.array([np.asarray(T, float).ravel()
                                           for T in V3.plans_to(mc, B, epsilon=eps)]))
            d_m = np.linalg.norm(Tms - U, axis=1)
        auc = 1.0 - float((d_c[:, None] > d_m[None, :]).mean())
        by_eps[str(eps)] = {"macro_struct_mean": float(d_m.mean()),
                            "ctrl_struct_mean": float(d_c.mean()),
                            "ratio": float(d_m.mean() / d_c.mean()), "auc": auc,
                            "complete_separation": bool(d_c.max() < d_m.min())}
        print(f"matched eps={eps}: macro ||T-U|| {d_m.mean():.4f}, controls "
              f"{d_c.mean():.4f}, ratio {d_m.mean()/d_c.mean():.2f}, AUC {auc:.2f} "
              f"(no separation)", flush=True)
    d_ctrl = np.linalg.norm(_row_normalise(np.array([T.ravel() for T in plans])) - U, axis=1)
    auc = by_eps[str(EPS)]["auc"]

    # placement: Nystrom the controls into the coordinates, same basis as the
    # 281k extension, and ask whether they land within manifold support
    S = np.memmap(C.require_artifact("sqrtplans_dist.dat"), dtype=np.float32,
                  mode="r", shape=(n, N_S * N_S))
    Xex = np.asarray(S[np.sort(valid)], dtype=np.float64)
    XL = Xex[NY.fps(Xex, NY.K_LAND, seed=NY.SEED)]
    D = squareform(pdist(XL))
    sig, _ = AD.kernel_sum_bandwidth(D)
    KL = np.exp(-(D ** 2) / (2 * sig ** 2))
    dL = KL.sum(1)
    w, Uv = np.linalg.eigh(KL / np.sqrt(np.outer(dL, dL)))
    order = np.argsort(w)[::-1][1:NY.Q + 1]
    lam, Phi = w[order], Uv[:, order]

    def nystrom(X):
        d2 = ((X ** 2).sum(1)[:, None] + (XL ** 2).sum(1)[None, :] - 2 * X @ XL.T)
        Kn = np.exp(-np.maximum(d2, 0) / (2 * sig ** 2))
        Kn = Kn / np.sqrt(np.outer(Kn.sum(1), dL))
        return (Kn @ Phi) / lam

    Cex = nystrom(Xex)
    Xc = np.array([np.sqrt(T / T.sum()).ravel() for T in plans])
    Cc = nystrom(Xc)
    tree = cKDTree(Cex)
    d_new = tree.query(Cc, k=1)[0]
    rng = np.random.default_rng(NY.SEED)
    q = rng.choice(len(Cex), min(4000, len(Cex)), replace=False)
    d_self = tree.query(Cex[q], k=2)[0][:, 1]
    thr = np.percentile(d_self, 95)
    frac = float((d_new <= thr).mean())
    print(f"placement: {frac:.3f} of controls within manifold support "
          f"(median dist {np.median(d_new):.4f} vs within-manifold NN "
          f"{np.median(d_self):.4f})", flush=True)

    res = {"eps_matched": EPS, "by_eps": by_eps, "n_macro": int(len(d_macro)), "n_controls": int(len(d_ctrl)),
           "macro_struct_mean": float(d_macro.mean()), "ctrl_struct_mean": float(d_ctrl.mean()),
           "ratio_macro_over_ctrl": float(d_macro.mean() / d_ctrl.mean()),
           "auc_macro_vs_ctrl": auc,
           "complete_separation": bool(d_ctrl.max() < d_macro.min()),
           "ctrl_frac_within_support": frac,
           "ctrl_median_dist_to_manifold": float(np.median(d_new)),
           "within_manifold_nn_median": float(np.median(d_self)),
           "by_source_median_dist": {s: float(np.median(d_new[src == s]))
                                     for s in np.unique(src)},
           "note": "matched-settings rerun 2026-09-05; supersedes "
                   "controls_separation.json, whose 42x/AUC 1.000 compared "
                   "eps=0.008 macro plans against eps=0.05 control plans"}
    json.dump(res, open(os.path.join(OUT, "controls_matched.json"), "w"), indent=2)
    print("wrote artifacts/controls_matched.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
