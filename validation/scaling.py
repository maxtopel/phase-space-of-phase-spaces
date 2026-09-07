#!/usr/bin/env python3
r"""Re-run the Frechet-mean scaling sweep recording the Coifman eps-scaling dimension.

The stored artifacts/scaling.json contains only two_nn and levina_bickel, both of
which are rejected as the figure of record (density- and n_components-dependent on
diffusion coordinates; see pipeline/dimension.py). Fig. fig_aoa_scaling therefore plots the
wrong estimator, which is why it appears to plateau near 11, far from the headline 7.3.

Writes artifacts/scaling_coifman_planspace.json with coifman_dim alongside the
legacy columns.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
import pipeline.coordinates as E
import pipeline.dimension as AD
import pipeline.bridge as V3

OUT = E.OUT


def _distinct(X, eps=0.03, seed=0):
    """Near-duplicate gate (a legacy of an earlier pipeline/compute_dimensions.py protocol;
    the headline itself no longer deduplicates -- see compute_dimensions.corpus_dims)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    keep = []
    for i in order:
        if not keep:
            keep.append(i); continue
        d = np.linalg.norm(X[keep] - X[i], axis=1)
        if d.min() > eps:
            keep.append(i)
    return np.array(sorted(keep))


def main():
    arr = E._arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    idx = bal["idx"]
    rng = np.random.default_rng(E.SEED)
    perm = rng.permutation(idx)
    eval_idx, pool = perm[:E.EVAL_N], perm[E.EVAL_N:]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for Nb in [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, min(8000, len(pool))]:
            if Nb > len(pool):
                break
            land, _ = E._costs_of(arr, pool[:Nb])
            B = V3.gw_barycenter_nonentropic(land[:Nb])
            P, _ = E._plans_parallel(ex, eval_idx, B)
            coords, evals, gap = V3.aoa_coords(P, n_components=20)
            # Measure the SAME object as the headline d_eff: the Hellinger plan
            # vectors themselves (pipeline/compute_dimensions.py), not the 20-d diffusion
            # coordinates. Otherwise the plateau value is not comparable to the
            # headline.
            X = np.array([np.sqrt(np.asarray(T, float) / np.asarray(T, float).sum()).ravel()
                          for T in P]) / np.sqrt(2.0)
            Xd = X[_distinct(X)]
            res_plan, _ = AD.analyse(Xd)
            res_crd, _ = AD.analyse(np.asarray(coords, float))
            row = {"N_frechet": int(Nb),
                   "coifman_dim": float(res_plan["coifman_dim"]),      # plan space (headline object)
                   "lmethod_knee": int(res_plan["lmethod_knee"]),
                   "coifman_dim_coords": float(res_crd["coifman_dim"]),  # 20-d coords, for reference
                   "n_distinct": int(len(Xd)),
                   **E._deff_row(coords),
                   "curvature": E.curvature(coords), "gap": float(gap)}
            rows.append(row)
            print(f"  N={Nb:5d}  coifman(plan)={row['coifman_dim']:.2f}  coifman(coords)={row['coifman_dim_coords']:.2f}  "
                  f"n_distinct={row['n_distinct']}  two_nn={row['two_nn']:.2f}", flush=True)
            json.dump({"eval_n": E.EVAL_N, "scaling": rows, "eps": E.EPS},
                      open(os.path.join(OUT, "scaling_coifman_planspace.json"), "w"), indent=2)
    print("wrote artifacts/scaling_coifman_planspace.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
