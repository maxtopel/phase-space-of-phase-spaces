#!/usr/bin/env python3
r"""Uniform bootstrap for every cell of Table I.

Table I previously mixed two protocols in one table. The universal-PSoPS row carried
bootstrap intervals from pipeline/compute_dimensions.py, while the three law rows carried single
deterministic runs of validation/law_dimensionality.py and so were quoted bare. The beta_1
column was bootstrapped for the laws and empty for the universal manifold. A reader
comparing 8.0 +/- 2.3 against a bare 5 cannot tell whether the law estimate is precise
or simply unmeasured.

This script puts all four rows on the same protocol: 40 subsamples at 85% without
replacement, matching scripts/fig7_law_attractors.py, recomputing the Coifman epsilon-scaling
dimension, the L-method knee and the significant-loop count on each draw.

The universal row's beta_1 is computed on the PSoPS diffusion coordinates rather than on a
raw plane, because the universal manifold has no two-variable plane to be the analogue
of the (unemployment, inflation) cloud. It answers a real question even so: whether the
macroeconomic manifold carries persistent one-cycles at all, or is contractible.

Writes artifacts/table1_bootstrap.json
Run    python3 scripts/table1_dimensions.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np

import pipeline.dimension as AD
from aoa_repro.systems import correlation_dimension
import pipeline.corpus as C
import validation.laws_pipeline as L
from validation.law_dimensionality import standardize, joint_embed, cyclical

OUT = C.OUT
N_BOOT, FRAC, SEED = 40, 0.85, 0
NS = 40
# ripser on a few thousand points in 10-d is not tractable; the laws are counted on a
# few hundred, so the universal row is subsampled to the same order for comparability
N_UNIVERSAL = 400


def _ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def bootstrap(points, raw2d=None, n_boot=N_BOOT, frac=FRAC, seed=SEED, label=""):
    """Resample the cloud and recompute the full battery on each draw.

    `points` is the object the dimension is measured on. `raw2d` is the object the loops
    are counted on, which for the laws is the bare (x, y) plane and not the embedding.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    dims, knees, loops, d2s = [], [], [], []
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        try:
            res, coords = AD.analyse(points[idx], label=label)
        except Exception as e:                     # a degenerate draw is dropped, not faked
            print(f"    draw {b} failed: {e}", flush=True)
            continue
        dims.append(res["coifman_dim"])
        knees.append(res["lmethod_knee"])
        # Grassberger-Procaccia D2: an independent scaling estimator in the same family
        # as Coifman (it reads a slope, not a cutoff), so it is not subject to the
        # retained-component ceiling that disqualified TWO-NN and Levina-Bickel
        try:
            d2s.append(correlation_dimension(points[idx]))
        except Exception:
            pass
        t = AD.loop_test(raw2d[idx] if raw2d is not None else coords)
        loops.append(t.get("n_significant_loops"))
    d, ds = _ms(dims)
    k, ks = _ms(knees)
    b1, b1s = _ms(loops)
    d2, d2sd = _ms(d2s)
    return {"n": int(n), "n_boot": len(dims), "frac": frac,
            "corr_dim_d2": d2, "corr_dim_sd": d2sd,
            "coifman_dim": d, "coifman_sd": ds,
            "lmethod_knee": k, "lmethod_sd": ks,
            "beta1": b1, "beta1_sd": b1s}


def main():
    out = {}

    # --- the three canonical laws, in the paper's canonical order ---------------
    laws = {"Phillips": L.phillips_cloud(), "Okun": L.okun_cloud(),
            "Solow": L.solow_cloud()}
    for name, (_, Xc, Yc, _, _) in laws.items():
        X, Y = standardize(np.asarray(Xc, float), np.asarray(Yc, float))
        m = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[m], Y[m]
        # dimension on the joint delay embedding, loops on the cyclical-detrended plane:
        # the same split the two source scripts use, now under one bootstrap
        cx, cy = standardize(cyclical(X), cyclical(Y))
        out[name] = bootstrap(joint_embed(X, Y), raw2d=np.c_[cx, cy], label=name)
        r = out[name]
        print(f"[{name:9s}] n={r['n']:4d}  coifman={r['coifman_dim']:.2f}+/-{r['coifman_sd']:.2f}  "
              f"D2={r['corr_dim_d2']:.2f}+/-{r['corr_dim_sd']:.2f}  "
              f"knee={r['lmethod_knee']:.1f}+/-{r['lmethod_sd']:.1f}  "
              f"beta1={r['beta1']:.1f}+/-{r['beta1_sd']:.1f}", flush=True)

    # --- the universal manifold, same protocol ---------------------------------
    # The Universal row needs the sqrt-plan binary from the Zenodo deposit; the
    # law rows above reproduce from the shipped CSVs alone.
    if not os.path.exists(os.path.join(OUT, "sqrtplans_dist.dat")):
        print("NOTE Universal row skipped: artifacts/sqrtplans_dist.dat is in the "
              "Zenodo deposit (DOI in the README); download it to reproduce this row. "
              "The shipped table1_bootstrap.json carries the full published table.", flush=True)
        return 0
    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    v = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]
    S = np.memmap(os.path.join(OUT, "sqrtplans_dist.dat"), dtype=np.float32, mode="r",
                  shape=(len(z["idx"]), NS * NS))
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(v), N_UNIVERSAL, replace=False))
    Xu = np.asarray(S[np.sort(v)[sel]], dtype=np.float64)
    out["Universal"] = bootstrap(Xu, label="Universal")
    r = out["Universal"]
    print(f"[Universal] n={r['n']:4d}  coifman={r['coifman_dim']:.2f}+/-{r['coifman_sd']:.2f}  "
          f"D2={r['corr_dim_d2']:.2f}+/-{r['corr_dim_sd']:.2f}  "
          f"knee={r['lmethod_knee']:.1f}+/-{r['lmethod_sd']:.1f}  "
          f"beta1={r['beta1']:.1f}+/-{r['beta1_sd']:.1f}", flush=True)
    print("\nNOTE the universal row is a subsample of 400 of the 17,025 plan vectors, so "
          "its Coifman dimension is not the headline 7.30 +/- 0.21, which is bootstrapped "
          "over the full corpus by pipeline/compute_dimensions.py. The headline stays the reported "
          "value; this row exists to put beta_1 and the knee on the laws' protocol.",
          flush=True)

    p = os.path.join(OUT, "table1_bootstrap.json")
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
