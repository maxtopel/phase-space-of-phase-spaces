#!/usr/bin/env python3
r"""Convergence, size dependence and cost of the Coifman dimension on the 17,025-series PSoPS.

WHY
    The headline dimension moved from 6.05 to 7.30 once the estimator was run at a
    converged sample size, without replacement, in Hellinger coordinates. That raises the
    obvious question: is 7.30 itself converged, or is it the next point on the same
    upward drift? "It stopped moving much" is not an answer, because a biased estimator
    can flatten below the true value.

    The only way to settle it is to calibrate the estimator against manifolds whose
    dimension is known, at the same sample sizes, in the same ambient dimension. If a
    true 8-manifold also reads 7.3 at n = 17,025, the corpus is an 8-manifold and the
    residual gap is estimator bias, not geometry.

WHAT IT MEASURES
    1  convergence   d(n) on the real corpus, n = 200 .. 17,025, repeated draws without
                     replacement, plus the whole corpus with no resampling at all
    2  calibration   d(n) on synthetic manifolds of KNOWN intrinsic dimension embedded in
                     the same 1600-dimensional ambient space, over the same ladder, which
                     gives the bias curve and lets the corpus value be corrected
    3  cost          wall-clock and peak memory against n, since the estimator needs the
                     dense n x n distance matrix and that is what bounds reproducibility

Writes artifacts/convergence_study.json
Run    python3 validation/convergence_floor.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import gc
import json
import time
import tracemalloc
import numpy as np
from scipy.spatial.distance import pdist, squareform

import pipeline.dimension as AD
import pipeline.corpus as C

OUT = C.OUT
NS, AMBIENT = 40, 1600
LADDER = (200, 400, 800, 1600, 3200, 6400, 9600, 12800)
SYNTH_D = (5, 7, 9, 11)
SYNTH_LADDER = (400, 1600, 6400, 12800)
SEED = 0


def coifman(X):
    """The estimator exactly as the paper uses it: 2 * max d log S / d log eps."""
    return float(AD.kernel_sum_bandwidth(squareform(pdist(X)))[1])


def load_corpus():
    z = np.load(os.path.join(OUT, "corpus_dist.npz"), allow_pickle=True)
    v = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]
    S = np.memmap(C.require_artifact("sqrtplans_dist.dat"), dtype=np.float32, mode="r",
                  shape=(len(z["idx"]), NS * NS))
    return S, np.sort(v)


def synth(d, n, seed=0, ambient=AMBIENT):
    """n points on a d-sphere, randomly embedded in `ambient` dimensions.

    Intrinsic dimension is exactly d. The sphere is used rather than a flat patch because
    the plan vectors are unit-norm Hellinger coordinates and therefore also live on a
    sphere, so the calibration object shares the corpus's curvature and normalisation.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d + 1))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = np.linalg.qr(rng.standard_normal((ambient, d + 1)))[0]
    return X @ Q.T


def main():
    res = {"ladder": list(LADDER), "synth_d": list(SYNTH_D),
           "synth_ladder": list(SYNTH_LADDER)}

    # ---- 1 + 3: convergence and cost on the real corpus ----------------------
    S, sv = load_corpus()
    conv, cost = {}, {}
    print("CORPUS  n=17,025 Hellinger plan vectors, sampled without replacement\n")
    print(f"  {'n':>6}  {'reps':>4}  {'d_eff':>14}   {'s':>7}  {'peak MB':>8}")
    for n in LADDER:
        reps = 8 if n <= 3200 else (4 if n <= 9600 else 2)
        vals, t0 = [], time.time()
        tracemalloc.start()
        for r in range(reps):
            rng = np.random.default_rng(SEED + r)
            X = np.asarray(S[sv[np.sort(rng.choice(len(sv), n, replace=False))]], np.float64)
            vals.append(coifman(X))
            del X
            gc.collect()
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        el = (time.time() - t0) / reps
        conv[n] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "reps": reps}
        cost[n] = {"seconds_per_call": el, "peak_mb": peak}
        print(f"  {n:6d}  {reps:4d}  {np.mean(vals):6.3f} +/- {np.std(vals):5.3f}   "
              f"{el:7.1f}  {peak:8.0f}", flush=True)

    X = np.asarray(S[sv], np.float64)
    t0 = time.time()
    d_full = coifman(X)
    res["full_corpus"] = {"n": int(len(sv)), "d_eff": d_full,
                          "seconds": time.time() - t0}
    del X
    gc.collect()
    print(f"  {len(sv):6d}     1  {d_full:6.3f}  (no resampling)", flush=True)
    res["convergence"] = conv
    res["cost"] = cost

    # increments per doubling: the practical convergence test
    ks = sorted(conv)
    print("\n  increment per doubling of n")
    inc = {}
    for a, b in zip(ks, ks[1:]):
        per = (conv[b]["mean"] - conv[a]["mean"]) / np.log2(b / a)
        inc[f"{a}->{b}"] = float(per)
        print(f"    {a:6d} -> {b:6d}   {per:+.3f}")
    res["increment_per_doubling"] = inc

    # ---- 2: calibration against known dimension ------------------------------
    print("\nCALIBRATION  d-spheres embedded in 1600 dimensions, identical estimator\n")
    print(f"  {'true d':>6} " + "".join(f"{n:>9}" for n in SYNTH_LADDER))
    cal = {}
    for d in SYNTH_D:
        row = {}
        for n in SYNTH_LADDER:
            reps = 3 if n <= 6400 else 1
            vals = [coifman(synth(d, n, seed=SEED + r)) for r in range(reps)]
            row[n] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
            gc.collect()
        cal[d] = row
        print(f"  {d:6d} " + "".join(f"{row[n]['mean']:9.2f}" for n in SYNTH_LADDER),
              flush=True)
    res["calibration"] = cal

    # ---- what the corpus value implies once the bias is removed --------------
    n_ref = SYNTH_LADDER[-1]
    truth = np.array(SYNTH_D, float)
    read = np.array([cal[d][n_ref]["mean"] for d in SYNTH_D], float)
    # invert the calibration curve at the corpus's own reading
    corpus_read = conv[12800]["mean"]
    if read[0] <= corpus_read <= read[-1]:
        implied = float(np.interp(corpus_read, read, truth))
    else:
        implied = float(np.polyval(np.polyfit(read, truth, 1), corpus_read))
    res["implied_true_dimension"] = {"corpus_reading_at_n": n_ref,
                                     "corpus_reading": float(corpus_read),
                                     "implied_d": implied}
    print(f"\n  at n={n_ref} the corpus reads {corpus_read:.2f}; inverting the "
          f"calibration curve gives an implied true dimension of {implied:.1f}")

    p = os.path.join(OUT, "convergence_study.json")
    json.dump(res, open(p, "w"), indent=2)
    print("\nwrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
