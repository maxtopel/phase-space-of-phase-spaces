"""
aoa_v3.barycenter — Gromov-Wasserstein barycenter construction.

Implements paper § 3.3:
  B* = argmin_B sum_i lambda_i * d_GW^2(B, X_i)

  Frechet mean in GW space. Two-phase alternating optimization:
    Phase A: fix B, solve the GW transport plans gamma_i for each X_i.
    Phase B: fix all gamma_i, update B via
        B <- sum_i gamma_i^T D_i gamma_i  (paper § 3.3 "Update D_B")
  Iterate until convergence. Restart 10x from different seeds to escape
  local minima (paper § 3.3).

Paper defaults (binding per CLAUDE.md):
  n_s = 40 support points, N_L = 300 FPS landmarks on the global 18k-corpus
  AoA. For law-scoped corpora (N ~ 20-50) we scale N_L down proportionally;
  this is flagged as a conditional ε/n_s calibration in plan.tex.

Thin wrapper over POT's `entropic_gromov_barycenters` for the outer solver;
FPS landmark selection + HDF5 caching of final barycenter are our own code.

HDF5 cache per memory `feedback_gw_always_cache`: the final barycenter and
its transport plans back to landmarks are always persisted.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import ot

from aoa_v3.gw import GWConfig, _gw_objective, gw_distance, normalize_cost


@dataclass(frozen=True)
class BarycenterConfig:
    """Barycenter hyperparameters. Paper defaults are binding (no MVP)."""

    n_supports: int = 40           # paper § 3.3: n_s = 40
    n_landmarks: int = 300         # paper § 3.3: 300 FPS landmarks on global AoA
    n_restarts: int = 10           # paper § 3.3: 10 random restarts
    epsilon: float = 0.008         # shared with GW; paper § 3.2 optimum
    max_iter: int = 1000
    tol: float = 1e-9
    random_state: int = 0
    n_workers: int = 1             # parallel restarts (operator-sanctioned up to 9)


# ---------------------------------------------------------- FPS landmarks


def fps_landmarks(
    distances: np.ndarray,
    n_landmarks: int,
    seed: int = 0,
) -> np.ndarray:
    """Farthest-point sampling over an upper-triangular distance array.

    Given a pairwise-distance matrix (n, n) between candidate attractors,
    greedily pick `n_landmarks` maximally-spread indices. Paper § 3.3 uses
    this on the Frobenius-distance-on-intra-distance-matrices space.

    The first landmark is drawn uniformly at random; subsequent ones maximize
    the minimum distance to all previously-selected landmarks.
    """
    distances = np.asarray(distances, dtype=np.float64)
    n = distances.shape[0]
    if distances.shape != (n, n):
        raise ValueError(f"distances must be square, got {distances.shape}")
    if n_landmarks > n:
        raise ValueError(f"cannot pick {n_landmarks} landmarks from {n} candidates")
    if n_landmarks < 1:
        raise ValueError("n_landmarks must be >= 1")

    rng = np.random.default_rng(seed)
    selected = np.empty(n_landmarks, dtype=np.int64)
    selected[0] = int(rng.integers(n))
    min_dists = distances[selected[0]].astype(np.float64, copy=True)
    done = np.zeros(n, dtype=bool)
    done[selected[0]] = True

    for i in range(1, n_landmarks):
        # argmax over candidates we haven't selected yet.
        # Using np.where here does not mutate min_dists (R16-MINOR).
        selected[i] = int(np.argmax(np.where(done, -np.inf, min_dists)))
        done[selected[i]] = True
        min_dists = np.minimum(min_dists, distances[selected[i]])

    return selected


def frobenius_distance_matrix(costs: list[np.ndarray]) -> np.ndarray:
    """Pairwise Frobenius distance between the upper-triangular parts of
    a list of intra-distance matrices.

    Memory: `O(K^2)` not `O(K^2 * max_len)` — uses the sklearn-style
    `||a||^2 + ||b||^2 - 2 a.b` identity via `sklearn.metrics.pairwise_distances`
    so we can run on 18k landmark candidates without blowing memory (R16).

    Mismatched `n_i` across inputs is handled by zero-padding to the max
    upper-triangular length; paper § 3.3 uses this for FPS-over-attractors.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    if not costs:
        raise ValueError("empty costs list")
    flats = []
    for C in costs:
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"each cost must be square 2-D, got {C.shape}")
        iu = np.triu_indices(C.shape[0], k=1)
        flats.append(C[iu])
    max_len = max(f.size for f in flats)
    padded = np.zeros((len(flats), max_len), dtype=np.float64)
    for i, f in enumerate(flats):
        padded[i, : f.size] = f
    return euclidean_distances(padded)


# ---------------------------------------------------------- barycenter solve


def _one_restart_worker(args):
    """Picklable single-restart barycenter solve for parallel execution."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    seed, n_s, costs_n, ps, p_bary, weights, epsilon, max_iter, tol = args
    init_rng = np.random.default_rng(seed)
    init_C = init_rng.random((n_s, n_s))
    init_C = 0.5 * (init_C + init_C.T)
    np.fill_diagonal(init_C, 0.0)
    K = len(costs_n)

    B, log = ot.gromov.entropic_gromov_barycenters(
        N=n_s,
        Cs=costs_n,
        ps=ps,
        p=p_bary,
        lambdas=weights,
        loss_fun="square_loss",
        epsilon=epsilon,
        symmetric=True,
        max_iter=max_iter,
        tol=tol,
        init_C=init_C,
        random_state=seed,
        log=True,
    )
    plans_from_log = log.get("T", None)
    obj = 0.0
    if plans_from_log is not None and len(plans_from_log) == K:
        for w, C, T_from in zip(weights, costs_n, plans_from_log):
            obj += float(w) * _gw_objective(B, C, T_from)
    else:
        gw_cfg = GWConfig(
            epsilon=epsilon, max_iter=max_iter, tol=tol, normalize_cost=False
        )
        for w, C in zip(weights, costs_n):
            d2, _, _ = gw_distance(B, C, cfg=gw_cfg, p=p_bary)
            obj += float(w) * d2
    return B, float(obj)


def gw_barycenter(
    costs: list[np.ndarray],
    cfg: Optional[BarycenterConfig] = None,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Compute the GW barycenter of a list of intra-distance matrices.

    Uses POT's `entropic_gromov_barycenters` (Phase A + Phase B iterated
    internally) with `n_restarts` random initializations; returns the
    restart with the lowest total GW objective.

    Returns (B, info) where:
      - B: (n_s, n_s) barycenter intra-distance matrix
      - info: {"objective", "n_iterations", "restarts_scores", "epsilon", "n_s"}
    """
    cfg = cfg or BarycenterConfig()
    if not costs:
        raise ValueError("need at least one cost matrix")

    # Normalize each cost matrix (paper § 3.2 convention shared with gw module).
    costs_n = [normalize_cost(C) for C in costs]

    n_s = cfg.n_supports
    K = len(costs_n)

    if weights is None:
        weights = np.full(K, 1.0 / K)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (K,):
        raise ValueError(f"weights must have shape ({K},), got {weights.shape}")
    # Fréchet-mean simplex constraint (paper § 3.3).
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative (Fréchet-mean simplex)")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"weights must sum to 1; got sum={weights.sum():.6g}")

    # Uniform marginals on each input and on the barycenter.
    ps = [np.full(C.shape[0], 1.0 / C.shape[0]) for C in costs_n]
    p_bary = np.full(n_s, 1.0 / n_s)

    master_rng = np.random.default_rng(cfg.random_state)
    restart_seeds = [int(master_rng.integers(1 << 30)) for _ in range(cfg.n_restarts)]

    if cfg.n_workers > 1 and cfg.n_restarts > 1:
        ctx = mp.get_context("spawn")
        work = [
            (seed, n_s, costs_n, ps, p_bary, weights, cfg.epsilon,
             cfg.max_iter, cfg.tol)
            for seed in restart_seeds
        ]
        with ProcessPoolExecutor(
            max_workers=min(cfg.n_workers, cfg.n_restarts), mp_context=ctx
        ) as pool:
            results = list(pool.map(_one_restart_worker, work, chunksize=1))
    else:
        results = [
            _one_restart_worker((
                seed, n_s, costs_n, ps, p_bary, weights, cfg.epsilon,
                cfg.max_iter, cfg.tol,
            ))
            for seed in restart_seeds
        ]

    best_B = None
    best_obj = np.inf
    restart_scores: list[float] = []
    for B_r, obj in results:
        restart_scores.append(obj)
        if obj < best_obj:
            best_obj = obj
            best_B = B_r

    # Explicit symmetrization (R07-MINOR): POT's symmetric=True gives B symmetric
    # up to floating roundoff (~1e-12); enforce it so downstream gw.py symmetry
    # check (atol=1e-10) does not trip.
    best_B = 0.5 * (best_B + best_B.T)

    info = {
        "objective": float(best_obj),
        "restarts_scores": list(restart_scores),
        "restart_seeds": list(restart_seeds),
        "epsilon": float(cfg.epsilon),
        "n_s": int(n_s),
        "n_inputs": int(K),
    }
    return best_B, info


# -------------------------------------------------- member transport plans


def member_transport_plans(
    costs: list[np.ndarray],
    B: np.ndarray,
    cfg: Optional[BarycenterConfig] = None,
) -> tuple[list[np.ndarray], list[float]]:
    """For each input cost matrix X_i, compute the entropic GW transport plan
    T_i from X_i to the barycenter B.

    Returns (plans, d2s).

    These plans are the central data object downstream — paper § 3.3 writes
    "transport plans themselves carry all of the information about the
    original attractors" and paper § 4 defines the sensitivity tensor in
    terms of T_ij.
    """
    cfg = cfg or BarycenterConfig()
    # B returned by `gw_barycenter` is already on the normalized-cost scale
    # (inputs were normalized before solve). Normalize every input C to match,
    # then pass to gw_distance with normalize_cost=False — consistent scaling
    # (R01-MINOR: fixes the asymmetric re-normalize of only C but not B).
    B = np.asarray(B, dtype=np.float64)
    costs_n = [normalize_cost(C) for C in costs]
    gw_cfg = GWConfig(
        epsilon=cfg.epsilon,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        normalize_cost=False,
    )
    plans = []
    d2s = []
    for C in costs_n:
        d2, T, _ = gw_distance(C, B, cfg=gw_cfg)
        plans.append(T)
        d2s.append(float(d2))
    return plans, d2s


# ------------------------------------------------------------ HDF5 cache


def _bary_cache_key(costs: list[np.ndarray], cfg: BarycenterConfig) -> str:
    """Content-addressed key for the barycenter problem."""
    h = hashlib.sha256()
    for C in costs:
        h.update(f"shape={C.shape}|".encode())
        h.update(np.ascontiguousarray(C).tobytes())
    h.update(
        f"n_s={cfg.n_supports}|n_L={cfg.n_landmarks}|restarts={cfg.n_restarts}"
        f"|eps={cfg.epsilon}|max_iter={cfg.max_iter}|tol={cfg.tol}"
        f"|seed={cfg.random_state}".encode()
    )
    return h.hexdigest()[:24]


def _compression_for(cells: int) -> dict:
    """Size-adaptive HDF5 compression (matches gw.py convention).

    gzip has fixed overhead that dominates for small arrays; only compress when
    the array exceeds ~4k cells. At n_s=40 (1600 cells) no compression is
    faster; at barycenter-of-barycenters scale (n≈300, 90k cells) gzip(4) wins.
    """
    if cells <= 4096:
        return {}
    return {"compression": "gzip", "compression_opts": 4}


def save_barycenter(
    path: Path,
    B: np.ndarray,
    info: dict,
    plans: Optional[list[np.ndarray]] = None,
    key: str = "bary",
    cache_key: Optional[str] = None,
) -> None:
    """Persist a barycenter (and optionally its member transport plans) to HDF5.

    NOTE on multi-writer safety: HDF5 is NOT multi-writer safe in default mode.
    Concurrent workers must write to per-law / per-key files (not a shared
    `barycenters.h5` across workers). See AoA-40b shard pattern for the
    established convention. A single-writer `"a"` (append) is safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        g.create_dataset("B", data=B, **_compression_for(B.size))
        g.attrs["objective"] = float(info.get("objective", np.nan))
        g.attrs["epsilon"] = float(info.get("epsilon", np.nan))
        g.attrs["n_s"] = int(info.get("n_s", 0))
        g.attrs["n_inputs"] = int(info.get("n_inputs", 0))
        if cache_key is not None:
            g.attrs["cache_key"] = cache_key
        scores = info.get("restarts_scores", [])
        if scores:
            g.create_dataset("restarts_scores", data=np.asarray(scores, dtype=np.float64))
        seeds = info.get("restart_seeds", [])
        if seeds:
            g.create_dataset("restart_seeds", data=np.asarray(seeds, dtype=np.int64))
        if plans is not None:
            plans_grp = g.create_group("plans")
            for i, T in enumerate(plans):
                plans_grp.create_dataset(
                    f"T_{i:05d}", data=T, **_compression_for(T.size)
                )


def load_barycenter(path: Path, key: str = "bary") -> tuple[np.ndarray, dict, Optional[list[np.ndarray]]]:
    """Load a barycenter (and its plans, if stored) from HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"no barycenter at key={key!r} in {path}")
        g = f[key]
        B = np.asarray(g["B"])
        info = {
            "objective": float(g.attrs.get("objective", np.nan)),
            "epsilon": float(g.attrs.get("epsilon", np.nan)),
            "n_s": int(g.attrs.get("n_s", 0)),
            "n_inputs": int(g.attrs.get("n_inputs", 0)),
        }
        if "restarts_scores" in g:
            info["restarts_scores"] = list(np.asarray(g["restarts_scores"]))
        if "restart_seeds" in g:
            info["restart_seeds"] = list(np.asarray(g["restart_seeds"]))
        if "cache_key" in g.attrs:
            info["cache_key"] = str(g.attrs["cache_key"])
        plans = None
        if "plans" in g:
            plans_grp = g["plans"]
            plans = [np.asarray(plans_grp[name]) for name in sorted(plans_grp.keys())]
    return B, info, plans
