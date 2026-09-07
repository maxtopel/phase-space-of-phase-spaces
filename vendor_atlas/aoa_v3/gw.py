"""
aoa_v3.gw — Entropic Gromov-Wasserstein transport.

Implements paper § 3.1-3.2:
  d_GW^2(X, Y) = inf_{gamma in Pi(mu_X, mu_Y)}
                 int |d_X(x,x') - d_Y(y,y')|^2 dgamma(x,y) dgamma(x',y')

  Entropic regularization (paper § 3.2):
    d_GW^{2, epsilon}(X, Y) = inf_{gamma in Pi} { ... + epsilon * H(gamma) }
  solved via projected gradient descent with inner Sinkhorn iterations.

Paper-binding default (§ 3.2): epsilon = 0.008. This value was empirically
optimized on v1's 18k-corpus global AoA. Law-scoped corpora may need
recalibration — see plan.tex Key Design Decision on ε/n_s.

Paper § 3.2's empirically-optimal epsilon = 0.008 is scale-sensitive: the
entropic term `epsilon * H(gamma)` competes with the quadratic cost which
scales as max(C)^2. We therefore normalize each intra-distance matrix to
unit max before the GW solve so epsilon has consistent meaning across
corpora. This is an implementation convention supporting the paper's
corpus-size invariance claim (§ 3.2), not a paper axiom of the GW definition
itself.

All GW solves write to an HDF5 cache per memory `feedback_gw_always_cache`
(no transient GW — pilots included).

Thin wrapper over POT (`ot.gromov.entropic_gromov_wasserstein`) — no reinvention
of Sinkhorn.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import ot


@dataclass(frozen=True)
class GWConfig:
    """GW hyperparameters. Paper defaults are binding (no MVP)."""

    epsilon: float = 0.008          # paper § 3.2 empirical optimum (v1 default)
    max_iter: int = 1000
    tol: float = 1e-9
    loss_fun: str = "square_loss"   # paper eq. uses |d_X - d_Y|^2
    normalize_cost: bool = True     # divide each C by its max before solve


# --------------------------------------------------------------------- utils


def _require_cost_matrix(C: np.ndarray, name: str = "C") -> np.ndarray:
    """Validate that C is a finite symmetric square distance-like matrix.

    Paper § 3.1 requires (X, d_X, mu_X) to be a metric-measure space, so C
    must be symmetric. POT's solver will run on asymmetric C but the
    resulting d_GW is not a valid pseudo-metric on isometry classes.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name} must be square 2-D, got shape {C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError(f"{name} contains NaN or Inf")
    # Symmetry check with float tolerance. DMAP/diffusion distances are
    # symmetric by construction; asymmetric C indicates caller bug.
    if C.shape[0] > 1 and not np.allclose(C, C.T, atol=1e-10, rtol=1e-8):
        asym = float(np.abs(C - C.T).max())
        raise ValueError(
            f"{name} is not symmetric (max |C - C.T| = {asym:.3e}); "
            "GW requires a metric-measure space"
        )
    return C


def normalize_cost(C: np.ndarray) -> np.ndarray:
    """Rescale C to unit max. Returns a copy."""
    C = _require_cost_matrix(C, "C")
    m = float(np.abs(C).max())
    if m < 1e-12:
        raise ValueError("cost matrix is ~zero; cannot normalize")
    return C / m


def uniform_marginals(n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform probability marginals on the two point sets.

    Paper § 3.3 uses uniform-marginal couplings on barycenter supports.
    """
    return np.full(n, 1.0 / n), np.full(m, 1.0 / m)


# ------------------------------------------------------------- GW solve


def gw_distance(
    C1: np.ndarray,
    C2: np.ndarray,
    cfg: Optional[GWConfig] = None,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray, dict]:
    """Entropic Gromov-Wasserstein distance + transport plan between two
    metric-measure spaces described by intra-distance matrices C1, C2.

    Returns (d_gw_squared, plan, info) where:
      - d_gw_squared: the optimal GW quadratic objective value
        (int |d_X - d_Y|^2 d gamma d gamma), NOT including the entropic term
      - plan: (n, m) doubly-scaled transport plan with marginals (p, q)
      - info: solver metadata (epsilon, iterations, converged)

    Uses POT's `entropic_gromov_wasserstein` under the hood (projected gradient
    descent with Sinkhorn inner loop).
    """
    cfg = cfg or GWConfig()
    C1 = _require_cost_matrix(C1, "C1")
    C2 = _require_cost_matrix(C2, "C2")

    if cfg.normalize_cost:
        C1 = normalize_cost(C1)
        C2 = normalize_cost(C2)

    n, m = C1.shape[0], C2.shape[0]
    if p is None or q is None:
        p, q = uniform_marginals(n, m)

    plan, log = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        p=p,
        q=q,
        loss_fun=cfg.loss_fun,
        epsilon=cfg.epsilon,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        log=True,
    )

    if not np.all(np.isfinite(plan)):
        raise RuntimeError(
            "GW solver produced non-finite plan: "
            f"epsilon={cfg.epsilon}, n1={n}, n2={m}, "
            f"plan_min={np.nanmin(plan):.3e}, plan_max={np.nanmax(plan):.3e}; "
            "try larger epsilon or check cost normalization"
        )

    # POT returns the GW quadratic objective in log["gw_dist"] when available.
    # Its exact definition has varied across POT versions (some include the
    # entropic term, some don't), so we ALSO compute the direct Frobenius-form
    # objective and log the discrepancy. Prefer log["gw_dist"] for the
    # returned value when available (it uses the solver's own accumulated
    # loss, free of downstream subnormal-matmul noise from plan entries
    # near 1e-25) but expose the delta for downstream audit.
    d2_direct = _gw_objective(C1, C2, plan)  # raises on non-finite
    if "gw_dist" in log:
        d2 = float(log["gw_dist"])
        gw_dist_discrepancy = abs(d2 - d2_direct)
    else:
        d2 = d2_direct
        gw_dist_discrepancy = 0.0

    err = list(log.get("err", [])) if "err" in log else []
    # POT's err is the change in plan between outer iters (not absolute loss);
    # converged iff the last err < tol AND we did not hit max_iter.
    iterations = len(err) if err else cfg.max_iter
    converged = bool(err[-1] < cfg.tol) if err else True
    if iterations >= cfg.max_iter:
        converged = False
    info = {
        "epsilon": float(cfg.epsilon),
        "iterations": iterations,
        "converged": converged,
        "gw_dist_discrepancy": float(gw_dist_discrepancy),
        "log_keys": list(log.keys()),
        "normalized_cost": cfg.normalize_cost,
    }
    return float(d2), plan, info


def _gw_objective(C1: np.ndarray, C2: np.ndarray, plan: np.ndarray) -> float:
    """Direct evaluation of sum_{i,j,k,l} (C1[i,k] - C2[j,l])^2 * plan[i,j] * plan[k,l].

    Expand the squared difference:
      (C1[i,k] - C2[j,l])^2 = C1[i,k]^2 - 2 C1[i,k] C2[j,l] + C2[j,l]^2
    so the full sum factorizes as
      d^2 = <C1.^2, p p^T>_F  -  2 <C1, plan @ C2 @ plan.T>_F  +  <C2.^2, q q^T>_F
    with p = plan @ 1_m and q = plan.T @ 1_n (Peyre 2016, eq. 6).
    """
    # Sinkhorn plans can have entries ~1e-25 (subnormals). LAPACK gemm on
    # subnormals emits harmless over/invalid/divide RuntimeWarnings while
    # returning finite products. We silence all three across every gemm in
    # this function (term1, term2, term3 all ultimately contract with subnormal
    # plan-derived quantities via `p = plan.sum(...)`). The post-block
    # `np.isfinite(result)` check still surfaces genuine non-finite bugs
    # (NaN in C1/C2/plan via an upstream path that bypassed validation).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = plan.sum(axis=1)
        q = plan.sum(axis=0)
        term1 = float(p @ (C1 * C1) @ p)
        term3 = float(q @ (C2 * C2) @ q)
        pCp = plan @ C2 @ plan.T
        term2 = float(2.0 * np.einsum("ij,ij->", C1, pCp))
    result = term1 - term2 + term3
    if not np.isfinite(result):
        raise RuntimeError(
            f"GW objective non-finite: term1={term1}, term2={term2}, "
            f"term3={term3}, plan range=[{plan.min():.3e}, {plan.max():.3e}]"
        )
    return result


# ------------------------------------------------------------- HDF5 cache


def _cache_key(C1: np.ndarray, C2: np.ndarray, cfg: GWConfig) -> str:
    """Content-addressed key for (C1, C2, epsilon, max_iter, normalize, loss).

    Shape is included in the hash to eliminate any ambiguity when
    concatenating raw buffers of different-sized matrices.
    """
    h = hashlib.sha256()
    h.update(f"shape1={C1.shape}|shape2={C2.shape}|".encode())
    h.update(np.ascontiguousarray(C1).tobytes())
    h.update(np.ascontiguousarray(C2).tobytes())
    h.update(f"{cfg.epsilon}|{cfg.max_iter}|{cfg.tol}|{cfg.normalize_cost}|{cfg.loss_fun}".encode())
    return h.hexdigest()[:24]


def gw_distance_cached(
    C1: np.ndarray,
    C2: np.ndarray,
    cache_path: Path,
    cfg: Optional[GWConfig] = None,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray, dict]:
    """Cached GW solve. Writes every GW result to HDF5.

    Memory `feedback_gw_always_cache`: every GW solve writes to disk; no transient
    GW, pilots included.

    Cache layout (HDF5):
        /<key>/plan          (n, m) float64
        /<key>/d_gw_squared  scalar
        /<key>/epsilon       scalar
        /<key>/n1, /n2       scalars
        /<key>/converged     scalar (u1)

    NOTE: HDF5 files are NOT multi-writer safe. For concurrent workers,
    use per-worker or per-window shards (see AoA-40b pattern).
    """
    cfg = cfg or GWConfig()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = _cache_key(C1, C2, cfg)

    if cache_path.exists():
        with h5py.File(cache_path, "r") as f:
            if key in f:
                g = f[key]
                plan = np.asarray(g["plan"])
                d2 = float(g["d_gw_squared"][()])
                info = {
                    "epsilon": float(g["epsilon"][()]),
                    "converged": bool(g["converged"][()]),
                    "cache_hit": True,
                    "key": key,
                }
                return d2, plan, info

    d2, plan, info = gw_distance(C1, C2, cfg=cfg, p=p, q=q)
    info["cache_hit"] = False
    info["key"] = key

    # Compression policy: gzip(level=4) on a 40x40 float64 plan (12.8 KB raw)
    # costs ~0.5-2ms per call for negligible disk savings. At n <= 64 the
    # compression overhead dominates I/O; use cheap lzf above that only if
    # it helps. For barycenter-scale arrays (n ~ 300) gzip pays off, so we
    # switch policy on plan size.
    plan_cells = plan.shape[0] * plan.shape[1]
    if plan_cells <= 64 * 64:
        compression = None
        compression_opts = None
    else:
        compression = "gzip"
        compression_opts = 4

    with h5py.File(cache_path, "a") as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        if compression is None:
            g.create_dataset("plan", data=plan)
        else:
            g.create_dataset(
                "plan", data=plan, compression=compression, compression_opts=compression_opts
            )
        g.create_dataset("d_gw_squared", data=d2)
        g.create_dataset("epsilon", data=cfg.epsilon)
        g.create_dataset("n1", data=C1.shape[0])
        g.create_dataset("n2", data=C2.shape[0])
        g.create_dataset("converged", data=np.uint8(info["converged"]))

    return d2, plan, info
