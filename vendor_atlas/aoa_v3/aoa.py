"""
aoa_v3.aoa — Hellinger-DMAP on transport plans (Attractor-of-Attractors).

Implements paper § 3.3 end. Given a set of entropic GW transport plans
{T_i in R^{n_s x n_s}} between series attractors and the corpus barycenter,
construct the AoA coordinate system as follows:

  Step 1. Map each T_i to sqrt(T_i) (element-wise), embedding the Birkhoff
          polytope into a submanifold of the unit sphere.
  Step 2. Compute pairwise Hellinger distances:
            d_H(T_i, T_j) = || sqrt(T_i) - sqrt(T_j) ||_F
  Step 3. Construct an adaptive-bandwidth Gaussian kernel on d_H.
  Step 4. alpha = 1 normalization -> Laplace-Beltrami on the plan manifold.
  Step 5. Eigendecomposition yields the AoA diffusion coordinates.

Paper-binding defaults:
  alpha = 1 (Laplace-Beltrami), k = 10 adaptive kNN bandwidth.

The key difference from `aoa_v3.dmap` (per-series DMAP on attractor point
clouds) is the metric: here points are transport plans living in the Birkhoff
polytope, and the Hellinger distance amplifies small differences between
near-uniform plans by roughly `n_s / 2` vs. raw Frobenius (paper § 3.3).

No temporal Theiler window in this module — transport plans are indexed by
corpus members (not by time), so same-orbit correlation is not a concern.

Reuses `aoa_v3.dmap.dmap_dense` and helpers where possible; this module's
responsibility is the Hellinger-specific distance computation and the public
AoA-coordinate API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import squareform

from aoa_v3.dmap import (
    adaptive_bandwidth_from_dists,
    intrinsic_dim_lmethod,
    participation_ratio,
    spectral_gap,
)


@dataclass(frozen=True)
class AoAConfig:
    """AoA-coordinate hyperparameters. Paper defaults are binding (no MVP)."""

    n_components: int = 10
    alpha: float = 1.0         # paper § 3.3 end: Laplace-Beltrami on Birkhoff polytope
    k_adaptive: int = 10       # paper § 3.3 end via § 2.2 convention
    marginal_atol: float = 1e-6  # plan sum-to-1 tolerance (R01-MEDIUM)


# ---------------------------------------------------------- validation


def _validate_plans(plans: list[np.ndarray], marginal_atol: float = 1e-6) -> np.ndarray:
    """Validate a list of transport plans and return their stacked 3D array.

    Each plan must be:
      - 2-D, same shape across the list (paper § 3.3 n_s fixed)
      - finite, non-negative
      - sum to 1 within `marginal_atol` (doubly-stochastic probability plans
        per paper § 3.3; if this is violated upstream Sinkhorn broke)
    """
    if not plans:
        raise ValueError("empty plans list")
    arr = np.stack([np.asarray(T, dtype=np.float64) for T in plans], axis=0)
    if arr.ndim != 3:
        raise ValueError(f"plans must all be 2-D; got stacked shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("plans contain NaN or Inf")
    if np.any(arr < -1e-12):
        raise ValueError("plans must be non-negative (transport mass)")
    # Clip tiny negative roundoff to zero so sqrt is real.
    np.clip(arr, 0.0, None, out=arr)
    # Paper § 3.3: plans are doubly-stochastic so sum_{ij} T = 1.
    sums = arr.sum(axis=(1, 2))
    if not np.allclose(sums, 1.0, atol=marginal_atol):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(
            f"plan[{bad}] total mass = {sums[bad]:.6g}; expected 1.0 "
            f"(atol={marginal_atol}); upstream Sinkhorn may not have converged"
        )
    return arr


# ------------------------------------------- Hellinger distance over plans


def sqrt_embed(plans: list[np.ndarray], marginal_atol: float = 1e-6) -> np.ndarray:
    """Embed each plan into the unit sphere via sqrt(T) (paper § 3.3 Step 1).

    Returns a (K, n*m) array with each row the flattened element-wise
    square root of a plan. Since plans sum to 1, each row has unit Frobenius
    norm so rows lie on the unit sphere in R^{n*m}.
    """
    arr = _validate_plans(plans, marginal_atol=marginal_atol)
    K = arr.shape[0]
    return np.sqrt(arr).reshape(K, -1)


def hellinger_distance_matrix(
    plans: list[np.ndarray],
    marginal_atol: float = 1e-6,
) -> np.ndarray:
    """Pairwise Hellinger distances d_H(T_i, T_j) = ||sqrt(T_i) - sqrt(T_j)||_F.

    Uses the unit-sphere identity d² = 2 − 2·<x, y> (each sqrt-embedded plan
    has unit norm), which preserves precision for near-identical plans down
    to ~machine-epsilon scale — the naive `||x − y||² = ||x||² + ||y||² − 2xy`
    formulation loses ~half of double precision near the diagonal (R14-MEDIUM).

    Returns a (K, K) symmetric non-negative distance matrix with zero diagonal.
    """
    sqrt_flat = sqrt_embed(plans, marginal_atol=marginal_atol)
    # Each row has Frobenius norm 1 (plans sum to 1 -> sum sqrt(T)^2 = 1).
    gram = sqrt_flat @ sqrt_flat.T
    # d^2 = 2 - 2 gram (unit norms); clip small negatives from float roundoff.
    d2 = np.clip(2.0 - 2.0 * gram, 0.0, None)
    D = np.sqrt(d2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def frobenius_distance_matrix_on_plans(plans: list[np.ndarray]) -> np.ndarray:
    """Pairwise Frobenius distances ||T_i - T_j||_F. Used to compare against
    the Hellinger metric (paper § 3.3 claims roughly n_s/2 amplification).
    """
    arr = _validate_plans(plans)
    K = arr.shape[0]
    flat = arr.reshape(K, -1)
    n_sq = np.einsum("ij,ij->i", flat, flat)
    gram = flat @ flat.T
    d2 = np.clip(n_sq[:, None] + n_sq[None, :] - 2.0 * gram, 0.0, None)
    D = np.sqrt(d2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def frobenius_vs_hellinger_ratio(plans: list[np.ndarray]) -> dict:
    """Compare Frobenius vs Hellinger on the same plan set (paper § 3.3 claim
    that Hellinger amplifies differences by ~n_s/2 relative to Frobenius).

    Returns medians of both metrics (off-diagonal) and the ratio. A ratio
    substantially greater than 1 supports the Hellinger-necessity claim.
    """
    D_H = hellinger_distance_matrix(plans)
    D_F = frobenius_distance_matrix_on_plans(plans)
    iu = np.triu_indices(D_H.shape[0], k=1)
    med_H = float(np.median(D_H[iu]))
    med_F = float(np.median(D_F[iu]))
    ratio = med_H / med_F if med_F > 1e-12 else float("inf")
    return {
        "median_hellinger": med_H,
        "median_frobenius": med_F,
        "hellinger_over_frobenius": ratio,
    }


# ------------------------------------------------ Hellinger-kernel DMAP


def _dmap_from_distance_matrix(
    D: np.ndarray,
    n_components: int,
    alpha: float,
    k_adaptive: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run DMAP on a precomputed distance matrix D (no re-evaluation of point
    distances). Mirrors `aoa_v3.dmap.dmap_dense` but skips the kernel-from-X
    step since D is the input.

    Returns (coords, evals, aux) where aux includes bandwidth statistics for
    corner-proximity diagnostics (R12-MEDIUM).
    """
    n = D.shape[0]
    if n < 5:
        raise ValueError(f"need at least 5 transport plans for AoA; got K={n}")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains NaN or Inf")

    bw = adaptive_bandwidth_from_dists(D, k=k_adaptive, theiler=0)
    bw_outer = bw[:, None] * bw[None, :]
    K = np.exp(-(D**2) / bw_outer)
    if not np.all(np.isfinite(K)):
        raise ValueError("Hellinger-kernel produced non-finite entries; check plans")

    if alpha > 0:
        d_alpha_raw = K.sum(axis=1)
        d_alpha = np.maximum(d_alpha_raw, 1e-12 * np.median(d_alpha_raw)) ** alpha
        K = K / (d_alpha[:, None] * d_alpha[None, :])

    row_sums_raw = K.sum(axis=1)
    row_sums = np.maximum(row_sums_raw, 1e-12 * np.median(row_sums_raw))
    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / np.maximum(d_sqrt, 1e-12)
    S = K * (d_sqrt_inv[:, None] * d_sqrt_inv[None, :])
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    right = eigvecs * d_sqrt_inv[:, None]

    n_keep = min(n_components, n - 1)
    coords = right[:, 1 : n_keep + 1] * eigvals[1 : n_keep + 1][None, :]
    evals = eigvals[1 : n_keep + 1]
    aux = {
        "bandwidth_min": float(bw.min()),
        "bandwidth_max": float(bw.max()),
        "bandwidth_dynamic_range": float(bw.max() / max(bw.min(), 1e-30)),
    }
    return coords, evals, aux


def aoa_coordinates(
    plans: list[np.ndarray],
    cfg: Optional[AoAConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hellinger-DMAP AoA coordinates from a list of transport plans.

    Returns (coords, eigenvalues):
      - coords: (K, n_components) — AoA diffusion coordinates, one row per plan
      - eigenvalues: (n_components,) — diffusion spectrum excluding lambda_0 = 1

    For law-scoped runs we report observed eigenvalue structure without
    pegging d_eff to any theoretical number (per operator memory
    `feedback_deff_not_pegged_to_theory`).
    """
    cfg = cfg or AoAConfig()
    D = hellinger_distance_matrix(plans, marginal_atol=cfg.marginal_atol)
    coords, evals, _ = _dmap_from_distance_matrix(
        D, cfg.n_components, cfg.alpha, cfg.k_adaptive
    )
    return coords, evals


def _plan_entropy(T: np.ndarray) -> float:
    """Shannon entropy H(T) = -sum T log T over the full joint (flattened)."""
    flat = T.ravel()
    flat = flat[flat > 0]
    return float(-np.sum(flat * np.log(flat)))


# --------------------------------------------- nonlinearity diagnostics


def hellinger_dmap_diagnostics(
    plans: list[np.ndarray], cfg: Optional[AoAConfig] = None
) -> dict:
    """Compute the standard set of AoA structural diagnostics (paper + plan.tex
    Nonlinearity Demonstrations section).

    Returns a dict with:
      - n_plans: K
      - coords: (K, n_components)
      - eigenvalues: top n_components
      - d_eff_lmethod: L-method elbow
      - participation_ratio: PR = (Σλ)² / Σλ²
      - spectral_gap: λ_1 / λ_2
      - hellinger_over_frobenius: median d_H / median d_F (paper § 3.3 ~n_s/2 claim)
      - bandwidth_dynamic_range: max/min adaptive bandwidth (corner-proximity flag)
      - min_plan_entropy: entropy of the lowest-entropy plan (closest to the
        Birkhoff polytope vertices = deterministic couplings)
    """
    cfg = cfg or AoAConfig()
    D = hellinger_distance_matrix(plans, marginal_atol=cfg.marginal_atol)
    coords, evals, aux = _dmap_from_distance_matrix(
        D, cfg.n_components, cfg.alpha, cfg.k_adaptive
    )
    ratio = frobenius_vs_hellinger_ratio(plans)
    min_entropy = min(_plan_entropy(T) for T in plans)
    return {
        "n_plans": len(plans),
        "coords": coords,
        "eigenvalues": evals,
        "d_eff_lmethod": int(intrinsic_dim_lmethod(evals)),
        "participation_ratio": float(participation_ratio(evals)),
        "spectral_gap": float(spectral_gap(evals)),
        "hellinger_over_frobenius": ratio["hellinger_over_frobenius"],
        "bandwidth_dynamic_range": aux["bandwidth_dynamic_range"],
        "min_plan_entropy": min_entropy,
    }
