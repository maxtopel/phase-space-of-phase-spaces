"""
aoa_v3.dmap — Per-series diffusion maps.

Implements paper § 2.2:
  Step 1 — Adaptive Gaussian kernel:
      K_0(x_i, x_j) = exp(-||x_i - x_j||^2 / (sigma(x_i) sigma(x_j)))
      sigma(x_i) = distance to k-th nearest neighbor (k = 10 default)
  Step 2 — Theiler window: zero out K_0(x_i, x_j) for |i - j| <= w (default w = tau).
  Step 3 — alpha-normalization:
      K_alpha(x_i, x_j) = K_0(x_i, x_j) / (d_alpha(x_i)^alpha d_alpha(x_j)^alpha)
      alpha = 1 → Laplace-Beltrami operator on the underlying manifold.
  Step 4 — Markov matrix: P = K_alpha row-normalized.
  Step 5 — Eigendecomposition: 1 = lambda_0 >= lambda_1 >= ...
  Step 6 — Diffusion coordinates: Phi_t(x_i) = (lambda_1^t psi_1, ..., lambda_r^t psi_r).
  Step 7 — Intrinsic dim from L-method elbow on the eigenvalue spectrum
           (Salvador & Chen).

Paper binding defaults (per CLAUDE.md, no MVP):
  k = 10 adaptive NN, alpha = 1, Theiler = tau.

Ported from AoA v1 (`core/AoA/diffuse.py`); v3 drops the external `config`
dependency, enforces NaN guards, uses symmetric similarity transform for
numerical stability, and adds L-method elbow detection.

References:
  - Coifman & Lafon (2006) — Diffusion maps.
  - Theiler (1990) — Spurious dimension from correlation algorithms.
  - Salvador & Chen (2004) — Determining the number of clusters/segments (L-method).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class DmapConfig:
    """Diffusion-map hyperparameters. Paper defaults are binding (no MVP)."""

    n_components: int = 10
    alpha: float = 1.0            # paper § 2.2 Step 3 (Laplace-Beltrami)
    k_adaptive: int = 10          # paper § 2.2 Step 1 (adaptive bandwidth)
    theiler: int = -1             # -1 = auto-set to tau


# ---------------------------------------------------------------------- utils


def _require_finite(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or Inf")
    return X


def adaptive_bandwidth_from_dists(
    D: np.ndarray,
    k: int = 10,
    theiler: int = 0,
) -> np.ndarray:
    """Adaptive bandwidth from a precomputed pairwise-distance matrix.

    This is the mathematically-primary route when distances are already known
    (e.g. Hellinger on sqrt-embedded transport plans). Avoids the API pattern
    of threading a dummy point cloud through `adaptive_bandwidth` just to get
    a shape.
    """
    n = D.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"k={k} requires at least 2 rows; got n={n}")
    dists = D
    if theiler > 0:
        dists = dists.copy()
        dists[_theiler_mask(n, theiler)] = np.inf
    partitioned = np.partition(dists, k_eff, axis=1)
    bw = partitioned[:, k_eff]
    med = np.median(bw[np.isfinite(bw)])
    floor = max(1e-12 * med, 1e-12)
    return np.maximum(bw, floor)


def adaptive_bandwidth(
    X: np.ndarray,
    k: int = 10,
    dists: Optional[np.ndarray] = None,
    theiler: int = 0,
) -> np.ndarray:
    """Per-point bandwidth = distance to k-th nearest neighbor (paper § 2.2 Step 1).

    If `dists` is given (the precomputed pairwise Euclidean distance matrix),
    it is used directly. Otherwise it is computed from `X`.

    If `theiler > 0`, pairs with |i-j| <= theiler are masked to +inf before
    the k-NN search, so temporally-close neighbors (same-orbit points) don't
    collapse the bandwidth (Berry & Harlim 2016).

    Uses `np.partition` (O(n) per row) rather than full sort.

    Floor is *relative* to median bandwidth to remain scale-invariant.
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"k_adaptive={k} requires at least 2 points; got n={n}")

    if dists is None:
        dists = cdist(X, X)

    if theiler > 0:
        mask = _theiler_mask(n, theiler)
        dists = dists.copy()
        dists[mask] = np.inf

    # k_eff-th nearest neighbor via partition (ignores self at index 0 naturally
    # since self-distance is 0 = minimum).
    partitioned = np.partition(dists, k_eff, axis=1)
    bw = partitioned[:, k_eff]

    # Relative floor (R14-MEDIUM).
    med = np.median(bw[np.isfinite(bw)])
    floor = max(1e-12 * med, 1e-12)
    return np.maximum(bw, floor)


def _theiler_mask(n: int, w: int) -> np.ndarray:
    """Boolean mask marking pairs within |i-j| <= w (including diagonal).

    These pairs get K_0 zeroed per paper § 2.2 Step 2.
    """
    if w <= 0:
        mask = np.eye(n, dtype=bool)
    else:
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        mask = np.abs(i - j) <= w
    return mask


# -------------------------------------------------------------- dense DMAP


def dmap_dense(
    X: np.ndarray,
    n_components: int = 10,
    alpha: float = 1.0,
    k_adaptive: int = 10,
    theiler: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense diffusion map (paper § 2.2 Steps 1–6).

    Returns (coords, eigenvalues) of the nontrivial (non-trivial-1) part.
    coords[:, i] = lambda_{i+1} * psi_{i+1}(x)   [t = 1]
    """
    X = _require_finite(X, "X")
    n = X.shape[0]
    if n < 5:
        raise ValueError(f"need at least 5 points for DMAP; got n={n}")
    n_components = min(n_components, n - 1)

    # Pairwise distances — computed once and shared with adaptive_bandwidth.
    dists = cdist(X, X)
    bw = adaptive_bandwidth(X, k_adaptive, dists=dists, theiler=theiler)

    # Step 1: adaptive-bandwidth Gaussian kernel.
    bw_outer = bw[:, None] * bw[None, :]
    K = np.exp(-(dists**2) / bw_outer)

    if not np.all(np.isfinite(K)):
        raise ValueError("Gaussian kernel produced non-finite entries; rescale X")

    # Step 2: Theiler window — zero out pairs with |i-j| <= theiler
    # (paper § 2.2 Step 2: "whenever |i - j| <= w"; inclusive per paper text).
    if theiler > 0:
        K[_theiler_mask(n, theiler)] = 0.0

    # Step 3: alpha-normalization (d_alpha = row sum of K_0).
    if alpha > 0:
        d_alpha_raw = K.sum(axis=1)
        d_alpha = np.maximum(d_alpha_raw, 1e-12 * np.median(d_alpha_raw)) ** alpha
        K = K / (d_alpha[:, None] * d_alpha[None, :])

    # Step 4: Markov row-normalization (stable via symmetric similarity).
    row_sums_raw = K.sum(axis=1)
    row_sums = np.maximum(row_sums_raw, 1e-12 * np.median(row_sums_raw))
    # S = D^{-1/2} K D^{-1/2} is symmetric with the same spectrum as P = D^{-1} K.
    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / np.maximum(d_sqrt, 1e-12)
    S = K * (d_sqrt_inv[:, None] * d_sqrt_inv[None, :])
    S = 0.5 * (S + S.T)

    # Step 5: eigendecomposition. `eigh` returns ascending; reverse for descending.
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Right eigenvectors of P: psi_k = eigvec_k * d^{-1/2}
    right = eigvecs * d_sqrt_inv[:, None]

    # Step 6: diffusion coordinates, drop trivial lambda_0 = 1.
    coords = right[:, 1 : n_components + 1] * eigvals[1 : n_components + 1][None, :]
    evals = eigvals[1 : n_components + 1]
    return coords, evals


def dmap_fit_train(
    X: np.ndarray,
    n_components: int = 10,
    alpha: float = 1.0,
    k_adaptive: int = 10,
    theiler: int = 0,
) -> dict:
    """Fit DMAP on training data and return all parameters needed for
    Nyström OOS extension.

    Returns a dict with: X_train (n, d), bw_train (n,) per-point bandwidths,
    right_train (n, n_components+1) right-eigenvectors including the trivial,
    evals_nontrivial (n_components,), d_alpha_train (n,) alpha-normalization
    denominator, alpha, k_adaptive, n_components, coords (n, n_components)
    for quick training-side access.
    """
    X = _require_finite(X, "X")
    n = X.shape[0]
    if n < 5:
        raise ValueError(f"need at least 5 points for DMAP; got n={n}")
    n_components = min(n_components, n - 1)

    dists = cdist(X, X)
    bw = adaptive_bandwidth(X, k_adaptive, dists=dists, theiler=theiler)
    bw_outer = bw[:, None] * bw[None, :]
    K0 = np.exp(-(dists**2) / bw_outer)
    if theiler > 0:
        K0 = K0.copy()
        K0[_theiler_mask(n, theiler)] = 0.0

    if alpha > 0:
        d_alpha_raw = K0.sum(axis=1)
        d_alpha = np.maximum(d_alpha_raw, 1e-12 * np.median(d_alpha_raw)) ** alpha
        K = K0 / (d_alpha[:, None] * d_alpha[None, :])
    else:
        d_alpha = np.ones(n)
        K = K0

    row_sums_raw = K.sum(axis=1)
    row_sums = np.maximum(row_sums_raw, 1e-12 * np.median(row_sums_raw))
    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / np.maximum(d_sqrt, 1e-12)
    S = K * (d_sqrt_inv[:, None] * d_sqrt_inv[None, :])
    S = 0.5 * (S + S.T)

    eigvals_full, eigvecs_full = np.linalg.eigh(S)
    eigvals_full = eigvals_full[::-1]
    eigvecs_full = eigvecs_full[:, ::-1]

    right = eigvecs_full * d_sqrt_inv[:, None]

    coords = right[:, 1 : n_components + 1] * eigvals_full[1 : n_components + 1][None, :]
    evals_nontrivial = eigvals_full[1 : n_components + 1]

    return {
        "X_train": X,
        "bw_train": bw,
        "right_train": right[:, : n_components + 1],
        "eigvals_full_top": eigvals_full[: n_components + 1],
        "evals_nontrivial": evals_nontrivial,
        "d_alpha_train": d_alpha,
        "alpha": alpha,
        "k_adaptive": k_adaptive,
        "n_components": n_components,
        "theiler": theiler,
        "coords_train": coords,
    }


def dmap_nystrom_extend(
    X_oos: np.ndarray,
    fit: dict,
) -> np.ndarray:
    """Nyström-extend DMAP coordinates to out-of-sample points.

    Standard kernel Nyström: for a new point x*, compute the alpha-normalized
    Markov kernel to each training point and project onto the training
    eigenvectors:

        psi_k(x*) = (1 / lambda_k) * sum_j K_row(x*, X_train[j]) * psi_k(X_train[j])

    where K_row is the row-normalized alpha-normalized kernel of x* against
    training points, using x*'s adaptive bandwidth estimated from the kth
    nearest training-point distance. Returns coords (n_oos, n_components)
    with the trivial eigenvector dropped, matching `dmap_dense` convention.

    Reference: Coifman & Lafon (2006); Bengio et al. (2004) NIPS for the
    Nyström consistency argument.
    """
    X_oos = np.asarray(X_oos, dtype=np.float64)
    if X_oos.ndim != 2:
        raise ValueError(f"X_oos must be 2-D, got shape {X_oos.shape}")
    X_tr = fit["X_train"]
    bw_tr = fit["bw_train"]
    d_alpha_tr = fit["d_alpha_train"]
    right_tr = fit["right_train"]           # (n_train, n_components+1)
    eigvals_top = fit["eigvals_full_top"]   # (n_components+1,)
    alpha = fit["alpha"]
    k = fit["k_adaptive"]
    n_comp = fit["n_components"]

    if X_oos.shape[1] != X_tr.shape[1]:
        raise ValueError(
            f"OOS dim mismatch: X_oos has {X_oos.shape[1]} cols, train has {X_tr.shape[1]}"
        )

    # Step 1: adaptive bandwidth for each OOS point = kth-NN distance to train
    d_oos_tr = cdist(X_oos, X_tr)  # (n_oos, n_train)
    k_eff = min(k, X_tr.shape[0] - 1)
    bw_oos = np.partition(d_oos_tr, k_eff - 1, axis=1)[:, k_eff - 1]
    bw_oos = np.maximum(bw_oos, 1e-12 * np.median(bw_tr))

    # Step 2: Gaussian kernel OOS -> train with (oos_bw, train_bw) outer product
    bw_outer = bw_oos[:, None] * bw_tr[None, :]     # (n_oos, n_train)
    K0 = np.exp(-(d_oos_tr**2) / bw_outer)

    # Step 3: alpha-normalize using train-side d_alpha (train) and oos-side
    # d_alpha computed from OOS-to-train sums (consistent with Coifman-Lafon
    # OOS extension: alpha denominator on train uses train d_alpha; on OOS
    # uses OOS-to-train sum).
    if alpha > 0:
        d_alpha_oos_raw = K0.sum(axis=1)
        d_alpha_oos = np.maximum(d_alpha_oos_raw, 1e-12 * np.median(d_alpha_oos_raw)) ** alpha
        K = K0 / (d_alpha_oos[:, None] * d_alpha_tr[None, :])
    else:
        K = K0

    # Step 4: row-normalize into Markov P_oos(oos, train)
    row_sums_raw = K.sum(axis=1)
    row_sums = np.maximum(row_sums_raw, 1e-12 * np.median(row_sums_raw))
    P = K / row_sums[:, None]

    # Step 5: Nyström projection. The eigenvector equation P psi_k = lam_k psi_k
    # gives psi_k(x*) = (1/lam_k) * (P_oos @ psi_k). Training coords use the
    # convention `coords = psi_k * lam_k`, so:
    #   coords(x*) = lam_k * psi_k(x*) = lam_k * (1/lam_k) * (P_oos @ psi_k)
    #             = P_oos @ psi_k = (P_oos @ right_tr)[:, k]
    # No extra eigval multiplication is required.
    proj = P @ right_tr                      # (n_oos, n_components+1)
    coords_oos = proj[:, 1 : n_comp + 1]
    return coords_oos


# ----------------------------------------------------------- L-method elbow


def _segment_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """RMSE of a linear fit y ~ a*x + b. Returns inf if fewer than 2 points."""
    if len(x) < 2:
        return np.inf
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def intrinsic_dim_lmethod(eigvals: np.ndarray, min_components: int = 2) -> int:
    """Estimate intrinsic dimensionality via the L-method elbow (Salvador-Chen 2004).

    Fits two linear segments to `(k, lambda_k)` for k = 1..n and picks the split
    `c in [2, n-2]` minimizing the Salvador-Chen weighted RMSE:

        RMSE_c = ((c - 1) / (n - 2)) * RMSE(L_c) + ((n - c - 1) / (n - 2)) * RMSE(R_c)

    where L_c = indices 1..c and R_c = indices c+1..n. Weights are segment
    lengths minus one (degrees of freedom of each linear fit).

    Falls back to `min_components` for very short spectra.
    """
    vals = np.asarray(eigvals, dtype=np.float64).ravel()
    n = len(vals)
    if n < 4:
        return max(min_components, n)

    indices = np.arange(1, n + 1, dtype=np.float64)
    best_c = min_components
    best_rmse = np.inf
    denom = float(n - 2)

    for c in range(2, n - 1):
        r_a = _segment_rmse(indices[:c], vals[:c])
        r_b = _segment_rmse(indices[c:], vals[c:])
        weighted = ((c - 1) * r_a + (n - c - 1) * r_b) / denom
        if weighted < best_rmse:
            best_rmse = weighted
            best_c = c

    return max(min_components, min(best_c, n))


def spectral_gap(eigvals: np.ndarray) -> float:
    """Ratio lambda_1 / lambda_2 (first non-trivial to second).

    Large spectral gap → well-defined intrinsic dimensionality. Returns 0 if
    fewer than 2 eigenvalues or denominator too small.
    """
    vals = np.asarray(eigvals).ravel()
    if len(vals) < 2 or abs(vals[1]) < 1e-12:
        return 0.0
    return float(vals[0] / vals[1])


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2) — a linear intrinsic-dim proxy.

    Used in the nonlinearity tests (paper plan.tex §Nonlinearity Demonstrations):
    PR/d_eff > 1.5 indicates genuine nonlinear compression.
    """
    vals = np.asarray(eigvals, dtype=np.float64).ravel()
    if len(vals) == 0:
        return 0.0
    s1 = float(vals.sum())
    s2 = float((vals**2).sum())
    if s2 < 1e-12:
        return 0.0
    return s1 * s1 / s2


# -------------------------------------------------------- convenience API


def dmap_series(
    dvecs: np.ndarray,
    cfg: Optional[DmapConfig] = None,
    tau: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diffusion map for a single series' delay vectors.

    Paper-binding defaults: alpha=1, k=10. Theiler defaults to tau.
    """
    cfg = cfg or DmapConfig()
    theiler = cfg.theiler if cfg.theiler >= 0 else tau
    return dmap_dense(
        dvecs,
        n_components=cfg.n_components,
        alpha=cfg.alpha,
        k_adaptive=cfg.k_adaptive,
        theiler=theiler,
    )


def intra_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Intra-series Euclidean distance matrix in diffusion coordinates.

    This matrix is the input to the GW alignment step (paper § 3).
    """
    return cdist(coords, coords)
