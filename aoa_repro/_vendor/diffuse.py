"""
GW Attractor Landscape — Per-Series Diffusion Maps.

Status: ACTIVE

Implements Coifman-Lafon (2006) diffusion maps on attractor point clouds
(delay vectors). Each series gets its own DMAP.

Key design choices:
- alpha=1.0 (Laplace-Beltrami normalization, removes sampling density bias)
- Adaptive bandwidth: per-point k-th NN distance
- Theiler temporal separation window (reduces temporal correlation in kernel)
- Dense eigendecomposition for N <= 2000 (99.5% of corpus)
- FPS + Nystrom only for the ~0.5% of long series with N > 2000

Theory gap (acknowledged): Coifman-Lafon convergence (Ch14 Thm 14.diffusion-laplacian)
requires smooth manifold + i.i.d. sampling. Attractor point clouds are fractal +
temporally correlated. Neither holds. DMAP is a principled heuristic for attractor
point clouds, not a formally guaranteed recovery. Mitigations:
  1. Theiler window reduces temporal correlation
  2. alpha=1.0 removes density bias
  3. Validated on dysts systems (Lorenz, Rossler) where true structure is known
  4. Diffusion distance is a valid metric for ANY graph Laplacian

References:
- Coifman & Lafon (2006): Diffusion maps. Applied and Computational Harmonic Analysis.
- Theiler (1990): Spurious dimension from correlation algorithms applied to limited time series data.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import cdist, squareform, pdist

from config import DmapConfig


def adaptive_bandwidth(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute adaptive bandwidth: distance to k-th nearest neighbor for each point.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Point cloud.
    k : int
        k-th neighbor distance to use.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-point bandwidth.
    """
    n = X.shape[0]
    k = min(k, n - 1)

    dists = cdist(X, X)
    # Sort each row; column 0 is self (=0), column k is k-th neighbor
    sorted_dists = np.sort(dists, axis=1)
    bandwidth = sorted_dists[:, k]

    # Floor to avoid division by zero
    bandwidth = np.maximum(bandwidth, 1e-10)
    return bandwidth


def apply_theiler_window(dists: np.ndarray, theiler: int) -> np.ndarray:
    """
    Apply Theiler temporal separation window.

    Sets distances between temporally close points to infinity,
    preventing them from being neighbors in the kernel.

    Parameters
    ----------
    dists : np.ndarray, shape (n, n)
        Distance matrix.
    theiler : int
        Window size. Points within |i - j| <= theiler are excluded.

    Returns
    -------
    np.ndarray
        Modified distance matrix.
    """
    if theiler <= 0:
        return dists

    n = dists.shape[0]
    result = dists.copy()
    for offset in range(1, theiler + 1):
        idx = np.arange(n - offset)
        result[idx, idx + offset] = np.inf
        result[idx + offset, idx] = np.inf

    return result


def dmap_dense(X: np.ndarray, n_components: int = 10, alpha: float = 1.0,
               k_adaptive: int = 10, theiler: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense diffusion map computation.

    Steps:
    1. Compute pairwise distances
    2. Apply Theiler window
    3. Compute adaptive-bandwidth Gaussian kernel
    4. Alpha-normalize (Laplace-Beltrami for alpha=1.0)
    5. Row-normalize to Markov matrix
    6. Eigendecompose
    7. Return diffusion coordinates (eigenvalue-weighted eigenvectors)

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Point cloud (delay vectors).
    n_components : int
        Number of diffusion coordinates to return.
    alpha : float
        Density normalization. 1.0 = Laplace-Beltrami.
    k_adaptive : int
        k for adaptive bandwidth.
    theiler : int
        Theiler window size.

    Returns
    -------
    coords : np.ndarray, shape (n, n_components)
        Diffusion coordinates (eigenvalue-weighted).
    eigenvalues : np.ndarray, shape (n_components,)
        Eigenvalues (excluding trivial λ=1).
    """
    n = X.shape[0]
    n_components = min(n_components, n - 1)

    # Step 1: Pairwise distances
    dists = cdist(X, X)

    # Step 2: Theiler window
    if theiler > 0:
        dists = apply_theiler_window(dists, theiler)

    # Step 3: Adaptive-bandwidth Gaussian kernel
    bw = adaptive_bandwidth(X, k_adaptive)
    # K(i,j) = exp(-d(i,j)^2 / (bw[i] * bw[j]))
    bw_outer = bw[:, None] * bw[None, :]
    K = np.exp(-dists ** 2 / bw_outer)
    # Zero out self for Theiler-excluded pairs (inf dist → 0)
    K[~np.isfinite(K)] = 0.0

    # Step 4: Alpha-normalization
    if alpha > 0:
        d_alpha = K.sum(axis=1) ** alpha
        d_alpha = np.maximum(d_alpha, 1e-10)
        K = K / (d_alpha[:, None] * d_alpha[None, :])

    # Step 5: Row-normalize to Markov matrix
    row_sums = K.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-10)
    P = K / row_sums[:, None]

    # Step 6: Eigendecompose
    # Use symmetric similarity transform for numerical stability:
    # D^{-1/2} K D^{-1/2} is symmetric
    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / np.maximum(d_sqrt, 1e-10)
    S = P * d_sqrt[:, None] * d_sqrt_inv[None, :]
    S = 0.5 * (S + S.T)  # enforce symmetry

    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform back: right eigenvectors of P
    right_evecs = eigenvectors * d_sqrt_inv[:, None]

    # Step 7: Diffusion coordinates (skip trivial eigenvalue=1)
    # coords[:, k] = lambda_{k+1} * psi_{k+1}
    coords = right_evecs[:, 1:n_components + 1] * eigenvalues[1:n_components + 1][None, :]
    evals_out = eigenvalues[1:n_components + 1]

    return coords, evals_out


def subsample_fps(X: np.ndarray, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Farthest Point Sampling (FPS) to select n_target representative points.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    n_target : int

    Returns
    -------
    X_sub : np.ndarray, shape (n_target, d)
    indices : np.ndarray, shape (n_target,)
    """
    n = X.shape[0]
    if n <= n_target:
        return X, np.arange(n)

    indices = np.zeros(n_target, dtype=int)
    indices[0] = 0  # start from first point
    min_dists = cdist(X[0:1], X)[0]

    for i in range(1, n_target):
        indices[i] = np.argmax(min_dists)
        new_dists = np.linalg.norm(X - X[indices[i]], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return X[indices], indices


def dmap_nystrom(X: np.ndarray, n_landmarks: int = 500,
                 n_components: int = 10, alpha: float = 1.0,
                 k_adaptive: int = 10, theiler: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nystrom-extended diffusion map for large point clouds (N > 2000).

    1. FPS to select landmarks
    2. Dense DMAP on landmarks
    3. Nystrom extension to all points

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    n_landmarks : int
        Number of FPS landmarks.
    n_components : int
    alpha : float
    k_adaptive : int
    theiler : int

    Returns
    -------
    coords : np.ndarray, shape (n, n_components)
    eigenvalues : np.ndarray, shape (n_components,)
    """
    n = X.shape[0]
    X_lm, lm_idx = subsample_fps(X, n_landmarks)

    # Dense DMAP on landmarks
    coords_lm, evals = dmap_dense(X_lm, n_components, alpha, k_adaptive, theiler)

    # Nystrom extension: for each non-landmark point,
    # compute kernel to landmarks and interpolate coordinates
    bw_lm = adaptive_bandwidth(X_lm, min(k_adaptive, n_landmarks - 1))

    # Kernel from all points to landmarks
    dists_to_lm = cdist(X, X_lm)
    bw_all = adaptive_bandwidth(X, k_adaptive)
    bw_outer = bw_all[:, None] * bw_lm[None, :]
    K_ext = np.exp(-dists_to_lm ** 2 / bw_outer)

    # Row-normalize
    row_sums = K_ext.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-10)
    K_ext = K_ext / row_sums[:, None]

    # Nystrom extension: K_ext @ psi(lm) ≈ lambda * psi(x) (the Markov eigenequation).
    # But coords_lm = lambda * psi(lm), so K_ext @ coords_lm = lambda^2 * psi(x).
    # We need lambda * psi(x), so divide by eigenvalues once to get psi(lm) first.
    evals_safe = np.maximum(np.abs(evals), 1e-10)
    psi_lm = coords_lm / evals_safe[None, :]  # recover unweighted eigenvectors
    coords_all = K_ext @ psi_lm  # gives lambda * psi(x) = diffusion coords

    return coords_all, evals


def dmap_series(dvecs: np.ndarray, cfg: Optional[DmapConfig] = None,
                tau: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diffusion map for a single series' delay vectors.

    Automatically chooses dense or Nystrom based on point count.

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
        Delay vectors.
    cfg : DmapConfig
    tau : int
        Embedding delay (used for Theiler window if auto).

    Returns
    -------
    coords : np.ndarray, shape (n_support, n_components)
        Diffusion coordinates.
    eigenvalues : np.ndarray, shape (n_components,)
    """
    if cfg is None:
        cfg = DmapConfig()

    n = dvecs.shape[0]

    # Theiler window
    theiler = cfg.theiler_window
    if theiler == 0:
        theiler = tau  # auto: set to embedding delay

    # Subsample to support size if needed
    if n > cfg.n_support:
        dvecs_sub, _ = subsample_fps(dvecs, cfg.n_support)
    else:
        dvecs_sub = dvecs

    n_sub = dvecs_sub.shape[0]

    if n_sub <= cfg.nystrom_threshold:
        return dmap_dense(
            dvecs_sub,
            n_components=cfg.n_components,
            alpha=cfg.alpha,
            k_adaptive=min(cfg.k_adaptive, n_sub - 1),
            theiler=theiler,
        )
    else:
        return dmap_nystrom(
            dvecs_sub,
            n_landmarks=min(500, n_sub // 2),
            n_components=cfg.n_components,
            alpha=cfg.alpha,
            k_adaptive=cfg.k_adaptive,
            theiler=theiler,
        )


def intra_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute intra-series distance matrix from diffusion coordinates.

    This (n_support, n_support) matrix is what gets fed to GW alignment.

    Parameters
    ----------
    coords : np.ndarray, shape (n_support, n_components)

    Returns
    -------
    np.ndarray, shape (n_support, n_support)
    """
    return cdist(coords, coords)


def spectral_gap(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral gap: ratio of first to second eigenvalue.

    Large spectral gap → well-defined intrinsic dimensionality.
    """
    if len(eigenvalues) < 2 or eigenvalues[0] < 1e-10:
        return 0.0
    return eigenvalues[0] / eigenvalues[1]


def intrinsic_dim_from_spectrum(eigenvalues: np.ndarray,
                                 threshold: float = 0.01) -> int:
    """
    Estimate intrinsic dimensionality from eigenvalue spectrum.

    Count eigenvalues above threshold * max_eigenvalue.
    """
    if len(eigenvalues) == 0:
        return 0
    cutoff = threshold * eigenvalues[0]
    return int(np.sum(eigenvalues > cutoff))
