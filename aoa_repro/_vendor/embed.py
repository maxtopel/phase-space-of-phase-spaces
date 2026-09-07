"""
GW Attractor Landscape — Takens Delay-Coordinate Embedding.

Status: ACTIVE

Implements:
- AMI (average mutual information) for tau selection (Fraser-Swinney 1986)
- FNN (false nearest neighbors) with adaptive noise floor for dim (Kennel et al. 1992)
- Delay vector construction: raw (levels) and tau-differenced
- Global normalization (per-row z-score)
- Routing: stationary → raw delay vectors; differenced → tau-diff delay vectors

Key invariant:
  offset = dim * tau  (NOT (dim-1)*tau — accounts for tau-differencing step)

References:
- Takens (1981): Detecting strange attractors in turbulence
- Sauer, Yorke, Casdagli (1991): Embedology (fractal extension)
- Fraser & Swinney (1986): AMI for delay selection
- Kennel, Brown, Abarbanel (1992): FNN algorithm
"""

import numpy as np
from typing import Optional, Tuple

from config import EmbedConfig


def ami(series: np.ndarray, max_lag: int = 50, n_bins: int = 64) -> np.ndarray:
    """
    Average Mutual Information as a function of lag.

    Selects tau as first local minimum of AMI(lag).

    Parameters
    ----------
    series : np.ndarray
        1D time series.
    max_lag : int
        Maximum lag to compute.
    n_bins : int
        Number of histogram bins for MI estimation.

    Returns
    -------
    np.ndarray
        AMI values for lags 0..max_lag.
    """
    n = len(series)
    max_lag = min(max_lag, n // 3)

    # Bin the data
    edges = np.linspace(np.min(series), np.max(series) + 1e-10, n_bins + 1)
    binned = np.digitize(series, edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)

    ami_vals = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag >= n:
            break
        x = binned[:n - lag] if lag > 0 else binned
        y = binned[lag:] if lag > 0 else binned

        # Joint histogram via np.histogram2d-style bincount
        joint = np.zeros((n_bins, n_bins), dtype=np.float64)
        np.add.at(joint, (x, y), 1)
        joint /= joint.sum()

        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        # Vectorized: outer product of marginals
        outer = px[:, None] * py[None, :]
        mask = (joint > 0) & (outer > 0)
        mi = np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))
        ami_vals[lag] = mi

    return ami_vals


def select_tau(series: np.ndarray, cfg: Optional[EmbedConfig] = None) -> int:
    """
    Select embedding delay tau via first local minimum of AMI.

    Falls back to cfg.tau_default if no minimum found.
    """
    if cfg is None:
        cfg = EmbedConfig()

    ami_vals = ami(series, max_lag=cfg.ami_max_lag, n_bins=cfg.ami_bins)

    # Find first local minimum
    for i in range(1, len(ami_vals) - 1):
        if ami_vals[i] < ami_vals[i - 1] and ami_vals[i] <= ami_vals[i + 1]:
            tau = int(i)
            return max(cfg.tau_min, min(tau, cfg.tau_max))

    return cfg.tau_default


def fnn(series: np.ndarray, tau: int, max_dim: int = 20,
        rtol: float = 15.0, atol: float = 2.0,
        noise_floor: float = 0.01) -> np.ndarray:
    """
    False Nearest Neighbors fraction as a function of embedding dimension.

    Adaptive noise floor threshold: FNN fraction below noise_floor
    (proportional to (eps/sigma_s)^2 per Ch12 line 946) is considered zero.

    Parameters
    ----------
    series : np.ndarray
        1D time series.
    tau : int
        Embedding delay.
    max_dim : int
        Maximum dimension to test.
    rtol : float
        Relative distance tolerance for FNN criterion.
    atol : float
        Absolute distance tolerance (Kennel's second criterion).
    noise_floor : float
        FNN fraction below this = converged.

    Returns
    -------
    np.ndarray
        FNN fraction for dimensions 1..max_dim.
    """
    from scipy.spatial import KDTree

    sigma = np.std(series)
    n = len(series)
    fnn_fracs = np.ones(max_dim)

    for dim in range(1, max_dim + 1):
        # Build delay vectors at this dimension
        offset = dim * tau
        n_vecs = n - offset
        if n_vecs < 10:
            break

        dvecs = np.zeros((n_vecs, dim))
        for d in range(dim):
            dvecs[:, d] = series[d * tau:d * tau + n_vecs]

        # Build delay vectors at dim+1
        offset_next = (dim + 1) * tau
        n_vecs_next = n - offset_next
        if n_vecs_next < 10:
            break

        dvecs_next = np.zeros((n_vecs_next, dim + 1))
        for d in range(dim + 1):
            dvecs_next[:, d] = series[d * tau:d * tau + n_vecs_next]

        # Find nearest neighbors in dim-dimensional space
        tree = KDTree(dvecs[:n_vecs_next])
        dists, idx = tree.query(dvecs[:n_vecs_next], k=2)
        # k=2 because first neighbor is self
        nn_idx = idx[:, 1]
        nn_dist = dists[:, 1]

        # Count false nearest neighbors
        n_false = 0
        n_valid = 0
        for i in range(n_vecs_next):
            j = nn_idx[i]
            if j >= n_vecs_next:
                continue
            if nn_dist[i] < 1e-10:
                continue

            n_valid += 1

            # Criterion 1: relative distance increase
            dist_next = np.abs(dvecs_next[i, -1] - dvecs_next[j, -1])
            if dist_next / nn_dist[i] > rtol:
                n_false += 1
                continue

            # Criterion 2: absolute distance (Kennel's R_tol)
            dist_full = np.linalg.norm(dvecs_next[i] - dvecs_next[j])
            if dist_full / sigma > atol:
                n_false += 1

        fnn_fracs[dim - 1] = n_false / max(n_valid, 1)

    return fnn_fracs


def select_dim(series: np.ndarray, tau: int,
               cfg: Optional[EmbedConfig] = None) -> int:
    """
    Select embedding dimension via FNN with adaptive noise floor.

    Returns first dim where FNN fraction drops below noise_floor.
    """
    if cfg is None:
        cfg = EmbedConfig()

    fnn_fracs = fnn(
        series, tau,
        max_dim=cfg.fnn_max_dim,
        rtol=cfg.fnn_rtol,
        atol=cfg.fnn_atol,
        noise_floor=cfg.fnn_noise_floor,
    )

    # Find first dim where FNN < noise_floor
    for dim_idx, frac in enumerate(fnn_fracs):
        if frac < cfg.fnn_noise_floor:
            dim = dim_idx + 1
            return max(cfg.dim_min, min(dim, cfg.dim_max))

    return cfg.dim_default


def make_raw_delay_vectors(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Construct raw (level) delay vectors.

    dvec[i] = [x(i), x(i+tau), x(i+2*tau), ..., x(i+(dim-1)*tau)]
    offset = dim * tau (accounts for tau-differencing step in pipeline)

    Used for: already-stationary series, level layer for cointegration.
    """
    n = len(series)
    offset = dim * tau
    n_vecs = n - offset
    if n_vecs <= 0:
        raise ValueError(
            f"Series too short ({n}) for dim={dim}, tau={tau} (need > {offset})"
        )

    dvecs = np.zeros((n_vecs, dim))
    for d in range(dim):
        dvecs[:, d] = series[d * tau:d * tau + n_vecs]

    return dvecs


def make_delay_vectors(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Construct tau-differenced delay vectors.

    First computes tau-differences: y(t) = x(t) - x(t-tau)
    Then builds delay vectors from y.

    Used for: series that need differencing (transform route includes diff).
    Internally handles the tau-step, so the input is the RAW (pre-diff) series.
    """
    # Tau-difference
    y = series[tau:] - series[:-tau]
    return make_raw_delay_vectors(y, dim, tau)


def standardize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-row z-score normalization.

    Returns (X - mu) / clip(std, eps) where mu and std are per-row.

    This is the global normalization step. Affine transformation preserves
    embedding (smooth monotone transform → PE invariance per Ch20 Thm 20.pe-invariance).
    """
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.clip(std, eps, None)
    return (X - mu) / std


def embed_series(series: np.ndarray, transform_route: str,
                 cfg: Optional[EmbedConfig] = None) -> Tuple[np.ndarray, int, int]:
    """
    Full embedding pipeline for a single (already-transformed) series.

    1. Select tau via AMI
    2. Select dim via FNN
    3. Construct delay vectors (raw or tau-diff based on route)
    4. Global normalization

    Parameters
    ----------
    series : np.ndarray
        Transformed series (output of transform.py).
    transform_route : str
        The route string, used to determine raw vs tau-diff embedding.
    cfg : EmbedConfig

    Returns
    -------
    dvecs : np.ndarray, shape (n_vecs, dim)
        Normalized delay vectors.
    dim : int
        Selected embedding dimension.
    tau : int
        Selected embedding delay.
    """
    if cfg is None:
        cfg = EmbedConfig()

    if len(series) < 50:
        raise ValueError(f"Series too short ({len(series)}) for embedding")

    # Select parameters
    tau = select_tau(series, cfg)
    dim = select_dim(series, tau, cfg)

    # Construct delay vectors
    # If route involves differencing, the series is already differenced by transform.py,
    # so we use raw delay vectors on the transformed series
    dvecs = make_raw_delay_vectors(series, dim, tau)

    # Global normalization
    if cfg.normalize:
        dvecs = standardize_rows(dvecs)

    return dvecs, dim, tau


def _embed_one_worker(args):
    """Top-level worker function for embed_batch (must be picklable)."""
    s, r, cfg = args
    try:
        return embed_series(s, r, cfg)
    except ValueError:
        return (None, None, None)


def embed_batch(series_list: list, routes: list,
                cfg: Optional[EmbedConfig] = None,
                n_workers: int = 1) -> list:
    """
    Embed multiple series in parallel.

    Returns list of (dvecs, dim, tau) tuples.
    """
    if cfg is None:
        cfg = EmbedConfig()

    if n_workers <= 1:
        results = []
        for s, r in zip(series_list, routes):
            try:
                results.append(embed_series(s, r, cfg))
            except ValueError:
                results.append((None, None, None))
        return results

    from concurrent.futures import ProcessPoolExecutor

    work_items = [(s, r, cfg) for s, r in zip(series_list, routes)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(
            _embed_one_worker,
            work_items,
            chunksize=100,
        ))

    return results
