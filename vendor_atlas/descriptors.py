"""
GW Attractor Landscape — 16D Attractor Descriptor Computation.

Status: ACTIVE

Revised 16D descriptor set (R3-validated):

| # | Name       | Invariance         |
|---|------------|--------------------|
| 1 | λ_max      | Smooth conjugacy   |
| 2 | D2         | Smooth conjugacy   |
| 3 | PE3        | Monotone transform |
| 4 | PE4        | Monotone transform |
| 5 | PE5        | Monotone transform |
| 6 | H_spec     | Moderate           |
| 7 | RR         | Geometric          |
| 8 | DET        | Geometric          |
| 9 | MI_lag1    | Moderate           |
|10 | Cao_E1     | Embedding-dep      |
|11 | FNN_floor  | Embedding-dep      |
|12 | anisotropy | Embedding-dep      |
|13 | curvature  | Moderate           |
|14 | PH_total   | Homeomorphism      |
|15 | PH_max     | Homeomorphism      |
|16 | β1_sig     | Homeomorphism      |

Cross-domain invariant subset (5D):
  PE3, PE4, PE5, PH_total, PH_max (indices 2, 3, 4, 13, 14)

References:
- Ch6, Ch8, Ch12, Ch14, Ch20 of textbook
- Rosenstein et al. (1993): Lyapunov exponent
- Grassberger & Procaccia (1983): correlation dimension
- Bandt & Pompe (2002): permutation entropy
- Marwan et al. (2007): recurrence quantification analysis
- Kraskov et al. (2004): KSG mutual information estimator
- Ripser (Bauer 2021): persistent homology
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import stats
from itertools import permutations
from math import factorial

from config import DescriptorConfig


# ============================================================
# Permutation entropy (PE3, PE4, PE5)
# ============================================================

def permutation_entropy(series: np.ndarray, order: int = 3,
                        delay: int = 1) -> float:
    """
    Permutation entropy (Bandt & Pompe 2002).

    Invariant under monotone transformations (Ch20 Thm 20.pe-invariance).

    Parameters
    ----------
    series : np.ndarray
        1D time series.
    order : int
        Permutation order (3, 4, or 5).
    delay : int
        Embedding delay for ordinal patterns.

    Returns
    -------
    float
        Normalized permutation entropy in [0, 1].
    """
    n = len(series)
    n_patterns = n - (order - 1) * delay
    if n_patterns <= 0:
        return np.nan

    # Count ordinal patterns
    n_perms = factorial(order)
    counts = np.zeros(n_perms, dtype=np.int64)

    # Map each ordinal pattern to an index via Lehmer code
    for i in range(n_patterns):
        pattern = series[i:i + order * delay:delay]
        # Rank pattern
        ranks = stats.rankdata(pattern, method='ordinal') - 1
        # Lehmer code → index
        idx = _lehmer_index(ranks, order)
        counts[idx] += 1

    # Normalize
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    # Shannon entropy, normalized by log(order!)
    h = -np.sum(probs * np.log(probs))
    h_max = np.log(n_perms)

    return h / h_max if h_max > 0 else 0.0


def _lehmer_index(ranks: np.ndarray, order: int) -> int:
    """Convert rank permutation to Lehmer code index."""
    idx = 0
    for i in range(order - 1):
        # Count elements after position i that are smaller
        smaller = np.sum(ranks[i + 1:] < ranks[i])
        idx = idx * (order - i) + smaller
    return idx


# ============================================================
# Lyapunov exponent (Rosenstein method)
# ============================================================

def lyapunov_rosenstein(dvecs: np.ndarray, dt: float = 1.0,
                        min_tsep: int = 0, max_iter: int = 500,
                        max_points: int = 2000) -> float:
    """
    Maximal Lyapunov exponent via Rosenstein et al. (1993).

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
        Delay vectors.
    dt : float
        Time step.
    min_tsep : int
        Minimum temporal separation for neighbor search.
    max_iter : int
        Maximum iterations for divergence tracking.
    max_points : int
        Subsample if n > max_points (cdist is O(n^2) memory).

    Returns
    -------
    float
        Estimated maximal Lyapunov exponent.
    """
    n = dvecs.shape[0]

    # Cap input size to avoid O(n^2) memory blowup in cdist
    if n > max_points:
        start = np.random.randint(0, n - max_points)
        dvecs = dvecs[start:start + max_points]
        n = max_points

    max_iter = min(max_iter, n // 2)

    if n < 20:
        return np.nan

    # Find nearest neighbor for each point (with temporal separation)
    dists = cdist(dvecs, dvecs)
    np.fill_diagonal(dists, np.inf)

    # Apply temporal separation
    for offset in range(1, min_tsep + 1):
        idx = np.arange(n - offset)
        dists[idx, idx + offset] = np.inf
        dists[idx + offset, idx] = np.inf

    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(n), nn_idx]

    # Track divergence
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n):
        j = nn_idx[i]
        d0 = nn_dist[i]
        if d0 < 1e-10:
            continue

        for k in range(max_iter):
            i_k = i + k
            j_k = j + k
            if i_k >= n or j_k >= n:
                break

            d_k = np.linalg.norm(dvecs[i_k] - dvecs[j_k])
            if d_k > 0:
                divergence[k] += np.log(d_k)
                counts[k] += 1

    # Average log-divergence
    valid = counts > 0
    if not np.any(valid):
        return np.nan

    avg_divergence = np.where(valid, divergence / counts, np.nan)

    # Linear fit to initial growth region
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 5:
        return np.nan

    # Use first 20% or at least 5 points
    n_fit = max(5, len(valid_idx) // 5)
    fit_idx = valid_idx[:n_fit]
    t = fit_idx * dt
    y = avg_divergence[fit_idx]

    # Slope = Lyapunov exponent (with R² quality gate)
    slope, _, r_value, _, _ = stats.linregress(t, y)
    if r_value ** 2 < 0.9:
        return np.nan  # no clean linear growth region
    return slope


# ============================================================
# Correlation dimension (Grassberger-Procaccia)
# ============================================================

def correlation_dimension(dvecs: np.ndarray, n_radii: int = None,
                          max_points: int = 2000,
                          small_n_threshold: int = 300) -> float:
    """
    Correlation dimension D2 via Grassberger-Procaccia algorithm.

    Gated by convergence check on scaling-region linearity. Per R14 review:
    R² gate is relaxed for small samples (n < 300) where 0.95 is too strict
    given the discrete-binning noise floor. n_radii is now adaptive.

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
    n_radii : int, optional
        Number of log-spaced radii. If None, scaled adaptively to sample size.
    max_points : int
        Subsample if n > max_points.
    small_n_threshold : int
        Below this, use the relaxed R² > 0.90 gate.

    Returns
    -------
    float
        Estimated D2, or NaN if no scaling region found.
    """
    n = dvecs.shape[0]

    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        dvecs = dvecs[idx]
        n = max_points

    # Adaptive n_radii (R14 fix): scales with sample size to balance binning noise
    if n_radii is None:
        n_radii = max(15, min(30, n // 10))

    # Pairwise distances
    dists = pdist(dvecs)

    # Log-spaced radii — use 5th percentile floor (skip near-zero distances)
    nonzero = dists[dists > 1e-10]
    if len(nonzero) < 5:
        return np.nan
    d_min = np.percentile(nonzero, 5)
    d_max = np.percentile(dists, 99)
    if d_min >= d_max:
        return np.nan

    radii = np.logspace(np.log10(d_min), np.log10(d_max), n_radii)

    # Correlation sum
    n_pairs = len(dists)
    C = np.array([np.sum(dists < r) / n_pairs for r in radii])

    # Log-log slope in scaling region
    valid = C > 0
    if np.sum(valid) < 5:
        return np.nan

    log_r = np.log(radii[valid])
    log_C = np.log(C[valid])

    # Find best linear region via sliding window
    best_r2 = 0.0
    best_slope = np.nan
    window = max(5, len(log_r) // 3)

    for i in range(len(log_r) - window):
        x = log_r[i:i + window]
        y = log_C[i:i + window]
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        if r_value ** 2 > best_r2:
            best_r2 = r_value ** 2
            best_slope = slope

    # Two-stage gate (R14 + Option B fix):
    #   1. R² > 0.92 (relaxed from 0.95 for small samples but tighter than 0.90
    #      which let through degenerate flat-line fits at slope ~0.2)
    #   2. Slope ≥ 1.0 — physically meaningful intrinsic dim
    #      (a slope of 0.2 would mean d_eff ≈ 0.2 which is geometrically
    #      meaningless on a 10-D Hellinger DMAP space)
    r2_gate = 0.92 if n < small_n_threshold else 0.95
    if best_r2 < r2_gate:
        return np.nan
    if best_slope < 1.0 or best_slope > 20.0:
        # Slope outside the plausible range for AoA manifolds (4-D macro
        # economy ± noise). Reject as a degenerate fit.
        return np.nan

    return best_slope


# ============================================================
# Spectral entropy
# ============================================================

def spectral_entropy(series: np.ndarray, nperseg: int = 256) -> float:
    """
    Spectral entropy: Shannon entropy of normalized power spectrum.

    Parameters
    ----------
    series : np.ndarray
    nperseg : int
        Segment length for Welch method.

    Returns
    -------
    float
        Normalized spectral entropy in [0, 1].
    """
    from scipy.signal import welch

    nperseg = min(nperseg, len(series))
    if nperseg < 4:
        return np.nan

    freqs, psd = welch(series, nperseg=nperseg)
    psd = psd[psd > 0]
    if len(psd) == 0:
        return np.nan

    psd_norm = psd / psd.sum()
    h = -np.sum(psd_norm * np.log(psd_norm))
    h_max = np.log(len(psd_norm))

    return h / h_max if h_max > 0 else 0.0


# ============================================================
# RQA: Recurrence Rate and Determinism
# ============================================================

def rqa_descriptors(dvecs: np.ndarray, threshold_pct: float = 10.0,
                    min_diag: int = 2, theiler: int = 1) -> Tuple[float, float]:
    """
    Recurrence Quantification Analysis: recurrence rate (RR) and determinism (DET).

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
    threshold_pct : float
        Recurrence threshold as percentile of max distance.
    min_diag : int
        Minimum diagonal line length for DET.
    theiler : int
        Theiler corridor exclusion.

    Returns
    -------
    rr : float
        Recurrence rate.
    det : float
        Determinism (fraction of recurrence points forming diagonal lines).
    """
    n = dvecs.shape[0]
    max_n = 500  # cap for memory
    if n > max_n:
        # Use contiguous block (not random) to preserve Theiler corridor semantics
        start = np.random.randint(0, n - max_n)
        dvecs = dvecs[start:start + max_n]
        n = max_n

    dists = cdist(dvecs, dvecs)
    threshold = np.percentile(dists, threshold_pct)

    # Recurrence matrix
    R = (dists <= threshold).astype(np.int8)

    # Exclude Theiler corridor
    for offset in range(theiler + 1):
        np.fill_diagonal(R[offset:, :n - offset], 0)
        if offset > 0:
            np.fill_diagonal(R[:n - offset, offset:], 0)

    # Recurrence rate
    n_recur = R.sum()
    n_possible = n * (n - 1) - 2 * theiler * (n - theiler)
    rr = n_recur / max(n_possible, 1)

    # Determinism: count diagonal lines of length >= min_diag
    diag_points = 0
    total_recur = 0

    for offset in range(theiler + 1, n):
        diag = np.diag(R, offset)
        total_recur += diag.sum()

        # Count runs of 1s
        if len(diag) == 0:
            continue

        in_run = False
        run_len = 0
        for val in diag:
            if val:
                run_len += 1
                in_run = True
            else:
                if in_run and run_len >= min_diag:
                    diag_points += run_len
                run_len = 0
                in_run = False
        if in_run and run_len >= min_diag:
            diag_points += run_len

    # Count both diagonals
    det = diag_points * 2 / max(n_recur, 1)
    det = min(det, 1.0)

    return rr, det


# ============================================================
# KSG Mutual Information at lag 1
# ============================================================

def mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KSG mutual information estimator (Kraskov et al. 2004, algorithm 1).

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
    k : int
        k-nearest neighbors.

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    from scipy.special import digamma

    n = len(x)
    if n < k + 1:
        return np.nan

    # Joint space: (x, y)
    xy = np.column_stack([x, y])

    # For each point, find distance to k-th neighbor in joint space (Chebyshev)
    from scipy.spatial import KDTree

    tree_xy = KDTree(xy)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    tree_x = KDTree(x.reshape(-1, 1))
    tree_y = KDTree(y.reshape(-1, 1))

    nx = np.array([
        len(tree_x.query_ball_point([xi], r=ei + 1e-15, p=np.inf)) - 1
        for xi, ei in zip(x, eps)
    ])
    ny = np.array([
        len(tree_y.query_ball_point([yi], r=ei + 1e-15, p=np.inf)) - 1
        for yi, ei in zip(y, eps)
    ])

    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return max(mi, 0.0)


# ============================================================
# Cao E1 (embedding quality)
# ============================================================

def cao_e1(series: np.ndarray, tau: int, dim: int) -> float:
    """
    Cao's E1 statistic for embedding quality.

    E1(d) = a(d+1) / a(d), where a(d) is the mean ratio of
    NN distances in d+1 vs d dimensions.

    E1 → 1 when d is sufficient.

    Parameters
    ----------
    series : np.ndarray
    tau : int
    dim : int
        Current embedding dimension.

    Returns
    -------
    float
        E1 value.
    """
    from scipy.spatial import KDTree

    n = len(series)

    # Build d-dimensional vectors
    offset_d = dim * tau
    n_d = n - offset_d
    if n_d < 10:
        return np.nan

    dvecs_d = np.zeros((n_d, dim))
    for d in range(dim):
        dvecs_d[:, d] = series[d * tau:d * tau + n_d]

    # Build (d+1)-dimensional vectors
    offset_d1 = (dim + 1) * tau
    n_d1 = n - offset_d1
    if n_d1 < 10:
        return np.nan

    dvecs_d1 = np.zeros((n_d1, dim + 1))
    for d in range(dim + 1):
        dvecs_d1[:, d] = series[d * tau:d * tau + n_d1]

    # NN in d dimensions
    tree = KDTree(dvecs_d[:n_d1])
    _, idx = tree.query(dvecs_d[:n_d1], k=2)
    nn_idx = idx[:, 1]

    # a(d) = mean |x_d+1(i,nn) - x_d+1(nn,nn)| / ||x_d(i) - x_d(nn)||
    ratios = []
    for i in range(n_d1):
        j = nn_idx[i]
        dist_d = np.linalg.norm(dvecs_d[i] - dvecs_d[j])
        if dist_d < 1e-10:
            continue
        dist_d1 = np.linalg.norm(dvecs_d1[i] - dvecs_d1[j])
        ratios.append(dist_d1 / dist_d)

    if len(ratios) < 5:
        return np.nan

    return np.mean(ratios)


# ============================================================
# Anisotropy (eig2/eig1 of PCA)
# ============================================================

def anisotropy(dvecs: np.ndarray) -> float:
    """
    Shape anisotropy: ratio of second to first PCA eigenvalue.

    Values near 1 = isotropic, near 0 = highly anisotropic (elongated).
    """
    if dvecs.shape[0] < 3 or dvecs.shape[1] < 2:
        return np.nan

    centered = dvecs - dvecs.mean(axis=0)
    cov = np.cov(centered.T)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]

    if evals[0] < 1e-10:
        return np.nan

    return evals[1] / evals[0]


# ============================================================
# Curvature (geodesic/Euclidean diameter ratio)
# ============================================================

def curvature_ratio(dvecs: np.ndarray, k: int = 8) -> float:
    """
    Geodesic/Euclidean diameter ratio via 2-Dijkstra on kNN graph.

    High ratio (>> 1) = curved/nonlinear manifold.
    Ratio ≈ 1 = flat/linear.

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
    k : int
        Number of neighbors for kNN graph.

    Returns
    -------
    float
        Geodesic diameter / Euclidean diameter.
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import lil_matrix

    n = dvecs.shape[0]
    max_n = 300
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        dvecs = dvecs[idx]
        n = max_n

    k = min(k, n - 1)

    # Build kNN graph
    dists = cdist(dvecs, dvecs)
    graph = lil_matrix((n, n))

    for i in range(n):
        nn = np.argsort(dists[i])[1:k + 1]
        for j in nn:
            graph[i, j] = dists[i, j]
            graph[j, i] = dists[i, j]

    # Shortest paths (Dijkstra)
    geo_dists = shortest_path(graph.tocsr(), directed=False)
    geo_dists[~np.isfinite(geo_dists)] = 0

    # Geodesic diameter
    geo_diameter = np.max(geo_dists)

    # Euclidean diameter
    euc_diameter = np.max(dists)

    if euc_diameter < 1e-10:
        return np.nan

    return geo_diameter / euc_diameter


# ============================================================
# Persistent Homology (PH_total, PH_max, β1_sig)
# ============================================================

def persistent_homology(dvecs: np.ndarray, max_points: int = 150,
                        max_dim: int = 1) -> Tuple[float, float]:
    """
    Persistent homology: total persistence and max persistence.

    Parameters
    ----------
    dvecs : np.ndarray, shape (n, dim)
    max_points : int
        Subsample for ripser.
    max_dim : int
        Maximum homology dimension (1 = H0 + H1).

    Returns
    -------
    ph_total : float
        Sum of all persistence lifetimes (H0 + H1).
    ph_max : float
        Longest-lived topological feature.
    """
    try:
        from ripser import ripser
    except ImportError:
        return np.nan, np.nan

    n = dvecs.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        dvecs = dvecs[idx]

    result = ripser(dvecs, maxdim=max_dim)
    diagrams = result['dgms']

    total_persistence = 0.0
    max_persistence = 0.0

    for dim_dgm in diagrams:
        finite_mask = np.isfinite(dim_dgm[:, 1])
        lifetimes = dim_dgm[finite_mask, 1] - dim_dgm[finite_mask, 0]
        if len(lifetimes) > 0:
            total_persistence += np.sum(lifetimes)
            max_persistence = max(max_persistence, np.max(lifetimes))

    return total_persistence, max_persistence


def beta1_significance(dvecs: np.ndarray, n_surrogates: int = 39,
                       max_points: int = 150) -> float:
    """
    Significance of 1-cycles via FT (phase-randomized) surrogates.

    Tests whether the original data has more significant H1 features
    than expected from its power spectrum alone.

    Returns z-score: (original_PH_H1 - mean_surrogate) / std_surrogate.
    """
    try:
        from ripser import ripser
    except ImportError:
        return np.nan

    n, dim = dvecs.shape
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        dvecs_sub = dvecs[idx]
    else:
        dvecs_sub = dvecs

    # Original H1 total persistence
    result = ripser(dvecs_sub, maxdim=1)
    h1 = result['dgms'][1]
    finite = np.isfinite(h1[:, 1])
    orig_h1 = np.sum(h1[finite, 1] - h1[finite, 0]) if np.any(finite) else 0.0

    # Surrogates: phase-randomize each dimension independently
    surrogate_h1 = np.zeros(n_surrogates)
    for s in range(n_surrogates):
        surr = _phase_randomize_matrix(dvecs_sub)
        result_s = ripser(surr, maxdim=1)
        h1_s = result_s['dgms'][1]
        finite_s = np.isfinite(h1_s[:, 1])
        surrogate_h1[s] = np.sum(h1_s[finite_s, 1] - h1_s[finite_s, 0]) if np.any(finite_s) else 0.0

    mu = np.mean(surrogate_h1)
    sigma = np.std(surrogate_h1)

    if sigma < 1e-10:
        return 0.0

    return (orig_h1 - mu) / sigma


def _phase_randomize_matrix(X: np.ndarray) -> np.ndarray:
    """Phase-randomize each column of X independently (FT surrogates)."""
    n, d = X.shape
    result = np.zeros_like(X)
    for col in range(d):
        result[:, col] = _phase_randomize(X[:, col])
    return result


def _phase_randomize(series: np.ndarray) -> np.ndarray:
    """
    Phase-randomize a 1D series (FT surrogate).

    Preserves power spectrum, destroys nonlinear structure.
    """
    n = len(series)
    fft = np.fft.rfft(series)
    phases = np.random.uniform(0, 2 * np.pi, len(fft))
    phases[0] = 0  # preserve DC
    if n % 2 == 0:
        phases[-1] = 0  # preserve Nyquist
    fft_surr = np.abs(fft) * np.exp(1j * phases)
    return np.fft.irfft(fft_surr, n=n)


# ============================================================
# Full 16D descriptor vector
# ============================================================

DESCRIPTOR_NAMES = [
    'lambda_max', 'D2', 'PE3', 'PE4', 'PE5',
    'H_spec', 'RR', 'DET', 'MI_lag1', 'Cao_E1',
    'FNN_floor', 'anisotropy', 'curvature',
    'PH_total', 'PH_max', 'beta1_sig',
]

INVARIANT_INDICES = [2, 3, 4, 13, 14]  # PE3, PE4, PE5, PH_total, PH_max
# NOTE: DET removed from invariant subset (R1 review W4) — DET depends on
# threshold percentile which is observation-function-sensitive, so it is NOT
# safe for cross-domain comparison. Replaced with PE4 (monotone-invariant).


def compute_descriptors(series: np.ndarray, dvecs: np.ndarray,
                        dim: int, tau: int,
                        cfg: Optional[DescriptorConfig] = None) -> np.ndarray:
    """
    Compute full 16D descriptor vector for one series.

    Parameters
    ----------
    series : np.ndarray
        Original (pre-embedding) time series.
    dvecs : np.ndarray, shape (n_vecs, dim)
        Delay vectors.
    dim : int
        Embedding dimension used.
    tau : int
        Embedding delay used.
    cfg : DescriptorConfig

    Returns
    -------
    np.ndarray, shape (16,)
        Descriptor vector. NaN for any descriptor that failed to compute.
    """
    if cfg is None:
        cfg = DescriptorConfig()

    desc = np.full(16, np.nan)

    # 1. λ_max (Rosenstein)
    min_tsep = cfg.lyap_min_tsep if cfg.lyap_min_tsep >= 0 else tau  # auto = tau
    desc[0] = lyapunov_rosenstein(dvecs, min_tsep=min_tsep,
                                   max_iter=cfg.lyap_max_iter)

    # 2. D2 (Grassberger-Procaccia)
    desc[1] = correlation_dimension(dvecs, n_radii=cfg.d2_n_radii,
                                     max_points=cfg.d2_max_points)

    # 3-5. PE3, PE4, PE5
    for i, order in enumerate(cfg.pe_orders):
        desc[2 + i] = permutation_entropy(series, order=order, delay=cfg.pe_delay)

    # 6. Spectral entropy
    desc[5] = spectral_entropy(series, nperseg=cfg.hspec_nperseg)

    # 7-8. RQA (RR, DET)
    rr, det = rqa_descriptors(dvecs, threshold_pct=cfg.rqa_threshold_percentile,
                               min_diag=cfg.rqa_min_diag, theiler=cfg.rqa_theiler)
    desc[6] = rr
    desc[7] = det

    # 9. MI_lag1
    if len(series) > 10:
        desc[8] = mi_ksg(series[:-1], series[1:], k=cfg.mi_k)

    # 10. Cao E1
    desc[9] = cao_e1(series, tau, dim)

    # 11. FNN floor
    from embed import fnn
    fnn_fracs = fnn(series, tau, max_dim=dim + 2)
    if dim - 1 < len(fnn_fracs):
        desc[10] = fnn_fracs[dim - 1]

    # 12. Anisotropy
    desc[11] = anisotropy(dvecs)

    # 13. Curvature
    desc[12] = curvature_ratio(dvecs, k=cfg.curvature_k_neighbors)

    # 14-15. PH_total, PH_max
    ph_total, ph_max = persistent_homology(dvecs, max_points=cfg.ph_max_points,
                                            max_dim=cfg.ph_max_dim)
    desc[13] = ph_total
    desc[14] = ph_max

    # 16. β1_sig
    desc[15] = beta1_significance(dvecs, n_surrogates=cfg.beta1_n_surrogates,
                                   max_points=cfg.ph_max_points)

    return desc


def invariant_subset(desc: np.ndarray) -> np.ndarray:
    """
    Extract 5D cross-domain invariant subset: PE3, PE4, PE5, PH_total, PH_max.

    All monotone-transform or homeomorphism invariant (safe for cross-domain comparison).
    DET was removed: it depends on threshold percentile, which is observation-function-sensitive.
    """
    return desc[INVARIANT_INDICES]
