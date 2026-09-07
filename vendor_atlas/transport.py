"""
GW Attractor Landscape — Transport-Plan DMAP Coordinates.

Status: ACTIVE

The AoA coordinate system: each series' attractor is represented by its
GW transport plan to a shared barycenter, then DMAP discovers the
low-dimensional manifold structure of these transport plans.

Architecture:
  Series → per-series DMAP → 20-point skeleton → exact GW transport to barycenter
  → T_i ∈ Birkhoff polytope → Hellinger DMAP on transport plans → AoA coordinates

Why not centroids: For any doubly-stochastic T with uniform marginals and
centered MDS coordinates, centroid = q @ bary = mean(bary) = 0. The centroid
summary discards all information. The transport plan itself IS the coordinate.

Why DMAP (not PCA): The Birkhoff polytope is nonlinear, and attractor
combinations produce nonlinear variation in transport-plan space. PCA
captures only linear modes. DMAP discovers the intrinsic geometry.

Why Hellinger (not Frobenius): Hellinger distance d_H(T_i, T_j) =
||sqrt(T_i) - sqrt(T_j)||_F amplifies differences near the uniform
coupling by ~10x (for n=20). The square-root embedding maps int(B_n)
onto a smooth submanifold of the sphere, where DMAP convergence to
Laplace-Beltrami is guaranteed (Coifman-Lafon 2006).

References:
- Mémoli (2011): GW distance as metric on mm-spaces
- Coifman & Lafon (2006): DMAP convergence on smooth manifolds
- Cuturi (2013): Sinkhorn distances (entropic regularization)
- R3/R1 reviewer analysis: centroid collapse identity, Hellinger amplification
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.distance import cdist

from config import PipelineConfig


def compute_transport_plan(
    series_dist_matrix: np.ndarray,
    barycenter_dist_matrix: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Compute GW transport plan from one series to the barycenter.

    Uses entropic GW with small epsilon to keep T in the interior
    of the Birkhoff polytope (required for Hellinger distance).

    Parameters
    ----------
    series_dist_matrix : np.ndarray, shape (n, n)
    barycenter_dist_matrix : np.ndarray, shape (n, n)
    epsilon : float
        Entropic regularization. Small (0.01-0.05) to keep T
        in int(B_n) while preserving structure.

    Returns
    -------
    np.ndarray, shape (n*n,)
        Flattened transport plan.
    """
    import ot

    n = series_dist_matrix.shape[0]
    p = np.ones(n, dtype=np.float64) / n
    q = np.ones(n, dtype=np.float64) / n

    # Normalize to [0, 1]
    s_max = series_dist_matrix.max()
    b_max = barycenter_dist_matrix.max()
    C1 = series_dist_matrix / s_max if s_max > 1e-10 else series_dist_matrix
    C2 = barycenter_dist_matrix / b_max if b_max > 1e-10 else barycenter_dist_matrix

    try:
        T = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q,
            loss_fun='square_loss',
            epsilon=epsilon,
            max_iter=200,
        )
    except Exception:
        # Secondary attempt: larger epsilon for stability
        try:
            T = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q,
                loss_fun='square_loss',
                epsilon=max(epsilon * 10, 0.1),
                max_iter=100,
            )
        except Exception:
            # Final fallback: uniform coupling (marks series as degenerate)
            import warnings
            warnings.warn("GW transport failed — returning uniform coupling", stacklevel=2)
            T = np.outer(p, q)

    # Ensure strict positivity for Hellinger (floor at machine epsilon)
    T = np.maximum(T, 1e-15)
    # Re-normalize total mass (marginals are approximately preserved
    # since the floor at 1e-15 is negligible relative to 1/n^2)
    T = T / T.sum()

    return T.flatten().astype(np.float32)


def _transport_worker(args):
    """Top-level worker for parallel transport plan computation."""
    series_dm, bary_dm, epsilon = args
    return compute_transport_plan(series_dm, bary_dm, epsilon)


def compute_transport_batch(
    dist_matrices: List[np.ndarray],
    barycenter_dist_matrix: np.ndarray,
    epsilon: float = 0.01,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Compute transport plans for multiple series.

    Returns
    -------
    np.ndarray, shape (n_series, n_support^2), dtype float32
    """
    if n_workers <= 1:
        plans = [
            compute_transport_plan(D, barycenter_dist_matrix, epsilon)
            for D in dist_matrices
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor
        work = [(D, barycenter_dist_matrix, epsilon) for D in dist_matrices]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            plans = list(pool.map(_transport_worker, work, chunksize=50))

    return np.array(plans)


# ============================================================
# Hellinger distance on transport plans
# ============================================================

def hellinger_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """
    Hellinger distance between two flattened transport plans.

    d_H(T1, T2) = ||sqrt(T1) - sqrt(T2)||_F

    Amplifies differences near the uniform coupling by ~n/2,
    which is critical for detecting structure in near-uniform plans.
    """
    return np.linalg.norm(np.sqrt(t1) - np.sqrt(t2))


def hellinger_distance_matrix(plans: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Hellinger distances between transport plans.

    Parameters
    ----------
    plans : np.ndarray, shape (n, d) where d = n_support^2

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    # Square-root embedding
    sqrt_plans = np.sqrt(plans)
    # Pairwise Euclidean in sqrt space = Hellinger
    return cdist(sqrt_plans, sqrt_plans)


# ============================================================
# DMAP on transport plans (Nystrom for scalability)
# ============================================================

def transport_dmap(
    plans: np.ndarray,
    n_components: int = 20,
    n_landmarks: int = 800,
    alpha: float = 1.0,
    k_bandwidth: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diffusion map on transport-plan space via Nystrom extension.

    Each transport plan is a point on the Birkhoff polytope.
    Hellinger distance in the square-root embedding gives DMAP
    convergence to Laplace-Beltrami on the polytope interior.

    Parameters
    ----------
    plans : np.ndarray, shape (n_series, n_support^2), dtype float32
        Flattened transport plans.
    n_components : int
        Number of diffusion coordinates to return.
    n_landmarks : int
        FPS landmarks for Nystrom.
    alpha : float
        Density normalization (1.0 = Laplace-Beltrami).
    k_bandwidth : int
        k-th NN for adaptive bandwidth.

    Returns
    -------
    coords : np.ndarray, shape (n_series, n_components)
        AoA diffusion coordinates.
    eigenvalues : np.ndarray, shape (n_components,)
    """
    n = plans.shape[0]
    n_landmarks = min(n_landmarks, n)
    n_components = min(n_components, n_landmarks - 1)

    # Square-root embedding for Hellinger
    sqrt_plans = np.sqrt(plans.astype(np.float64))

    if n <= n_landmarks and n <= 5000:
        # Dense DMAP (small corpus — n×n distance matrix fits in memory)
        return _transport_dmap_dense(sqrt_plans, n_components, alpha, k_bandwidth)

    # Nystrom: select landmarks via FPS in Hellinger space
    lm_idx = _fps_hellinger(sqrt_plans, n_landmarks)

    sqrt_lm = sqrt_plans[lm_idx]

    # Dense DMAP on landmarks
    coords_lm, evals_lm = _transport_dmap_dense(
        sqrt_lm, n_components, alpha, k_bandwidth
    )

    # Nystrom extension to all points
    coords_all = _nystrom_extend(
        sqrt_plans, sqrt_lm, lm_idx, coords_lm, evals_lm,
        alpha, k_bandwidth
    )

    return coords_all, evals_lm


def _transport_dmap_dense(
    sqrt_plans: np.ndarray,
    n_components: int,
    alpha: float,
    k_bandwidth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense DMAP on sqrt-embedded transport plans.

    Standard Coifman-Lafon construction:
    1. Pairwise Euclidean distances in sqrt space (= Hellinger)
    2. Adaptive-bandwidth Gaussian kernel
    3. Alpha-normalize (Laplace-Beltrami for alpha=1.0)
    4. Row-normalize to Markov matrix
    5. Symmetric eigendecomposition
    6. Eigenvalue-weighted diffusion coordinates
    """
    n = sqrt_plans.shape[0]
    n_components = min(n_components, n - 1)
    k_bw = min(k_bandwidth, n - 1)

    # Pairwise distances (Euclidean in sqrt space = Hellinger)
    dists = cdist(sqrt_plans, sqrt_plans)

    # Adaptive bandwidth: k-th NN distance per point
    sorted_dists = np.sort(dists, axis=1)
    bandwidth = sorted_dists[:, k_bw]
    bandwidth = np.maximum(bandwidth, 1e-10)

    # Gaussian kernel with adaptive bandwidth
    bw_outer = bandwidth[:, None] * bandwidth[None, :]
    K = np.exp(-dists ** 2 / bw_outer)

    # Alpha-normalization
    if alpha > 0:
        d_alpha = K.sum(axis=1) ** alpha
        d_alpha = np.maximum(d_alpha, 1e-10)
        K = K / (d_alpha[:, None] * d_alpha[None, :])

    # Row-normalize to Markov matrix
    row_sums = K.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-10)
    P = K / row_sums[:, None]

    # Symmetric similarity transform for numerical stability
    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / np.maximum(d_sqrt, 1e-10)
    S = P * d_sqrt[:, None] * d_sqrt_inv[None, :]
    S = 0.5 * (S + S.T)

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform back to right eigenvectors of P
    right_evecs = eigenvectors * d_sqrt_inv[:, None]

    # Diffusion coordinates (skip trivial eigenvalue)
    coords = right_evecs[:, 1:n_components + 1] * eigenvalues[1:n_components + 1][None, :]
    evals = eigenvalues[1:n_components + 1]

    return coords, evals


def _fps_hellinger(sqrt_plans: np.ndarray, n_landmarks: int) -> np.ndarray:
    """FPS in Hellinger (= Euclidean on sqrt) space."""
    n = sqrt_plans.shape[0]
    indices = np.zeros(n_landmarks, dtype=int)
    indices[0] = 0

    # Distance from first point to all
    min_dists = np.linalg.norm(sqrt_plans - sqrt_plans[0], axis=1)

    for i in range(1, n_landmarks):
        indices[i] = np.argmax(min_dists)
        new_dists = np.linalg.norm(sqrt_plans - sqrt_plans[indices[i]], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return indices


def _nystrom_extend(
    sqrt_plans_all: np.ndarray,
    sqrt_plans_lm: np.ndarray,
    lm_idx: np.ndarray,
    coords_lm: np.ndarray,
    evals_lm: np.ndarray,
    alpha: float,
    k_bandwidth: int,
) -> np.ndarray:
    """
    Nystrom extension of transport-plan DMAP to all points.

    Computes kernel from each point to landmarks, then interpolates
    the landmark eigenvectors.
    """
    n = sqrt_plans_all.shape[0]
    n_lm = sqrt_plans_lm.shape[0]

    # Distances from all points to landmarks
    dists_to_lm = cdist(sqrt_plans_all, sqrt_plans_lm)

    # Bandwidth: use landmark bandwidth (k-th NN among landmarks)
    lm_dists = cdist(sqrt_plans_lm, sqrt_plans_lm)
    k_bw = min(k_bandwidth, n_lm - 1)
    bw_lm = np.sort(lm_dists, axis=1)[:, k_bw]
    bw_lm = np.maximum(bw_lm, 1e-10)

    # Per-point bandwidth from distance to landmarks
    bw_all = np.sort(dists_to_lm, axis=1)[:, min(k_bw, n_lm - 1)]
    bw_all = np.maximum(bw_all, 1e-10)

    # Kernel
    bw_outer = bw_all[:, None] * bw_lm[None, :]
    K_ext = np.exp(-dists_to_lm ** 2 / bw_outer)

    # Alpha-normalize using landmark density
    if alpha > 0:
        d_alpha_lm = np.sum(np.exp(-lm_dists ** 2 /
                    (bw_lm[:, None] * bw_lm[None, :])), axis=1) ** alpha
        d_alpha_all = K_ext.sum(axis=1) ** alpha
        d_alpha_lm = np.maximum(d_alpha_lm, 1e-10)
        d_alpha_all = np.maximum(d_alpha_all, 1e-10)
        K_ext = K_ext / (d_alpha_all[:, None] * d_alpha_lm[None, :])

    # Row-normalize
    row_sums = K_ext.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-10)
    K_ext = K_ext / row_sums[:, None]

    # Nystrom extension:
    # The Markov eigenequation: P @ psi = lambda * psi
    # For new point x: K_ext[x,:] @ psi(lm) ≈ lambda * psi(x)
    # coords_lm = lambda * psi(lm), so psi_lm = coords_lm / lambda
    # K_ext @ psi_lm ≈ lambda * psi(x) = psi(x) (unweighted)
    # Diffusion coordinates are lambda * psi(x), so multiply back:
    evals_safe = np.maximum(np.abs(evals_lm), 1e-10)
    psi_lm = coords_lm / evals_safe[None, :]
    psi_all = K_ext @ psi_lm  # ≈ psi(x) (unweighted eigenvectors)
    coords_all = psi_all * evals_safe[None, :]  # lambda * psi(x) = diffusion coords

    return coords_all


# ============================================================
# Streaming pipeline: DMAP → GW transport → discard dist matrix
# ============================================================

def streaming_transport_plans(
    dvecs_list: List[np.ndarray],
    taus: List[int],
    barycenter_dist_matrix: np.ndarray,
    n_support: int = 20,
    n_dmap_components: int = 10,
    gw_epsilon: float = 0.01,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Streaming computation: per-series DMAP → distance matrix → GW transport → discard.

    Avoids holding all distance matrices in memory. Only stores the
    flattened transport plans (n_series × n_support², float32).

    Parameters
    ----------
    dvecs_list : list of np.ndarray
        Delay vectors per series.
    taus : list of int
        Embedding delays per series (for Theiler window).
    barycenter_dist_matrix : np.ndarray, shape (n_support, n_support)
    n_support : int
    n_dmap_components : int
    gw_epsilon : float
    n_workers : int

    Returns
    -------
    np.ndarray, shape (n_valid, n_support^2), dtype float32
        Flattened transport plans for valid series.
    """
    from diffuse import dmap_dense, subsample_fps, intra_distance_matrix

    def _process_one(dvecs, tau):
        """Process one series: DMAP → dist matrix → GW transport → flatten."""
        n = dvecs.shape[0]
        # Subsample to n_support via FPS
        if n > n_support:
            dvecs_sub, _ = subsample_fps(dvecs, n_support)
        else:
            dvecs_sub = dvecs[:n_support]

        if dvecs_sub.shape[0] < n_support:
            return None

        # Per-series DMAP
        n_comp = min(n_dmap_components, n_support - 1)
        coords, _ = dmap_dense(dvecs_sub, n_components=n_comp,
                               alpha=1.0, theiler=tau)

        # Intra-distance matrix
        D = intra_distance_matrix(coords)

        # GW transport to barycenter (this is the key step)
        T_flat = compute_transport_plan(D, barycenter_dist_matrix, gw_epsilon)

        # Distance matrix D is NOT stored — only T_flat survives
        return T_flat

    if n_workers <= 1:
        plans = []
        for dvecs, tau in zip(dvecs_list, taus):
            try:
                t = _process_one(dvecs, tau)
                if t is not None:
                    plans.append(t)
            except Exception:
                pass
        return np.array(plans) if plans else np.empty((0, n_support ** 2), dtype=np.float32)

    # Parallel: use top-level worker
    from concurrent.futures import ProcessPoolExecutor
    work = [(dvecs, tau, barycenter_dist_matrix, n_support,
             n_dmap_components, gw_epsilon) for dvecs, tau in zip(dvecs_list, taus)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_streaming_worker, work, chunksize=50))

    plans = [r for r in results if r is not None]
    return np.array(plans) if plans else np.empty((0, n_support ** 2), dtype=np.float32)


def _streaming_worker(args):
    """Top-level worker for streaming transport plan computation."""
    dvecs, tau, bary_dm, n_sup, n_comp, eps = args
    from diffuse import dmap_dense, subsample_fps, intra_distance_matrix

    try:
        n = dvecs.shape[0]
        if n > n_sup:
            dvecs_sub, _ = subsample_fps(dvecs, n_sup)
        else:
            dvecs_sub = dvecs[:n_sup]

        if dvecs_sub.shape[0] < n_sup:
            return None

        nc = min(n_comp, n_sup - 1)
        coords, _ = dmap_dense(dvecs_sub, n_components=nc, alpha=1.0, theiler=tau)
        D = intra_distance_matrix(coords)
        T_flat = compute_transport_plan(D, bary_dm, eps)
        return T_flat
    except Exception:
        return None


# ============================================================
# Multi-barycenter adaptive architecture
# ============================================================

def plan_entropy(T_flat: np.ndarray) -> float:
    """
    Shannon entropy of a flattened transport plan.

    Low entropy = structured coupling (series matches barycenter).
    High entropy = near-uniform (poor structural match).
    Bounded: H in [0, 2*log(n_support)].
    """
    T = np.maximum(T_flat, 1e-15)
    return float(-np.sum(T * np.log(T)))


def gw_transport_cost_from_plan(
    T_flat: np.ndarray,
    series_dist_matrix: np.ndarray,
    barycenter_dist_matrix: np.ndarray,
) -> float:
    """
    Extract GW cost from an already-computed transport plan.

    Avoids redundant GW solves: compute plan once, extract cost from it.
    Cost = sum_{a,b,c,d} (C1[a,b] - C2[c,d])^2 * T[a,c] * T[b,d]
    """
    n = series_dist_matrix.shape[0]
    s_max = series_dist_matrix.max()
    b_max = barycenter_dist_matrix.max()
    C1 = series_dist_matrix / s_max if s_max > 1e-10 else series_dist_matrix
    C2 = barycenter_dist_matrix / b_max if b_max > 1e-10 else barycenter_dist_matrix

    T = T_flat.reshape(n, n).astype(np.float64)
    # GW loss via marginal formulation (O(n^2), not O(n^4)):
    # gwloss = sum_{a,b,c,d} (C1_ab - C2_cd)^2 T_ac T_bd
    #        = p @ C1^2 @ p + q @ C2^2 @ q - 2 * trace(C1 @ T @ C2 @ T^T)
    # where p = T @ 1 (row marginals), q = T^T @ 1 (column marginals)
    p = T.sum(axis=1)
    q = T.sum(axis=0)
    C1T = C1 @ T
    TC2 = T @ C2
    cost = float(p @ (C1 ** 2) @ p + q @ (C2 ** 2) @ q - 2 * np.trace(C1T @ TC2.T))
    return cost


def assign_to_barycenters(
    series_dist_matrix: np.ndarray,
    barycenters: dict,
    epsilon: float = 0.01,
    soft: bool = True,
) -> dict:
    """
    Route a series to its best-matching barycenter(s).

    Uses plan entropy (not GW cost) for routing — entropy is bounded,
    automatically calibrated, and computed from the plan already in hand.
    Low entropy = structured coupling = good match.

    Parameters
    ----------
    series_dist_matrix : np.ndarray, shape (n, n)
    barycenters : dict
        {domain_name: barycenter_dist_matrix}
    epsilon : float
    soft : bool
        If True, return softmax over entropies (soft assignment).

    Returns
    -------
    dict with keys:
        'best': str — best-matching domain (lowest entropy)
        'entropies': dict[str, float] — plan entropy per barycenter
        'weights': dict[str, float] — soft assignment weights (if soft=True)
        'transport_plans': dict[str, np.ndarray] — plan to each barycenter
    """
    entropies = {}
    plans = {}
    for name, bary in barycenters.items():
        # Single GW solve per barycenter (no redundancy)
        plans[name] = compute_transport_plan(series_dist_matrix, bary, epsilon)
        entropies[name] = plan_entropy(plans[name])

    best = min(entropies, key=entropies.get)

    result = {'best': best, 'entropies': entropies, 'transport_plans': plans}

    if soft:
        # Softmax: lower entropy → higher weight
        ent_arr = np.array([entropies[k] for k in barycenters])
        # Temperature = IQR of entropies (spread-adaptive, not level-dependent)
        iqr = np.percentile(ent_arr, 75) - np.percentile(ent_arr, 25)
        temp = max(iqr, np.std(ent_arr), 1e-10)
        weights_arr = np.exp(-ent_arr / temp)
        weights_arr = weights_arr / weights_arr.sum()
        result['weights'] = {k: float(w) for k, w in zip(barycenters, weights_arr)}

    return result


def multi_barycenter_dmap(
    plans_by_domain: dict,
    n_components: int = 20,
    n_landmarks: int = 800,
    alpha: float = 1.0,
    k_bandwidth: int = 10,
    min_gap: float = 1.5,
) -> dict:
    """
    Per-domain DMAP with quality monitoring.

    Parameters
    ----------
    plans_by_domain : dict
        {domain: np.ndarray of shape (n_series, n_support^2)}
    min_gap : float
        Spectral gap threshold. Below this, domain should be split.

    Returns
    -------
    dict with per-domain results:
        {domain: {'coords': ndarray, 'evals': ndarray, 'gap': float,
                  'needs_split': bool}}
    """
    results = {}
    for domain, plans in plans_by_domain.items():
        if len(plans) < 20:
            results[domain] = {
                'coords': plans, 'evals': np.array([]),
                'gap': 0.0, 'needs_split': False,
                'reason': 'too_few_series',
            }
            continue

        coords, evals = transport_dmap(
            plans, n_components=n_components,
            n_landmarks=min(n_landmarks, len(plans)),
            alpha=alpha, k_bandwidth=k_bandwidth,
        )
        gap = evals[0] / evals[1] if len(evals) >= 2 and evals[1] > 1e-10 else 0

        # Diagnose spectral structure (split vs continuum vs noise)
        diag = diagnose_spectral_structure(evals, n_series=len(plans))
        needs_split = (
            gap < min_gap
            and len(plans) > 100
            and diag['diagnosis'] == 'split'  # only split if structure says so
        )

        results[domain] = {
            'coords': coords,
            'evals': evals,
            'gap': float(gap),
            'needs_split': needs_split,
            'spectral_diagnosis': diag,
            'reason': (f"gap={gap:.2f}<{min_gap}, diag={diag['diagnosis']}"
                      if needs_split else f"ok (gap={gap:.2f}, diag={diag['diagnosis']})"),
        }

    return results


def split_domain(
    plans: np.ndarray,
    k: int = 2,
) -> List[np.ndarray]:
    """
    Split a domain into k sub-domains via spectral clustering
    on the Hellinger affinity matrix.

    Spectral clustering is consistent with the DMAP philosophy:
    both use spectral methods on the same Hellinger geometry.
    Unlike Ward linkage, spectral clustering allows unequal cluster
    sizes (important: monetary=200 vs solow=2000 is expected).

    Returns list of index arrays, one per cluster.
    """
    from sklearn.cluster import SpectralClustering

    sqrt_plans = np.sqrt(plans.astype(np.float64))
    dists = cdist(sqrt_plans, sqrt_plans)

    # Gaussian affinity with median bandwidth
    sigma = max(np.median(dists), 1e-10)
    affinity = np.exp(-dists ** 2 / (2 * sigma ** 2))

    sc = SpectralClustering(
        n_clusters=k, affinity='precomputed', random_state=42,
    )
    labels = sc.fit_predict(affinity)

    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            clusters.append(idx)

    return clusters


def diagnose_spectral_structure(
    evals: np.ndarray,
    n_series: Optional[int] = None,
    n_support: Optional[int] = None,
) -> dict:
    """
    Diagnose whether low spectral gap means 'needs split' or 'high-dim continuum'.

    Examines the full eigenvalue spectrum, not just the first gap ratio.
    A clean split: lambda_1 >> lambda_2 >> lambda_3.
    A continuum: gradual decay across many eigenvalues.
    """
    if len(evals) < 3:
        return {'diagnosis': 'insufficient_evals', 'n_evals': len(evals)}

    ratios = evals[:-1] / np.maximum(evals[1:], 1e-10)
    max_gap_idx = int(np.argmax(ratios))
    max_gap_val = float(ratios[max_gap_idx])

    # Count eigenvalues above noise floor (2x last eigenvalue)
    noise_floor = evals[-1] * 2
    n_signal = int(np.sum(evals > noise_floor))

    # Marcenko-Pastur upper edge for noise eigenvalues
    # gamma = ambient_dim / n_series. For transport plans, ambient_dim = n_support^2.
    mp_upper = None
    if n_series is not None and n_series > 0:
        ambient_dim = n_support ** 2 if n_support else len(evals)
        gamma = ambient_dim / n_series
        if gamma < 1:
            mp_upper = (1 + np.sqrt(gamma)) ** 2

    # Diagnosis logic
    if max_gap_idx == 0 and max_gap_val > 1.5:
        diagnosis = 'split'  # dominant first gap → two clusters
    elif max_gap_idx > 0:
        diagnosis = 'high_dim'  # max gap not at position 0 → continuum
    else:
        diagnosis = 'weak'  # no clear structure

    return {
        'diagnosis': diagnosis,
        'max_gap_position': max_gap_idx,
        'max_gap_value': max_gap_val,
        'n_signal_evals': n_signal,
        'mp_upper': mp_upper,
        'top_5_ratios': [float(r) for r in ratios[:5]],
    }


# ============================================================
# Bandwidth persistence for robust eigengap
# ============================================================

def bandwidth_persistence(
    plans: np.ndarray,
    k_range: Optional[List[int]] = None,
    n_components: int = 20,
    n_landmarks: int = 800,
    alpha: float = 1.0,
) -> dict:
    """
    Sweep DMAP bandwidth (k-th NN) and find the most persistent eigengap.

    The eigengap for automatic k-selection depends on kernel bandwidth.
    We sweep k_bandwidth over a range and find where the eigengap is
    STABLE (persists across bandwidths). This is the spectral analogue
    of persistent homology — topological features stable across scales
    are real; transient features are noise.

    Parameters
    ----------
    plans : np.ndarray, shape (n, d)
    k_range : list of int, optional
        k-th NN values to sweep. Default: [3, 5, 7, 10, 15, 20, 30, 50].
    n_components : int
    n_landmarks : int
    alpha : float

    Returns
    -------
    dict with:
        'best_k': int — k-th NN with most persistent eigengap
        'best_auto_k': int — most persistent cluster count
        'persistence': list of (k_bandwidth, auto_k, gap_value, evals[:5])
        'k_histogram': dict[int, int] — count of how many bandwidths give each auto_k
        'stable_k': int — auto_k that appears in the widest contiguous bandwidth range
    """
    if k_range is None:
        n = plans.shape[0]
        # Scale k_range with corpus size
        max_k = min(50, n // 4)
        k_range = sorted(set([3, 5, 7, 10, 15, 20, 30, min(50, max_k)]))
        k_range = [k for k in k_range if k < n]

    # Pre-compute shared data: sqrt embedding, FPS landmarks, distance matrices.
    # This avoids recomputing FPS 8x (~7x speedup for large corpora).
    n = plans.shape[0]
    n_lm = min(n_landmarks, n)
    sqrt_plans = np.sqrt(plans.astype(np.float64))

    if n <= n_lm:
        # Dense path: all points are landmarks
        lm_idx = np.arange(n)
        sqrt_lm = sqrt_plans
        dists_lm = cdist(sqrt_lm, sqrt_lm)
        dists_to_lm = dists_lm  # same matrix
        is_dense = True
    else:
        # Nystrom: FPS once, reuse across bandwidths
        lm_idx = _fps_hellinger(sqrt_plans, n_lm)
        sqrt_lm = sqrt_plans[lm_idx]
        dists_lm = cdist(sqrt_lm, sqrt_lm)
        dists_to_lm = cdist(sqrt_plans, sqrt_lm)
        is_dense = False

    persistence = []
    for k_bw in k_range:
        try:
            if is_dense:
                coords, evals = _transport_dmap_dense(
                    sqrt_plans, n_components, alpha, k_bw)
            else:
                # Dense DMAP on landmarks with this bandwidth
                coords_lm, evals_lm = _transport_dmap_dense(
                    sqrt_lm, n_components, alpha, k_bw)
                # Nystrom extend (reusing precomputed distances)
                coords = _nystrom_extend(
                    sqrt_plans, sqrt_lm, lm_idx, coords_lm, evals_lm,
                    alpha, k_bw)
                evals = evals_lm

            if len(evals) < 3:
                continue
            ratios = evals[:-1] / np.maximum(evals[1:], 1e-10)
            # Eigengap heuristic: our evals skip trivial λ₀=1.
            # ratios[j] = λ_{j+1}/λ_{j+2}. Max ratio at position j means
            # j+2 near-unity eigenvalues (including λ₀=1) → j+2 clusters.
            gap_pos = int(np.argmax(ratios[:min(15, len(ratios))]))
            auto_k = gap_pos + 2
            gap_val = float(ratios[gap_pos])
            persistence.append((k_bw, auto_k, gap_val,
                               [float(e) for e in evals[:5]]))
        except Exception:
            continue

    if not persistence:
        return {'best_k': 10, 'best_auto_k': 2, 'persistence': [],
                'k_histogram': {}, 'stable_k': 2, 'contiguous_run': 0}

    # Weighted histogram: auto_k weighted by gap magnitude
    # (total persistence, not just count — an auto_k with large gaps
    # is more meaningful than one with many tiny gaps)
    k_hist = {}
    k_count = {}
    for _, auto_k, gap_val, _ in persistence:
        k_hist[auto_k] = k_hist.get(auto_k, 0) + gap_val
        k_count[auto_k] = k_count.get(auto_k, 0) + 1

    # Most common auto_k = most persistent
    most_common_k = max(k_hist, key=k_hist.get)

    # Find longest contiguous run of the most common auto_k
    auto_ks = [p[1] for p in persistence]
    max_run = 0
    cur_run = 0
    stable_k = most_common_k
    for ak in auto_ks:
        if ak == most_common_k:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # Best k_bandwidth = middle of the longest contiguous run
    run_indices = []
    cur_start = None
    for i, ak in enumerate(auto_ks):
        if ak == most_common_k:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                run_indices.append((cur_start, i))
                cur_start = None
    if cur_start is not None:
        run_indices.append((cur_start, len(auto_ks)))

    if run_indices:
        longest = max(run_indices, key=lambda x: x[1] - x[0])
        mid = (longest[0] + longest[1]) // 2
        best_k_bw = persistence[mid][0]
    else:
        best_k_bw = 10

    return {
        'best_k': best_k_bw,
        'k_count': k_count,  # unweighted count for diagnostics
        'best_auto_k': most_common_k,
        'persistence': persistence,
        'k_histogram': k_hist,
        'stable_k': stable_k,
        'contiguous_run': max_run,
    }


# ============================================================
# Eigengap stability check (empirical substitute for
# formal stability theorems like bottleneck stability in PH)
# ============================================================

def eigengap_stability(
    plans: np.ndarray,
    n_bootstrap: int = 20,
    subsample_frac: float = 0.7,
    k_range: Optional[List[int]] = None,
    n_components: int = 20,
    n_landmarks: int = 800,
    alpha: float = 1.0,
) -> dict:
    """
    Empirical stability of the eigengap across subsamples AND bandwidths.

    No formal stability theorem exists for spectral eigengaps (unlike
    bottleneck stability for persistent homology). This function provides
    an empirical substitute: bootstrap subsample the plans, sweep bandwidth,
    and check if the eigengap position (auto_k) is consistent.

    Returns
    -------
    dict with:
        'stable': bool — True if auto_k agrees across >80% of trials
        'consensus_k': int — most common auto_k
        'agreement': float — fraction of trials agreeing with consensus
        'grid': list of (k_bw, subsample_idx, auto_k, gap_val)
        'k_by_bandwidth': dict[int, list] — auto_k distribution per bandwidth
    """
    n = plans.shape[0]
    if k_range is None:
        max_k = min(30, n // 5)
        k_range = [k for k in [5, 10, 15, 20, 30] if k < max_k]
    if not k_range:
        k_range = [min(5, n // 3)]

    sub_size = max(20, int(n * subsample_frac))
    grid = []
    k_by_bw = {k: [] for k in k_range}

    for k_bw in k_range:
        for rep in range(n_bootstrap):
            idx = np.random.choice(n, sub_size, replace=False)
            try:
                _, evals = transport_dmap(
                    plans[idx], n_components=n_components,
                    n_landmarks=min(n_landmarks, sub_size),
                    alpha=alpha, k_bandwidth=k_bw,
                )
                if len(evals) < 3:
                    continue
                ratios = evals[:-1] / np.maximum(evals[1:], 1e-10)
                gap_pos = int(np.argmax(ratios[:min(15, len(ratios))]))
                auto_k = gap_pos + 2
                gap_val = float(ratios[gap_pos])
                grid.append((k_bw, rep, auto_k, gap_val))
                k_by_bw[k_bw].append(auto_k)
            except Exception:
                continue

    if not grid:
        return {'stable': False, 'consensus_k': 2, 'agreement': 0.0,
                'grid': [], 'k_by_bandwidth': k_by_bw}

    # Consensus: most common auto_k across ALL trials
    all_ks = [g[2] for g in grid]
    from collections import Counter
    counter = Counter(all_ks)
    consensus_k = counter.most_common(1)[0][0]
    agreement = counter[consensus_k] / len(all_ks)

    return {
        'stable': agreement > 0.8,
        'consensus_k': consensus_k,
        'agreement': float(agreement),
        'grid': grid,
        'k_by_bandwidth': {k: v for k, v in k_by_bw.items()},
    }


# ============================================================
# Intrinsic dimensionality from DMAP eigenspectrum
# ============================================================

def dmap_intrinsic_dimension(
    evals: np.ndarray,
    method: str = 'profile',
) -> dict:
    """
    Estimate intrinsic dimension of the transport-plan manifold
    from DMAP eigenvalues. NOT PCA — these are spectral properties
    of the Laplace-Beltrami operator on the Hellinger-embedded
    Birkhoff polytope.

    Unlike PCA participation ratio (which measures linear variance
    directions and is meaningless on a nonlinear manifold), this
    measures the number of independent diffusion modes — each
    corresponding to an independent axis of attractor variation.

    Parameters
    ----------
    evals : np.ndarray
        DMAP eigenvalues (excluding trivial λ=1).
    method : str
        'profile': eigenvalue profile analysis (default)
        'ratio': successive ratio test
        'both': return both estimates

    Returns
    -------
    dict with:
        'd_eff': int — estimated intrinsic dimension
        'method': str
        'eigenvalue_profile': list of (index, eigenvalue, ratio_to_next)
        'noise_floor': float
        'd_ratio': int — dimension from successive ratio test (if 'both')
    """
    if len(evals) < 3:
        return {'d_eff': len(evals), 'method': 'trivial'}

    # Profile method: find where eigenvalues transition from
    # power-law decay (signal) to exponential decay (noise).
    # On a d-dimensional manifold, the first d eigenvalues of
    # Laplace-Beltrami follow Weyl's law: λ_k ~ k^{2/d}.
    # Noise eigenvalues decay faster (exponentially).
    log_evals = np.log(np.maximum(evals, 1e-15))
    ratios = evals[:-1] / np.maximum(evals[1:], 1e-10)

    # Successive ratio test: eigenvalue ratios should be ~constant
    # in the signal regime (Weyl: λ_k/λ_{k+1} ≈ ((k+1)/k)^{2/d})
    # and jump up at the signal-noise boundary.
    # Find the first ratio that exceeds 2x the median of early ratios.
    early_median = np.median(ratios[:min(5, len(ratios))])
    d_ratio = len(evals)  # default: all are signal
    for i in range(2, len(ratios)):
        if ratios[i] > 2 * early_median and ratios[i] > 1.5:
            d_ratio = i + 1
            break

    # Profile method: fit piecewise linear to log-eigenvalues
    # Knee detection via maximum curvature
    if len(log_evals) >= 5:
        # Discrete second derivative of log-eigenvalues
        d2 = np.diff(log_evals, n=2)
        # Most negative d2 = sharpest downward bend = knee
        knee_idx = int(np.argmin(d2)) + 2  # +2 for diff offset
        d_profile = min(knee_idx, len(evals))
    else:
        d_profile = len(evals)

    # Noise floor estimate
    noise_floor = float(evals[-1]) if len(evals) > 0 else 0.0
    n_above_noise = int(np.sum(evals > 2 * noise_floor))

    profile = [(i, float(evals[i]), float(ratios[i]) if i < len(ratios) else 0.0)
               for i in range(min(15, len(evals)))]

    result = {
        'd_eff': d_profile,
        'method': 'profile',
        'eigenvalue_profile': profile,
        'noise_floor': noise_floor,
        'n_above_noise': n_above_noise,
    }
    if method in ('ratio', 'both'):
        result['d_ratio'] = d_ratio
    if method == 'ratio':
        result['d_eff'] = d_ratio
        result['method'] = 'ratio'

    return result


# ============================================================
# Domain discovery pipeline (full stack)
# ============================================================

def discover_domains(
    plans: np.ndarray,
    n_components: int = 20,
    n_landmarks: int = 800,
    alpha: float = 1.0,
    min_domain_size: int = 30,
    run_stability: bool = True,
) -> dict:
    """
    Full domain discovery pipeline:
    1. Bandwidth persistence → robust eigengap → auto k
    2. Spectral clustering at auto k → domain assignments
    3. Per-domain DMAP → per-domain coordinates + quality check
    4. Intrinsic dimensionality per domain
    5. (Optional) Eigengap stability check (50 DMAP calls)

    This is the main entry point for the AoA platform's domain
    discovery feature. Given universal transport plans, it discovers
    the natural dynamical families and computes per-family coordinates.

    Parameters
    ----------
    plans : np.ndarray, shape (n, n_support^2)
    n_components : int
    n_landmarks : int
    alpha : float
    min_domain_size : int
        Minimum series count for a domain to get its own DMAP.

    Returns
    -------
    dict with:
        'n_domains': int
        'bandwidth_persistence': dict — full persistence analysis
        'domains': dict[int, dict] — per-domain results:
            'indices': ndarray — indices into plans array
            'coords': ndarray — DMAP coordinates
            'evals': ndarray — eigenvalues
            'gap': float — spectral gap
            'd_eff': dict — intrinsic dimension estimate
            'size': int
        'universal_coords': ndarray — fallback universal coordinates
        'universal_evals': ndarray
    """
    n = plans.shape[0]

    # Guard: need minimum series for meaningful DMAP
    if n < 20:
        return {
            'n_domains': 0,
            'bandwidth_persistence': {},
            'domains': {},
            'universal_coords': plans if n > 0 else np.empty((0, 0)),
            'universal_evals': np.array([]),
            'error': f'insufficient_plans ({n} < 20)',
        }

    # Step 0: Universal DMAP as fallback
    universal_coords, universal_evals = transport_dmap(
        plans, n_components=n_components,
        n_landmarks=min(n_landmarks, n),
        alpha=alpha, k_bandwidth=10,
    )

    # Step 1: Bandwidth persistence → auto k
    bp = bandwidth_persistence(
        plans, n_components=n_components,
        n_landmarks=min(n_landmarks, n), alpha=alpha,
    )
    auto_k = bp['best_auto_k']

    # Guard: if auto_k=1 or persistence is weak, return universal
    if auto_k <= 1 or bp['contiguous_run'] < 2:
        dim_est = dmap_intrinsic_dimension(universal_evals, method='both')
        return {
            'n_domains': 1,
            'bandwidth_persistence': bp,
            'domains': {0: {
                'indices': np.arange(n),
                'coords': universal_coords,
                'evals': universal_evals,
                'gap': float(universal_evals[0] / universal_evals[1])
                       if len(universal_evals) >= 2 else 0,
                'd_eff': dim_est,
                'size': n,
            }},
            'universal_coords': universal_coords,
            'universal_evals': universal_evals,
        }

    # Step 2: Spectral clustering at auto k (using persistent bandwidth)
    # Recompute DMAP with best bandwidth for eigenvector-based clustering
    coords_best, evals_best = transport_dmap(
        plans, n_components=n_components,
        n_landmarks=min(n_landmarks, n),
        alpha=alpha, k_bandwidth=bp['best_k'],
    )

    # Cluster on the DMAP coordinates (not raw plans — more efficient)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=auto_k, random_state=42, n_init=10)
    labels = km.fit_predict(coords_best[:, :min(auto_k + 2, n_components)])

    # Step 3: Per-domain DMAP
    domains = {}
    for c in range(auto_k):
        idx = np.where(labels == c)[0]
        if len(idx) < min_domain_size:
            # Too small — merge into nearest domain later
            domains[c] = {
                'indices': idx,
                'coords': coords_best[idx],
                'evals': np.array([]),
                'gap': 0.0,
                'd_eff': {'d_eff': 0, 'method': 'too_small'},
                'size': len(idx),
            }
            continue

        domain_plans = plans[idx]
        d_coords, d_evals = transport_dmap(
            domain_plans, n_components=n_components,
            n_landmarks=min(n_landmarks, len(domain_plans)),
            alpha=alpha, k_bandwidth=bp['best_k'],
        )
        d_gap = (d_evals[0] / d_evals[1]
                 if len(d_evals) >= 2 and d_evals[1] > 1e-10 else 0)
        dim_est = dmap_intrinsic_dimension(d_evals, method='both')

        domains[c] = {
            'indices': idx,
            'coords': d_coords,
            'evals': d_evals,
            'gap': float(d_gap),
            'd_eff': dim_est,
            'size': len(idx),
        }

    # Step 4: Eigengap stability check (optional — 50 DMAP calls)
    # This is the empirical substitute for a formal stability theorem.
    # If agreement < 80%, the domain count is not robust.
    stability = None
    if run_stability and n >= 50:
        stability = eigengap_stability(
            plans, n_bootstrap=10, subsample_frac=0.7,
            n_components=n_components, n_landmarks=min(n_landmarks, n),
            alpha=alpha,
        )

    return {
        'n_domains': auto_k,
        'bandwidth_persistence': bp,
        'eigengap_stability': stability,
        'domains': domains,
        'universal_coords': universal_coords,
        'universal_evals': universal_evals,
    }


# ============================================================
# Axis interpretability
# ============================================================

def interpret_aoa_axes(
    aoa_coords: np.ndarray,
    descriptors: np.ndarray,
    descriptor_names: Optional[List[str]] = None,
    n_axes: int = 10,
    method: str = 'spearman',
) -> dict:
    """
    Correlate each AoA DMAP coordinate axis with the 16 attractor descriptors.

    This answers: what does each AoA dimension physically represent?
    E.g., axis 1 might correlate with D2 (complexity gradient),
    axis 2 with PE (entropy), axis 3 with PH_total (topology).

    Note: AoA coords are eigenvalue-weighted (axis 1 has more variance than
    axis 10 by construction). This is correct — eigenvalue-weighted coords
    encode diffusion distance, which is what clustering and distances use.

    Limitations: Spearman captures monotone relationships only. If an axis
    captures |D2 - 3| or a U-shaped function of lambda_max, Spearman will
    report low correlation. Use method='hsic' for non-monotone detection.

    Parameters
    ----------
    aoa_coords : np.ndarray, shape (n, d)
        DMAP diffusion coordinates (from transport_dmap or discover_domains).
    descriptors : np.ndarray, shape (n, 16)
        Attractor descriptor matrix. NaN/inf entries are handled per-pair.
    descriptor_names : list of str, optional
        Names for the 16 descriptors. Default: imported from descriptors.py.
    n_axes : int
        Number of AoA axes to analyze (default 10).
    method : str
        'spearman' (rank, robust to nonlinearity — default),
        'pearson' (linear), or
        'hsic' (kernel-based, detects arbitrary nonlinear dependence).

    Returns
    -------
    dict with:
        'correlations': np.ndarray, shape (n_axes, n_descriptors)
        'p_values': np.ndarray — raw p-values
        'p_values_fdr': np.ndarray — BH-FDR adjusted p-values
        'top_descriptors': list of list of (name, corr, p_fdr, ci_lo, ci_hi)
        'axis_labels': list of str — human-readable label per axis
        'collinear_groups': list of list of str — descriptor groups with |r|>0.7
        'descriptor_names': list of str
    """
    from scipy.stats import spearmanr, pearsonr
    import warnings as _warn

    if method not in ('spearman', 'pearson', 'hsic'):
        raise ValueError(f"method must be 'spearman', 'pearson', or 'hsic', got {method!r}")

    if descriptor_names is None:
        try:
            from descriptors import DESCRIPTOR_NAMES
            descriptor_names = list(DESCRIPTOR_NAMES)
        except ImportError:
            descriptor_names = [f'desc_{i}' for i in range(descriptors.shape[1])]

    n = aoa_coords.shape[0]
    if n < 10:
        _warn.warn(f"interpret_aoa_axes: only {n} samples, results unreliable", stacklevel=2)

    n_ax = min(n_axes, aoa_coords.shape[1])
    n_desc = min(descriptors.shape[1], len(descriptor_names))

    corr_matrix = np.full((n_ax, n_desc), np.nan)
    pval_matrix = np.full((n_ax, n_desc), np.nan)

    for ax in range(n_ax):
        for di in range(n_desc):
            # BUG-4 fix: use isfinite for BOTH (catches NaN, inf, -inf)
            valid = np.isfinite(descriptors[:, di]) & np.isfinite(aoa_coords[:, ax])
            n_valid = int(np.sum(valid))
            if n_valid < 10:
                continue

            x = aoa_coords[valid, ax]
            y = descriptors[valid, di]

            if np.std(x) < 1e-15 or np.std(y) < 1e-15:
                continue

            if method == 'spearman':
                r, p = spearmanr(x, y)
            elif method == 'pearson':
                r, p = pearsonr(x, y)
            elif method == 'hsic':
                # Kernel-based independence: detects arbitrary nonlinear dependence
                r, p = _hsic_test(x, y)

            corr_matrix[ax, di] = r
            pval_matrix[ax, di] = p

    # FDR correction (Benjamini-Hochberg) across all 160 tests
    fdr_matrix = np.full_like(pval_matrix, np.nan)
    flat_p = pval_matrix.ravel()
    valid_mask = np.isfinite(flat_p)
    if np.sum(valid_mask) > 0:
        adjusted = _benjamini_hochberg(flat_p[valid_mask])
        fdr_flat = np.full_like(flat_p, np.nan)
        fdr_flat[valid_mask] = adjusted
        fdr_matrix = fdr_flat.reshape(pval_matrix.shape)

    # Descriptor collinearity groups (|r| > 0.7 between descriptors)
    collinear_groups = _find_collinear_groups(descriptors, descriptor_names, threshold=0.7)

    # Top descriptors per axis (by |correlation|, with FDR p-value and CI)
    top_descriptors = []
    axis_labels = []
    for ax in range(n_ax):
        row = corr_matrix[ax]
        valid_idx = np.isfinite(row)
        if not np.any(valid_idx):
            top_descriptors.append([])
            axis_labels.append(f'axis_{ax+1}')
            continue

        sorted_idx = np.argsort(np.abs(np.where(valid_idx, row, 0)))[::-1]

        # Skip descriptors collinear with a higher-ranked one
        seen_groups = set()
        top_entries = []
        for idx in sorted_idx:
            if np.isnan(row[idx]):
                continue
            name = descriptor_names[idx]
            # Check if this descriptor is collinear with one already selected
            in_seen = False
            for grp in collinear_groups:
                if name in grp and any(g in seen_groups for g in grp):
                    in_seen = True
                    break
            if in_seen:
                continue

            n_valid_ax = int(np.sum(np.isfinite(descriptors[:, idx]) & np.isfinite(aoa_coords[:, ax])))
            ci_lo, ci_hi = _fisher_z_ci(float(row[idx]), n_valid_ax)
            top_entries.append((
                name, float(row[idx]),
                float(fdr_matrix[ax, idx]),
                ci_lo, ci_hi,
            ))
            seen_groups.add(name)
            if len(top_entries) >= 3:
                break
        top_descriptors.append(top_entries)

        # Human-readable label
        if top_entries and abs(top_entries[0][1]) > 0.3:
            sign = '+' if top_entries[0][1] > 0 else '−'
            star = '*' if top_entries[0][2] < 0.05 else ''
            axis_labels.append(f'{sign}{top_entries[0][0]} (r={top_entries[0][1]:.2f}{star})')
        else:
            axis_labels.append(f'axis_{ax+1} (mixed)')

    return {
        'correlations': corr_matrix,
        'p_values': pval_matrix,
        'p_values_fdr': fdr_matrix,
        'top_descriptors': top_descriptors,
        'axis_labels': axis_labels,
        'collinear_groups': collinear_groups,
        'descriptor_names': descriptor_names[:n_desc],
    }


def _hsic_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Kernel HSIC independence test. Returns (hsic_stat, p_value)."""
    from scipy.spatial.distance import pdist, squareform
    n = len(x)
    # RBF kernels with median bandwidth
    dx = pdist(x[:, None])
    dy = pdist(y[:, None])
    sx = max(np.median(dx), 1e-10)
    sy = max(np.median(dy), 1e-10)
    Kx = np.exp(-squareform(dx) ** 2 / (2 * sx ** 2))
    Ky = np.exp(-squareform(dy) ** 2 / (2 * sy ** 2))
    H = np.eye(n) - 1.0 / n
    hsic = np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2
    # Permutation p-value (fast, 200 permutations)
    null_hsic = []
    for _ in range(200):
        perm = np.random.permutation(n)
        null_hsic.append(np.trace(Kx @ H @ Ky[perm][:, perm] @ H) / (n - 1) ** 2)
    p = float(np.mean(np.array(null_hsic) >= hsic))
    return float(hsic), max(p, 1.0 / 201)  # floor at 1/201


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    result = np.zeros(n)
    result[sorted_idx] = np.clip(adjusted, 0, 1)
    return result


def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher-z 95% confidence interval for a correlation coefficient."""
    if n < 4:
        return (-1.0, 1.0)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


def _find_collinear_groups(
    descriptors: np.ndarray,
    names: List[str],
    threshold: float = 0.7,
) -> List[List[str]]:
    """Find groups of descriptors with |Spearman r| > threshold."""
    from scipy.stats import spearmanr
    n_desc = min(descriptors.shape[1], len(names))
    # Compute pairwise Spearman between descriptors
    groups = []
    assigned = set()
    for i in range(n_desc):
        if names[i] in assigned:
            continue
        group = [names[i]]
        for j in range(i + 1, n_desc):
            if names[j] in assigned:
                continue
            valid = np.isfinite(descriptors[:, i]) & np.isfinite(descriptors[:, j])
            if np.sum(valid) < 10:
                continue
            r, _ = spearmanr(descriptors[valid, i], descriptors[valid, j])
            if abs(r) > threshold:
                group.append(names[j])
        if len(group) > 1:
            groups.append(group)
            assigned.update(group)
    return groups


def print_axis_interpretation(result: dict, max_axes: int = 8) -> None:
    """Pretty-print axis interpretation results."""
    n_ax = min(max_axes, len(result['top_descriptors']))

    # Show collinear groups first
    if result.get('collinear_groups'):
        print("Collinear descriptor groups (|r|>0.7):")
        for grp in result['collinear_groups']:
            print(f"  {' ~ '.join(grp)}")
        print()

    print(f"{'Axis':>6}  {'Label':>30}  {'Top correlations (FDR-corrected)':>55}")
    print("-" * 95)
    for ax in range(n_ax):
        label = result['axis_labels'][ax]
        top = result['top_descriptors'][ax]
        if top:
            parts = []
            for name, r, p_fdr, ci_lo, ci_hi in top[:3]:
                star = '*' if p_fdr < 0.05 else ''
                parts.append(f"{name}={r:+.2f}{star}[{ci_lo:+.2f},{ci_hi:+.2f}]")
            corrs = ', '.join(parts)
        else:
            corrs = 'none'
        print(f"  ψ_{ax+1:>2}  {label:>30}  {corrs}")


# ============================================================
# Multi-Reference Transport Plans (Cross-Domain Coupling)
# ============================================================

def compute_multi_reference_plans(
    series_dist_matrices: List[np.ndarray],
    barycenter_dist_matrices: List[np.ndarray],
    primary_domain_idx: np.ndarray,
    epsilon: float = 0.01,
    probe_epsilon: float = 0.1,
    entropy_gate: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-reference transport plans with entropy gating.

    Each series gets K transport plans (one per domain barycenter).
    Entropy gating: probe foreign barycenters with large epsilon first;
    only compute full-precision plans where H/H_max < gate (indicating
    genuine cross-domain coupling). This is a MEASUREMENT from the GW
    geometry, not a supervised signal.

    The coupling signal S_ik = 1 - H(T_i^(k))/H_max is the natural
    measure: low entropy = structured plan = strong coupling.
    Asymmetric: S(monetary→solow) ≠ S(solow→monetary) because the
    underlying dynamical coupling is directed.

    Parameters
    ----------
    series_dist_matrices : list of np.ndarray, shape (n_sup, n_sup) each
    barycenter_dist_matrices : list of np.ndarray, shape (n_sup, n_sup) each
        K domain barycenters.
    primary_domain_idx : np.ndarray, shape (n_series,)
        Which domain each series belongs to (0..K-1).
    epsilon : float
        Full-precision GW regularization.
    probe_epsilon : float
        Fast probe regularization (larger = faster, less precise).
    entropy_gate : float
        H/H_max threshold. Plans above this are replaced with uniform.

    Returns
    -------
    plans : np.ndarray, shape (n_series, K * n_support^2)
        Concatenated multi-reference plans.
    coupling_signal : np.ndarray, shape (n_series, K)
        S_ik = 1 - H(T_i^(k)) / H_max.
    """
    n_series = len(series_dist_matrices)
    K = len(barycenter_dist_matrices)
    n_sup = barycenter_dist_matrices[0].shape[0]
    assert all(B.shape[0] == n_sup for B in barycenter_dist_matrices), \
        "All barycenters must have the same n_support"
    plan_dim = n_sup * n_sup
    H_max = 2 * np.log(n_sup)

    plans = np.zeros((n_series, K * plan_dim), dtype=np.float32)
    coupling = np.zeros((n_series, K), dtype=np.float32)

    # Null-relative entropy gate: compute expected entropy for uncoupled plans
    # by transporting a permuted (randomized) distance matrix to each barycenter.
    # This replaces the absolute gate (which never fires due to entropic GW).
    rng = np.random.default_rng(42)
    null_entropies = np.zeros(K)
    for k in range(K):
        perm = rng.permutation(n_sup)
        D_null = barycenter_dist_matrices[k][perm][:, perm]
        T_null = compute_transport_plan(D_null, barycenter_dist_matrices[k], probe_epsilon)
        null_entropies[k] = plan_entropy(T_null)

    for i in range(n_series):
        D_i = series_dist_matrices[i]
        primary_k = int(primary_domain_idx[i])

        for k in range(K):
            offset = k * plan_dim

            if k == primary_k:
                T = compute_transport_plan(D_i, barycenter_dist_matrices[k], epsilon)
                plans[i, offset:offset + plan_dim] = T
                coupling[i, k] = 1.0 - plan_entropy(T) / H_max
            else:
                # Probe with large epsilon
                T_probe = compute_transport_plan(
                    D_i, barycenter_dist_matrices[k], probe_epsilon)
                H_probe = plan_entropy(T_probe)

                # Null-relative gate: coupling exists if entropy is
                # significantly below the null (random structure)
                if H_probe < null_entropies[k] - 0.1:
                    # Coupling detected — compute full precision
                    T = compute_transport_plan(
                        D_i, barycenter_dist_matrices[k], epsilon)
                    plans[i, offset:offset + plan_dim] = T
                    coupling[i, k] = 1.0 - plan_entropy(T) / H_max
                else:
                    # No coupling — store the probe plan (not uniform,
                    # avoids cluster artifact) with zero coupling weight
                    plans[i, offset:offset + plan_dim] = T_probe
                    coupling[i, k] = 0.0

    return plans, coupling


def coupling_weighted_dmap(
    multi_plans: np.ndarray,
    K: int,
    n_support: int,
    coupling_signal: np.ndarray,
    n_components: int = 30,
    n_landmarks: int = 800,
    alpha: float = 1.0,
    k_bandwidth: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DMAP on coupling-weighted product Birkhoff polytope B_n^K.

    Each plan block is sqrt-embedded (Hellinger) and weighted by per-series
    coupling strength. This preserves the Riemannian product structure:
    d²(x,y) = Σ_k √(S_{x,k} · S_{y,k}) · d_H²(plan_x^k, plan_y^k).

    No centering/normalization — that would destroy the Hellinger geometry
    and break DMAP convergence (R2 review finding). Instead, coupling
    weights scale each block's contribution to the distance.

    Parameters
    ----------
    multi_plans : np.ndarray, shape (n_series, K * n_support^2)
    K : int
    n_support : int
    coupling_signal : np.ndarray, shape (n_series, K)
        Per-series coupling weights. Primary domain has S~0.14,
        coupled foreign ~0.05-0.13, uncoupled ~0.
    n_components : int
    n_landmarks : int
    alpha : float
    k_bandwidth : int

    Returns
    -------
    coords : np.ndarray, shape (n_series, n_components)
    eigenvalues : np.ndarray, shape (n_components,)
    block_weights : np.ndarray, shape (K,) — mean coupling per block
    """
    n = multi_plans.shape[0]
    plan_dim = n_support * n_support

    # Per-series coupling weights (floor at 0.01 to prevent zero blocks)
    coupling_weights = np.maximum(coupling_signal, 0.01)

    # Build weighted sqrt-Hellinger embedding preserving B_n geometry.
    # Each block stays on the positive orthant (no centering).
    # Per-series weighting: series with strong cross-domain coupling
    # have their foreign blocks upweighted.
    blocks = []
    for k in range(K):
        block = multi_plans[:, k * plan_dim:(k + 1) * plan_dim]
        sqrt_block = np.sqrt(np.maximum(block, 1e-15).astype(np.float64))
        # Per-series weight: sqrt(S_ik) scales Hellinger distances
        weighted = sqrt_block * np.sqrt(coupling_weights[:, k:k + 1])
        blocks.append(weighted)

    # Global block weights for reporting (mean per-series, informational)
    block_weights = coupling_weights.mean(axis=0)

    concat = np.hstack(blocks)

    # Standard DMAP (adaptive bandwidth, Nystrom if needed)
    n_lm = min(n_landmarks, n)
    n_comp = min(n_components, n_lm - 1)

    if n <= n_lm and n <= 5000:
        coords, evals = _transport_dmap_dense(concat, n_comp, alpha, k_bandwidth)
    else:
        lm_idx = _fps_hellinger(concat, n_lm)
        sqrt_lm = concat[lm_idx]
        coords_lm, evals_lm = _transport_dmap_dense(sqrt_lm, n_comp, alpha, k_bandwidth)
        coords = _nystrom_extend(concat, sqrt_lm, lm_idx, coords_lm, evals_lm,
                                 alpha, k_bandwidth)
        evals = evals_lm

    return coords, evals, block_weights


def extract_coupling_matrix(
    coords: np.ndarray,
    multi_plans: np.ndarray,
    K: int,
    n_support: int,
    eigenvalues: np.ndarray,
    primary_domain_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract K×K directed coupling matrix from DMAP eigenvectors.

    For each DMAP axis, measures the Spearman loading on each plan block.
    Uses |r| (not r²) so the coupling matrix is a proper eigenvalue-weighted
    cosine similarity in spectral-loading space.

    If primary_domain_idx is provided, computes DIRECTED coupling:
    C[a,b] = spectral weight of block b among series from domain a.
    This is asymmetric: C[a,b] ≠ C[b,a] because coupling is directed.

    No supervised signal — everything from spectral decomposition.

    Parameters
    ----------
    coords : np.ndarray, shape (n, n_components)
    multi_plans : np.ndarray, shape (n, K * n_support^2)
    K : int
    n_support : int
    eigenvalues : np.ndarray, shape (n_components,)
    primary_domain_idx : np.ndarray, optional, shape (n,)
        If provided, compute directed (asymmetric) coupling matrix.

    Returns
    -------
    np.ndarray, shape (K, K)
        Normalized coupling matrix. Diagonal ≈ 1.0 (within-domain).
    """
    from scipy.stats import spearmanr
    import warnings as _warn

    n_comp = min(len(eigenvalues), coords.shape[1], 20)
    plan_dim = n_support * n_support

    # DoF warning
    n_unique = K * (K + 1) // 2 if primary_domain_idx is None else K * K
    if n_comp < n_unique:
        _warn.warn(
            f"Only {n_comp} eigenvectors for {n_unique} coupling entries. "
            f"Consider increasing n_eigenvectors_coupling.", stacklevel=2)

    # Truncated SVD for per-block summaries (3 components each)
    block_summaries = []
    for k in range(K):
        block = multi_plans[:, k * plan_dim:(k + 1) * plan_dim]
        sqrt_block = np.sqrt(np.maximum(block, 1e-15).astype(np.float64))
        centered = sqrt_block - sqrt_block.mean(axis=0)
        try:
            from scipy.sparse.linalg import svds
            n_sv = min(3, min(centered.shape) - 1)
            if n_sv < 1:
                raise ValueError("too small")
            U, S, _ = svds(centered, k=n_sv)
            idx = np.argsort(S)[::-1]
            summary = U[:, idx] * S[idx]
        except Exception:
            summary = centered[:, :min(3, centered.shape[1])]
        block_summaries.append(summary)

    evals_safe = np.abs(eigenvalues[:n_comp])

    if primary_domain_idx is not None:
        # DIRECTED coupling: loadings computed per source domain
        C = np.zeros((K, K))
        for a in range(K):
            mask_a = primary_domain_idx == a
            n_a = int(np.sum(mask_a))
            if n_a < 10:
                continue
            for j in range(n_comp):
                axis = coords[mask_a, j]
                if np.std(axis) < 1e-15:
                    continue
                for b in range(K):
                    summary = block_summaries[b]
                    sub_summary = summary[mask_a]
                    max_corr = 0
                    for col in range(sub_summary.shape[1]):
                        if np.std(sub_summary[:, col]) < 1e-15:
                            continue
                        r, _ = spearmanr(axis, sub_summary[:, col])
                        max_corr = max(max_corr, abs(r))
                    C[a, b] += evals_safe[j] * max_corr  # |r| not r²
    else:
        # SYMMETRIC coupling: loadings computed over all series
        loadings = np.zeros((n_comp, K))
        for j in range(n_comp):
            axis = coords[:, j]
            if np.std(axis) < 1e-15:
                continue
            for k in range(K):
                summary = block_summaries[k]
                max_corr = 0
                for col in range(summary.shape[1]):
                    if np.std(summary[:, col]) < 1e-15:
                        continue
                    r, _ = spearmanr(axis, summary[:, col])
                    max_corr = max(max_corr, abs(r))
                loadings[j, k] = max_corr  # |r| not r² (proper cosine similarity)

        C = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                C[a, b] = np.sum(evals_safe * loadings[:, a] * loadings[:, b])

    # Normalize by diagonal
    diag = np.sqrt(np.maximum(np.diag(C), 1e-10))
    C_norm = C / (diag[:, None] * diag[None, :])

    return C_norm
