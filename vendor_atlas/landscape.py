"""
GW Attractor Landscape — Domain and Universal Landscape Objects.

Status: ACTIVE

Provides DomainLandscape and UniversalLandscape classes that wrap
the alignment pipeline and hold computed landscapes with metadata.

Domains:
- Monetary transmission (~500 series)
- Solow growth (~2000 series, PWT)
- Phillips curve (~500 series)
- Financial markets (~5000 series)
- Physical (~10000 series, climate, rivers)

Universal: all 281K series, expected d_eff ≈ 8.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from config import PipelineConfig


@dataclass
class DomainSpec:
    """Specification for a domain landscape."""
    name: str
    description: str
    series_filter: dict  # keys for filtering (e.g., {'category': 'monetary'})
    expected_d_eff: Optional[tuple] = None  # (low, high) expected range
    n_landmarks: int = 300
    color: str = "#333333"  # for visualization


# Pre-defined domain specifications
DOMAINS = {
    'monetary': DomainSpec(
        name='Monetary Transmission',
        description='Fed funds, T-bills, yields, spreads, lending, IP, CPI',
        series_filter={'category': 'monetary'},
        expected_d_eff=(3, 5),
        color='#e41a1c',
    ),
    'solow': DomainSpec(
        name='Solow Growth',
        description='PWT GDP/K/L/TFP for 183 countries',
        series_filter={'category': 'solow'},
        expected_d_eff=(2, 4),
        color='#377eb8',
    ),
    'phillips': DomainSpec(
        name='Phillips Curve',
        description='CPI variants, unemployment variants, EU HICP',
        series_filter={'category': 'phillips'},
        expected_d_eff=(2, 3),
        color='#4daf4a',
    ),
    'financial': DomainSpec(
        name='Financial Markets',
        description='Equities, rates, credit, vol',
        series_filter={'category': 'financial'},
        expected_d_eff=(4, 8),
        color='#984ea3',
    ),
    'physical': DomainSpec(
        name='Physical',
        description='Climate, rivers, NOAA',
        series_filter={'category': 'physical'},
        expected_d_eff=(3, 6),
        color='#ff7f00',
    ),
}


# Sub-domains for macro law validation
MACRO_LAWS = {
    'okun': {
        'name': "Okun's Law",
        'tier': 1,
        'description': 'Output gap <-> unemployment',
        'expected_d_eff': (1, 2),
        'series_tags': ['output_gap', 'unemployment_gap'],
    },
    'yield_curve': {
        'name': 'Yield Curve Inversion',
        'tier': 1,
        'description': 'Term spread predicts recessions',
        'expected_d_eff': (2, 3),
        'series_tags': ['yield_spread', 'recession_indicator'],
    },
    'fisher': {
        'name': 'Fisher Effect',
        'tier': 1,
        'description': 'Nominal = real + expected inflation (long-run)',
        'expected_d_eff': (1, 2),
        'series_tags': ['nominal_rate', 'real_rate', 'inflation_expectation'],
    },
    'solow_ss': {
        'name': 'Solow Steady-State',
        'tier': 1,
        'description': 'Y-K-L production function geometry',
        'expected_d_eff': (2, 4),
        'series_tags': ['gdp', 'capital', 'labor', 'tfp'],
    },
    'phillips': {
        'name': 'Phillips Curve',
        'tier': 2,
        'description': 'Inflation <-> unemployment (flattened post-2000)',
        'expected_d_eff': (2, 3),
        'series_tags': ['inflation', 'unemployment'],
    },
    'monetary_transmission': {
        'name': 'Monetary Transmission',
        'tier': 2,
        'description': '7-step directed flow: policy -> real economy',
        'expected_d_eff': (3, 5),
        'series_tags': ['fed_funds', 'tbill', 'treasury_yield', 'corporate_spread',
                        'lending', 'ip', 'cpi'],
    },
    'credit_cycle': {
        'name': 'Credit Cycle',
        'tier': 2,
        'description': 'Credit-assets-conditions-real economy loop',
        'expected_d_eff': (3, 5),
        'series_tags': ['credit_growth', 'asset_prices', 'financial_conditions', 'gdp_growth'],
    },
    'ppp': {
        'name': 'Purchasing Power Parity',
        'tier': 3,
        'description': 'Long-run price level convergence across countries',
        'expected_d_eff': (1, 3),
        'series_tags': ['exchange_rate', 'price_level'],
    },
    'taylor_rule': {
        'name': 'Taylor Rule',
        'tier': 3,
        'description': 'Normative: policy rate should follow f(inflation gap, output gap). Descriptive fit varies by era/central bank.',
        'expected_d_eff': (2, 3),
        'series_tags': ['policy_rate', 'inflation_gap', 'output_gap'],
    },
    'beveridge': {
        'name': 'Beveridge Curve',
        'tier': 3,
        'description': 'Vacancies <-> unemployment (shifts structurally)',
        'expected_d_eff': (1, 2),
        'series_tags': ['vacancies', 'unemployment'],
    },
}


@dataclass
class LandscapeResult:
    """Result of a landscape construction."""
    label: str
    n_series: int
    d_eff: Optional[float] = None
    d_eff_methods: Optional[Dict[str, float]] = None  # TWO-NN, D2, LB-MLE, PR
    barycenter: Optional[np.ndarray] = None
    bary_mds_coords: Optional[np.ndarray] = None
    mds_dim: Optional[int] = None
    mds_stress: Optional[float] = None
    centroids: Optional[np.ndarray] = None
    epsilon: Optional[float] = None
    landmark_indices: Optional[np.ndarray] = None


def estimate_d_eff(centroids: np.ndarray, dmap_evals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Estimate effective dimensionality of the AoA manifold.

    Methods:
    1. TWO-NN (Facco et al. 2017) — local, distance-based
    2. D2 (correlation dimension) — fractal, distance-based
    3. DMAP eigenspectrum — spectral, nonlinear (PREFERRED for transport plans)

    Dropped: PCA participation ratio — linear modes are meaningless on the
    Birkhoff polytope. A linear combination of transport plans is not a valid
    transport plan; PCA modes don't correspond to dynamical parameters.

    Parameters
    ----------
    centroids : np.ndarray, shape (n, d)
        AoA coordinates (DMAP diffusion coordinates or transport plans).
    dmap_evals : np.ndarray, optional
        DMAP eigenvalues. If provided, used for spectral dimension estimate.

    Returns
    -------
    dict mapping method name to d_eff estimate.
    """
    results = {}

    # === LANDSCAPE SIGNATURE (Phase 1, post R02/R04/R12/R13/R14 panel) ===
    # Single point estimators of d_eff are unreliable at our sample sizes.
    # We compute a "landscape signature" of complementary metrics that are
    # each robust at n=200-500 in 10-D Hellinger DMAP space.

    # 1. Heat kernel spectral dimension (R13 primary)
    #    Robust to N_DMAP_COMP because it's a log-derivative
    if dmap_evals is not None and len(dmap_evals) >= 3:
        hk = heat_kernel_spectral_dim(dmap_evals)
        results['heat_kernel_d_s'] = hk['d_s_plateau']
        results['heat_kernel_d_s_min'] = hk['d_s_min']
        results['heat_kernel_d_s_max'] = hk['d_s_max']

        # 2. Spectral entropy (single scalar, fully robust to N_DMAP_COMP)
        results['spectral_entropy'] = spectral_entropy_dim(dmap_evals)

    # 3. Levina-Bickel MLE (R14: better than TWO-NN at small n, no scaling region)
    if len(centroids) >= 25:
        results['levina_bickel'] = _levina_bickel(centroids)

    # === LEGACY POINT ESTIMATORS (kept for triangulation, NOT primary) ===

    # D2 correlation dimension — bimodal at our n, kept as cross-check
    from descriptors import correlation_dimension
    d2 = correlation_dimension(centroids, max_points=min(2000, len(centroids)))
    if d2 is not None and not np.isnan(d2):
        results['D2'] = d2

    # TWO-NN — biased by local density per R14, kept as density indicator
    results['two_nn'] = _two_nn(centroids)

    # DMAP eigenspectrum (knee detection) — fragile on short arrays
    if dmap_evals is not None and len(dmap_evals) >= 3:
        from transport import dmap_intrinsic_dimension
        dim_est = dmap_intrinsic_dimension(dmap_evals, method='both')
        if 'd_ratio' in dim_est:
            results['dmap_ratio'] = dim_est['d_ratio']
        results['dmap_profile'] = dim_est.get('d_eff')

    # Primary = heat kernel d_s if available, else Levina-Bickel
    if not np.isnan(results.get('heat_kernel_d_s', float('nan'))):
        results['primary'] = results['heat_kernel_d_s']
        results['primary_estimator'] = 'heat_kernel_d_s'
    elif not np.isnan(results.get('levina_bickel', float('nan'))):
        results['primary'] = results['levina_bickel']
        results['primary_estimator'] = 'levina_bickel'
    else:
        results['primary'] = float('nan')
        results['primary_estimator'] = 'none'

    return results


def _two_nn(X: np.ndarray, k: int = 2) -> float:
    """
    TWO-NN intrinsic dimension estimator (Facco et al. 2017).

    Most robust method — no scaling region needed.
    """
    from scipy.spatial import KDTree

    n = X.shape[0]
    if n < 10:
        return np.nan

    tree = KDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    # Ratio of 2nd to 1st NN distance
    r1 = dists[:, 1]
    r2 = dists[:, 2]

    valid = (r1 > 1e-10)
    if np.sum(valid) < 10:
        return np.nan

    mu = r2[valid] / r1[valid]
    mu_sorted = np.sort(mu)
    n_valid = len(mu_sorted)

    # Empirical CDF
    F = np.arange(1, n_valid + 1) / n_valid

    # d = -1 / mean(log(1 - F)) ... but more robustly:
    # d = n / sum(log(mu_i))
    d_hat = n_valid / np.sum(np.log(mu_sorted))

    return d_hat


def _participation_ratio(X: np.ndarray) -> float:
    """
    Participation ratio: PR = (Σλ_i)² / Σ(λ_i²).

    Measures effective number of PCA dimensions.
    """
    centered = X - X.mean(axis=0)
    cov = np.cov(centered.T)
    evals = np.linalg.eigvalsh(cov)
    evals = evals[evals > 1e-10]

    if len(evals) == 0:
        return 0.0

    return (np.sum(evals) ** 2) / np.sum(evals ** 2)


def _levina_bickel(X: np.ndarray, k1: int = 5, k2: int = 20) -> float:
    """
    Levina-Bickel MLE for local intrinsic dimension, averaged over points.
    """
    from scipy.spatial import KDTree

    n = X.shape[0]
    k2 = min(k2, n - 1)

    tree = KDTree(X)
    dists, _ = tree.query(X, k=k2 + 1)

    # Local dimension for each point
    dims = []
    for i in range(n):
        d_k = dists[i, k2]
        if d_k < 1e-10:
            continue
        log_ratios = np.log(d_k / np.maximum(dists[i, k1:k2], 1e-10))
        if len(log_ratios) > 0 and np.mean(log_ratios) > 1e-10:
            dims.append(1.0 / np.mean(log_ratios))

    if len(dims) == 0:
        return np.nan

    return np.mean(dims)


def heat_kernel_spectral_dim(evals: np.ndarray, n_t: int = 80) -> dict:
    """
    Heat kernel spectral dimension d_s(t) = -2 t Z'(t) / Z(t).

    Z(t) = sum_k exp(-t * lambda_k)

    R13's recommendation: this is automatically robust to N_DMAP_COMP
    because the LOG-DERIVATIVE of the trace scales out the absolute count
    of eigenvalues. On a smooth d-dim manifold the function reaches a
    plateau d_s(t) ≈ d for some t-range; that plateau is the intrinsic
    dimension.

    Adaptive t-range (post-debugging): the relevant t scale is set by the
    eigenvalue magnitudes. We pick t_min = 0.1 / max(λ) (small enough that
    no mode is yet filtered) and t_max = 50 / min(λ_pos) (large enough that
    most modes are damped). Negative eigenvalues from row-norm pathology
    are filtered out.
    """
    evals = np.asarray(evals, dtype=np.float64)
    evals = evals[np.isfinite(evals) & (evals > 0)]
    if len(evals) < 3:
        return {
            'd_s_curve': np.array([]),
            't_grid': np.array([]),
            'd_s_plateau': float('nan'),
            'd_s_min': float('nan'),
            'd_s_max': float('nan'),
            'plateau_t_range': (float('nan'), float('nan')),
        }

    lam_max = float(evals.max())
    lam_min = float(evals.min())
    if lam_max <= 0 or lam_min <= 0:
        return {
            'd_s_curve': np.array([]),
            't_grid': np.array([]),
            'd_s_plateau': float('nan'),
            'd_s_min': float('nan'),
            'd_s_max': float('nan'),
            'plateau_t_range': (float('nan'), float('nan')),
        }
    t_min = 0.1 / lam_max
    t_max = 50.0 / lam_min
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    # vectorized: Z[i] = sum_k exp(-t_grid[i] * evals[k])
    Z = np.exp(-np.outer(t_grid, evals)).sum(axis=1)
    log_Z = np.log(np.maximum(Z, 1e-300))
    log_t = np.log(t_grid)
    # Centered finite difference: d log Z / d log t
    dlogZ_dlogt = np.gradient(log_Z, log_t)
    d_s = -2.0 * dlogZ_dlogt

    # Plateau detection (RELAXED v2): find the segment with smallest relative
    # variation. Always returns something — the "plateau" is the flattest
    # region available, even if it's not perfectly flat. The d_s_min/d_s_max
    # tell the user how flat it actually is.
    window = max(5, n_t // 10)
    best_start, best_len, best_var = 0, window, float('inf')
    for i in range(len(d_s) - window):
        seg = d_s[i:i + window]
        if not np.all(np.isfinite(seg)) or seg.mean() <= 0:
            continue
        rel_var = seg.std() / abs(seg.mean())
        if rel_var < best_var:
            best_var = rel_var
            best_start = i
            # Try extending rightward
            j = i + window
            while j < len(d_s) and np.isfinite(d_s[j]):
                seg2 = d_s[i:j + 1]
                rv = seg2.std() / abs(seg2.mean())
                if rv < min(0.4, rel_var * 1.5):
                    j += 1
                else:
                    break
            best_len = j - i
    plateau_seg = d_s[best_start:best_start + best_len]
    if not np.all(np.isfinite(plateau_seg)) or len(plateau_seg) < 2:
        return {
            'd_s_curve': d_s,
            't_grid': t_grid,
            'd_s_plateau': float('nan'),
            'd_s_min': float('nan'),
            'd_s_max': float('nan'),
            'plateau_t_range': (float('nan'), float('nan')),
        }
    return {
        'd_s_curve': d_s,
        't_grid': t_grid,
        'd_s_plateau': float(plateau_seg.mean()),
        'd_s_min': float(plateau_seg.min()),
        'd_s_max': float(plateau_seg.max()),
        'plateau_t_range': (float(t_grid[best_start]),
                            float(t_grid[best_start + best_len - 1])),
    }


def spectral_entropy_dim(evals: np.ndarray) -> float:
    """
    Spectral entropy of normalized eigenvalues.

    S = -sum_k p_k log(p_k)  where  p_k = lambda_k / sum_j lambda_j

    Single scalar in [0, log(n)]. Normalized to [0, 1] by dividing by log(n).
    Robust to N_DMAP_COMP because the normalization is intrinsic.

    High S → uniform spectrum → high-dim landscape
    Low S → concentrated spectrum → low-dim landscape
    """
    evals = np.asarray(evals, dtype=np.float64)
    evals = evals[np.isfinite(evals) & (evals > 0)]
    if len(evals) < 2:
        return float('nan')
    p = evals / evals.sum()
    H = -np.sum(p * np.log(np.maximum(p, 1e-300)))
    return float(H / np.log(len(evals)))
