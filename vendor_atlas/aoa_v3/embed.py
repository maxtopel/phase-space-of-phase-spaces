"""
aoa_v3.embed — Takens delay-coordinate embedding.

Implements paper § 2.1:
  - Takens' theorem (Theorem 1): delay-coordinate map
      F(x) = (h(x), h(phi(x)), ..., h(phi^{d-1}(x)))
  - AMI for tau (first local minimum, Fraser-Swinney 1986)
  - FNN for d (Kennel-Brown-Abarbanel 1992; R_tol=15, A_tol=2, eta=0.01)
  - Offset rule (paper § 2.1.2): effective offset consumed is d*tau
    (when tau-differencing is applied upstream, which v3 does NOT do internally;
     caller must pass an already-stationary series)
  - Row normalization (paper § 2.1.3): per-row z-score, affine transform
    that preserves dynamical classifiers (permutation entropy etc.)

Paper defaults (binding per CLAUDE.md, no MVP):
  R_tol = 15, A_tol = 2, eta = 0.01, kNN bandwidth k = 10.

References:
  - Takens (1981) — Detecting strange attractors in turbulence
  - Sauer, Yorke, Casdagli (1991) — Embedology (fractal extension)
  - Fraser & Swinney (1986) — Average mutual information for tau
  - Kennel, Brown, Abarbanel (1992) — FNN algorithm
  - Roulston (1999); Miller (1955) — finite-sample MI bias corrections.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding hyperparameters. Paper-default values are binding per CLAUDE.md (no MVP)."""

    ami_max_lag: int = 50
    ami_bins: int = 64           # upper cap; adaptive to n (see `ami()`)
    tau_min: int = 1
    tau_max: int = 50

    fnn_max_dim: int = 20
    fnn_rtol: float = 15.0       # paper § 2.1.2 R_tol
    fnn_atol: float = 2.0        # paper § 2.1.2 A_tol
    fnn_eta: float = 0.01        # paper § 2.1.2 convergence threshold
    dim_min: int = 2
    dim_max: int = 10
    dim_default: int = 3         # used only when FNN never drops below eta

    normalize: bool = True       # paper § 2.1.3 row normalization


def _require_finite(x: np.ndarray, name: str = "series") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or Inf")
    return x


def ami(series: np.ndarray, max_lag: int = 50, n_bins: int = 64) -> np.ndarray:
    """Average mutual information for lags 0..max_lag.

    I(tau) = sum p(x_i, x_{j+tau}) log [ p(x_i, x_{j+tau}) / (p(x_i) p(x_{j+tau})) ]

    Implementation notes:
      - `np.bincount` joint histogram (faster than `np.add.at` for 2-D).
      - Bin count adapts to sample size as min(n_bins, max(8, int(sqrt(n/5))))
        (Fraser-Swinney §III.B warn against n_bins^2 >> n).
      - Miller-Madow finite-sample correction subtracts (k_nonempty - 1) / (2n).
    """
    series = _require_finite(series, "series")
    n = len(series)
    if n < 4:
        raise ValueError(f"series length {n} too short for AMI")

    max_lag = min(max_lag, n // 3)
    n_bins = max(8, min(n_bins, int(np.sqrt(max(n / 5, 1.0)))))

    lo, hi = float(series.min()), float(series.max())
    # Edge padding scaled to series magnitude (avoids collapse for large |x|).
    hi_pad = hi + max(abs(hi), 1.0) * 1e-10 + 1e-10
    edges = np.linspace(lo, hi_pad, n_bins + 1)
    binned = np.clip(np.digitize(series, edges) - 1, 0, n_bins - 1).astype(np.int64)

    out = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        if lag >= n:
            break
        x = binned[: n - lag] if lag > 0 else binned
        y = binned[lag:] if lag > 0 else binned

        n_pairs = len(x)
        flat = x * n_bins + y
        joint = np.bincount(flat, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
        joint_f = joint.astype(np.float64) / n_pairs

        px = joint_f.sum(axis=1)
        py = joint_f.sum(axis=0)
        outer = px[:, None] * py[None, :]
        mask = (joint_f > 0) & (outer > 0)
        mi = float(np.sum(joint_f[mask] * np.log(joint_f[mask] / outer[mask])))
        # Miller-Madow bias correction (Roulston 1999):
        k_nonempty = int(mask.sum())
        mi = mi - (k_nonempty - 1) / (2 * max(n_pairs, 1))
        out[lag] = mi

    return out


def select_tau(series: np.ndarray, cfg: Optional[EmbedConfig] = None) -> int:
    """Pick tau as the first local minimum of AMI; fallback to argmin of the scanned range.

    Paper § 2.1.2: "first local minimum of I(tau)". If AMI is monotone over the
    scanned range (common for OU / near-integrated series), fall back to the
    global minimum on the scanned range — conventional Fraser fallback.
    """
    cfg = cfg or EmbedConfig()
    vals = ami(series, max_lag=cfg.ami_max_lag, n_bins=cfg.ami_bins)
    # Skip lag 0 (= entropy).
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] <= vals[i + 1]:
            return max(cfg.tau_min, min(int(i), cfg.tau_max))
    # No local min → argmin over lags 1..max-1.
    if len(vals) >= 3:
        idx = int(np.argmin(vals[1:-1])) + 1
        return max(cfg.tau_min, min(idx, cfg.tau_max))
    return cfg.tau_min


def fnn(
    series: np.ndarray,
    tau: int,
    max_dim: int = 20,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> np.ndarray:
    """False nearest neighbors fraction for dimensions 1..max_dim (paper § 2.1.2 eq.).

    Vectorized inner loop. Pre-allocates the full `(n - max_dim*tau) x (max_dim+1)`
    delay matrix once and slices per dim, rather than rebuilding.

    Raises ValueError if the series has zero variance (FNN is undefined).
    """
    series = _require_finite(series, "series")
    sigma = float(series.std())
    if sigma < 1e-12:
        raise ValueError("series has zero variance; FNN is undefined")

    n = len(series)
    n_hi = n - max_dim * tau  # rows valid for the largest embedding tested
    if n_hi < 20:
        # shrink max_dim so we have at least 20 rows
        max_dim = max(1, (n - 20) // tau - 1)
        n_hi = n - max_dim * tau
        if n_hi < 20:
            raise ValueError("series too short for FNN at this tau")

    # Pre-build full delay matrix at dim = max_dim + 1.
    full_dim = max_dim + 1
    full = np.stack([series[d * tau : d * tau + n_hi] for d in range(full_dim)], axis=1)

    out = np.ones(max_dim, dtype=np.float64)
    near_tie_thresh = 1e-6 * sigma

    for dim in range(1, max_dim + 1):
        dvecs = full[:, :dim]
        tree = KDTree(dvecs)
        dists, idx = tree.query(dvecs, k=2)
        nn_idx = idx[:, 1]
        nn_dist = dists[:, 1]

        valid = (nn_idx != np.arange(n_hi)) & (nn_dist > near_tie_thresh)
        # Next coordinate (d+1) gap.
        dn = np.abs(full[np.arange(n_hi), dim] - full[nn_idx, dim])
        # Full (d+1)-D distance between the same pairs.
        full_next = full[:, : dim + 1]
        df = np.linalg.norm(full_next - full_next[nn_idx], axis=1)

        crit1 = dn / np.maximum(nn_dist, 1e-12) > rtol
        crit2 = df / sigma > atol
        false_mask = valid & (crit1 | crit2)
        n_valid = int(valid.sum())
        out[dim - 1] = (false_mask.sum() / n_valid) if n_valid > 0 else 1.0

    return out


def select_dim(series: np.ndarray, tau: int, cfg: Optional[EmbedConfig] = None) -> int:
    """Pick d as the smallest dimension where FNN fraction drops below `cfg.fnn_eta`."""
    cfg = cfg or EmbedConfig()
    fracs = fnn(series, tau, max_dim=cfg.fnn_max_dim, rtol=cfg.fnn_rtol, atol=cfg.fnn_atol)
    for idx, frac in enumerate(fracs):
        if frac < cfg.fnn_eta:
            return max(cfg.dim_min, min(idx + 1, cfg.dim_max))
    return cfg.dim_default


def make_delay_vectors(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """Construct delay vectors per paper § 2.1.2.

    dvec(t) = (x(t), x(t+tau), ..., x(t + (d-1)*tau)).

    Offset consumed from series is `dim * tau` (paper § 2.1.2 "Offset rule":
    if upstream tau-differencing is applied, it consumes one more tau of lag).
    v3 does NOT tau-difference internally; caller must pass a stationary series.
    """
    series = _require_finite(series, "series")
    n = len(series)
    offset = dim * tau
    n_vecs = n - offset
    if n_vecs <= 0:
        raise ValueError(f"series length {n} too short for dim={dim}, tau={tau} (need > {offset})")
    return np.stack([series[d * tau : d * tau + n_vecs] for d in range(dim)], axis=1)


def row_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-row z-score (paper § 2.1.3 global normalization).

    hat_dvec(t) = (dvec(t) - mu(t)) / max(sigma(t), eps)

    Rows with std < `eps` (near-constant) are set to zero explicitly rather
    than amplified by `1/eps` (which would produce numerical garbage downstream
    in GW). Affine + monotone → permutation-entropy invariance preserved on
    non-degenerate rows.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    Y = np.zeros_like(X)
    mask = (std >= eps).ravel()
    Y[mask] = (X[mask] - mu[mask]) / std[mask]
    return Y


def embed_series(
    series: np.ndarray,
    cfg: Optional[EmbedConfig] = None,
) -> tuple[np.ndarray, int, int]:
    """End-to-end Takens embedding for one series.

    Returns (delay_vectors, dim, tau). Delay vectors are row-normalized
    if `cfg.normalize` is True.
    """
    cfg = cfg or EmbedConfig()
    series = _require_finite(series, "series")
    if len(series) < 50:
        raise ValueError(f"series length {len(series)} < 50 minimum for embedding")

    tau = select_tau(series, cfg)
    dim = select_dim(series, tau, cfg)
    dvecs = make_delay_vectors(series, dim, tau)
    if cfg.normalize:
        dvecs = row_normalize(dvecs)
    return dvecs, dim, tau


# -- batch --------------------------------------------------------------------

_EmbedResult = tuple[Optional[np.ndarray], Optional[int], Optional[int]]


def _embed_worker(args: tuple) -> _EmbedResult:
    """Picklable worker for `embed_batch`.

    Catches ValueError (bad series) and numpy.linalg.LinAlgError (solver
    divergence). Other exceptions propagate — silent corruption is worse than
    a crash.
    """
    s, cfg = args
    try:
        return embed_series(s, cfg)
    except (ValueError, np.linalg.LinAlgError):
        return (None, None, None)


def embed_batch(
    series_list: list[np.ndarray],
    cfg: Optional[EmbedConfig] = None,
    n_workers: int = 1,
    strict: bool = False,
) -> list[_EmbedResult]:
    """Embed many series (optionally in parallel).

    Returns list of (dvecs, dim, tau) tuples; invalid series become
    (None, None, None) unless `strict=True`, in which case the first bad series
    raises.
    """
    cfg = cfg or EmbedConfig()

    if strict:
        return [embed_series(s, cfg) for s in series_list]

    if n_workers <= 1:
        return [_embed_worker((s, cfg)) for s in series_list]

    work = [(s, cfg) for s in series_list]
    chunksize = max(1, len(work) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_embed_worker, work, chunksize=chunksize))
