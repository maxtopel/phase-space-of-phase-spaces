"""aoa_v3.bootstrap_ci — Stationary-block bootstrap BCa CIs and DM-HLN tests.

Implements, for forecast-evaluation inference on walk-forward concatenated
errors:

  * Politis-White (2009, JASA) automatic block length for the stationary
    bootstrap (Politis-Romano 1994), via `arch.bootstrap.optimal_block_length`.
  * Stationary bootstrap (Politis-Romano 1994) replicates of a user-supplied
    statistic, via `arch.bootstrap.StationaryBootstrap`, with SeedSequence-
    pinned RNG and replicate caching to HDF5.
  * Bias-corrected-and-accelerated (BCa) 95% confidence intervals (Efron
    1987, JASA) with jackknife acceleration.
  * Diebold-Mariano (1995, JBES) test with Harvey-Leybourne-Newbold (1997,
    IJF) small-sample adjustment (HLN), two-sided, using the Student-t(T-1)
    reference distribution.
  * Benjamini-Hochberg (1995, JRSS-B) FDR adjustment across a family of
    p-values.

Design notes:
  - The stationary bootstrap is robust to ANY autocorrelation structure in
    the loss-differential series, with automatic geometric block lengths.
    This is the standard choice for forecast comparison (Politis-Romano
    1994; Sullivan-Timmermann-White 1999).
  - We bootstrap the SERIES of losses (per-observation squared errors,
    directional-accuracy indicators) and recompute the statistic on each
    replicate. This preserves the time structure.
  - BCa is the preferred CI method for ratios of statistics with possibly
    skewed sampling distributions (MSE ratios). It coincides with percentile
    CI under symmetry + bias-free statistic but has faster asymptotic
    coverage correction under skew.
  - HLN adjustment corrects the DM variance for finite sample by scaling by
    sqrt((T + 1 - 2h + h(h-1)/T) / T) and using t_{T-1} reference. At the
    h-step horizon the MA(h-1) structure of forecast errors inflates the
    naive DM variance; HLN is a standard fix.

Caching:
  Per `feedback_gw_always_cache` memory rule, every bootstrap run writes
  its replicate matrices to an HDF5 file keyed by (target, horizon,
  metric, seed, B, block_length). Re-runs with the same key read from
  cache. Cache path is `<repo>/.cache/bootstrap_ci/bootstrap_ci.h5` by
  default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from arch.bootstrap import StationaryBootstrap, optimal_block_length  # type: ignore
    _HAVE_ARCH = True
except Exception:  # pragma: no cover
    _HAVE_ARCH = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap CI and DM-HLN computations.

    Attributes
    ----------
    B : int
        Number of stationary-bootstrap replicates. Default 2000, the widely
        cited minimum for stable BCa tails at alpha=0.05.
    alpha : float
        Two-sided significance level for CIs. 0.05 for 95% coverage.
    block_length : Optional[float]
        Average block length for the stationary bootstrap. If None, computed
        via Politis-White automatic selector on the supplied loss series.
    seed : int
        Integer seed for the root SeedSequence; pinned per task at
        20260422.
    cache_path : Optional[Path]
        HDF5 cache path. If None, disables caching.
    """

    B: int = 2000
    alpha: float = 0.05
    block_length: Optional[float] = None
    seed: int = 20260422
    cache_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# Politis-White block length
# -----------------------------------------------------------------------------


def politis_white_block_length(series: np.ndarray, column: str = "stationary") -> float:
    """Return Politis-White (2009) automatic block length for the
    stationary bootstrap of the supplied 1-D series.

    Falls back to max(1, ceil(n^(1/3))) if `arch` is unavailable.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)) or len(arr) < 4:
        # Degenerate: short default
        return max(1.0, float(np.ceil(max(len(arr), 1) ** (1.0 / 3.0))))
    if _HAVE_ARCH:
        df = optimal_block_length(arr)
        # arch returns a DataFrame with columns "stationary" and "circular".
        return float(df.iloc[0][column])
    # Fallback: simple cube-root rule.
    return max(1.0, float(np.ceil(len(arr) ** (1.0 / 3.0))))


# -----------------------------------------------------------------------------
# Cache helpers (HDF5)
# -----------------------------------------------------------------------------


def _cache_key(
    target: str, horizon: int, metric: str, seed: int, B: int, block_length: float,
    extra: str = "",
) -> str:
    """Stable content-addressed key for the bootstrap replicate cache."""
    payload = f"{target}|{horizon}|{metric}|{seed}|{B}|{block_length:.6f}|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cache_read(path: Optional[Path], key: str) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if key in f:
                return np.asarray(f[key][...], dtype=np.float64)
    except Exception:
        return None
    return None


def _cache_write(path: Optional[Path], key: str, replicates: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
        with h5py.File(path, "a") as f:
            if key in f:
                del f[key]
            f.create_dataset(key, data=np.asarray(replicates, dtype=np.float64))
    except Exception:
        # Cache failure must not break the analysis.
        pass


# -----------------------------------------------------------------------------
# Stationary bootstrap with optional cache
# -----------------------------------------------------------------------------


def stationary_bootstrap_replicates(
    data: tuple[np.ndarray, ...],
    stat_fn: Callable[..., float],
    cfg: BootstrapConfig,
    cache_key_extra: str = "",
    cache_target: str = "",
    cache_horizon: int = 0,
    cache_metric: str = "",
) -> tuple[np.ndarray, float]:
    """Run B stationary-bootstrap replicates of `stat_fn` applied to the
    jointly-indexed arrays in `data`.

    Returns (replicates array of shape (B,), block_length used).

    The stationary bootstrap resamples the SAME random row indices across
    all supplied arrays, preserving joint observations.
    """
    if not _HAVE_ARCH:
        raise RuntimeError(
            "arch package required for StationaryBootstrap; install via "
            "`pip install arch`."
        )
    # Decide block length. If unset, base it on the first array in `data`.
    bl = cfg.block_length
    if bl is None:
        bl = politis_white_block_length(data[0])

    key = _cache_key(
        target=cache_target, horizon=cache_horizon, metric=cache_metric,
        seed=cfg.seed, B=cfg.B, block_length=bl, extra=cache_key_extra,
    )
    cached = _cache_read(cfg.cache_path, key)
    if cached is not None and cached.shape == (cfg.B,):
        return cached, float(bl)

    # SeedSequence-pinned generator.
    ss = np.random.SeedSequence(cfg.seed)
    child_seed = int(ss.generate_state(1)[0])
    bs = StationaryBootstrap(bl, *data, seed=child_seed)
    reps = np.empty(cfg.B, dtype=np.float64)
    for i, params in enumerate(bs.bootstrap(cfg.B)):
        pos_args = params[0]
        try:
            reps[i] = float(stat_fn(*pos_args))
        except Exception:
            reps[i] = np.nan
    _cache_write(cfg.cache_path, key, reps)
    return reps, float(bl)


# -----------------------------------------------------------------------------
# BCa interval
# -----------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro approximation.

    Adequate for bootstrap BCa endpoints; max error ~1e-8.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    try:
        from scipy.stats import norm as _sp_norm  # type: ignore
        return float(_sp_norm.ppf(p))
    except Exception:
        pass
    # Fallback (Moro 1995).
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = (-2 * np.log(1 - p)) ** 0.5
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def bca_ci(
    theta_hat: float,
    replicates: np.ndarray,
    jackknife_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, dict]:
    """Compute BCa confidence interval (Efron 1987, JASA).

    Parameters
    ----------
    theta_hat : point estimate of the statistic on the full sample.
    replicates : shape (B,), bootstrap draws of the statistic.
    jackknife_values : shape (T,), leave-one-out re-computations of the
        statistic. Used for the acceleration `a` estimate.
    alpha : two-sided level (0.05 for 95% CI).

    Returns (ci_low, ci_high, diagnostics_dict) where diagnostics_dict has
    keys `z0_hat`, `a_hat`, `alpha1`, `alpha2`, `n_finite_reps`.
    """
    reps = np.asarray(replicates, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if len(reps) < 20:
        return (np.nan, np.nan,
                {"z0_hat": np.nan, "a_hat": np.nan,
                 "alpha1": alpha / 2, "alpha2": 1 - alpha / 2,
                 "n_finite_reps": int(len(reps))})

    # Bias correction z0.
    p0 = float(np.mean(reps < theta_hat))
    # Clip to avoid p0 = 0 or 1 driving ppf to infinity.
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    z0 = _norm_ppf(p0)

    # Acceleration a via jackknife on the full-sample statistic.
    jk = np.asarray(jackknife_values, dtype=np.float64)
    jk = jk[np.isfinite(jk)]
    if len(jk) < 5:
        a = 0.0
    else:
        jk_mean = float(np.mean(jk))
        num = float(np.sum((jk_mean - jk) ** 3))
        den = 6.0 * (float(np.sum((jk_mean - jk) ** 2))) ** 1.5
        a = num / den if abs(den) > 1e-20 else 0.0

    zl = _norm_ppf(alpha / 2)
    zu = _norm_ppf(1 - alpha / 2)
    # Adjusted percentiles.
    alpha1 = _norm_cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    alpha2 = _norm_cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    # Clip to [0,1] for safety.
    alpha1 = min(max(alpha1, 0.0), 1.0)
    alpha2 = min(max(alpha2, 0.0), 1.0)
    ci_lo = float(np.quantile(reps, alpha1))
    ci_hi = float(np.quantile(reps, alpha2))
    return ci_lo, ci_hi, {
        "z0_hat": float(z0), "a_hat": float(a),
        "alpha1": float(alpha1), "alpha2": float(alpha2),
        "n_finite_reps": int(len(reps)),
    }


# -----------------------------------------------------------------------------
# Diebold-Mariano (HLN 1997 small-sample adjustment)
# -----------------------------------------------------------------------------


def dm_hln_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int,
    two_sided: bool = True,
) -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample
    adjustment.

    Null H0: E[d_t] = 0 where d_t = loss_a[t] - loss_b[t] (equal predictive
    accuracy). Under the HLN small-sample adjustment:

        DM     = mean(d) / sqrt(Var_hat(d) / T)
        DM_HLN = DM * sqrt((T + 1 - 2h + h(h-1)/T) / T)

    and the reference distribution is Student-t with T - 1 degrees of
    freedom. Var_hat uses the truncated Newey-West estimator at lag h - 1
    (the natural truncation for h-step forecast errors):

        Var_hat(d) = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k

    where gamma_k = (1/T) sum_{t} (d_t - dbar)(d_{t-k} - dbar).

    Returns dict with keys:
      T, h, mean_d, var_d, dm_stat, dm_hln_stat, p_value, df
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    T = len(d)
    h = max(1, int(horizon))
    if T < h + 2:
        return {
            "T": int(T), "h": int(h), "mean_d": float(np.mean(d)) if T else np.nan,
            "var_d": np.nan, "dm_stat": np.nan, "dm_hln_stat": np.nan,
            "p_value": np.nan, "df": max(T - 1, 1),
        }
    d_bar = float(np.mean(d))
    centered = d - d_bar
    gamma0 = float(np.mean(centered ** 2))
    var_d = gamma0
    for k in range(1, h):
        gk = float(np.mean(centered[k:] * centered[:-k]))
        var_d += 2.0 * gk
    # Bartlett-style positivity fallback: if serial-correlation correction
    # drives variance below zero, fall back to gamma_0 only.
    if var_d <= 0:
        var_d = gamma0

    if var_d <= 0:
        return {
            "T": int(T), "h": int(h), "mean_d": d_bar,
            "var_d": np.nan, "dm_stat": np.nan, "dm_hln_stat": np.nan,
            "p_value": np.nan, "df": int(T - 1),
        }
    dm = d_bar / np.sqrt(var_d / T)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln_factor

    # Student-t reference distribution.
    from math import gamma as _gamma, sqrt as _sqrt
    df = int(T - 1)
    try:
        from scipy.stats import t as _t
        if two_sided:
            p = 2.0 * float(_t.sf(abs(dm_hln), df=df))
        else:
            p = float(_t.sf(dm_hln, df=df))
    except Exception:
        # Fallback: normal approximation (acceptable for T > 30)
        if two_sided:
            p = 2.0 * (1.0 - _norm_cdf(abs(dm_hln)))
        else:
            p = 1.0 - _norm_cdf(dm_hln)
        del _gamma, _sqrt
    return {
        "T": int(T), "h": int(h), "mean_d": d_bar, "var_d": float(var_d),
        "dm_stat": float(dm), "dm_hln_stat": float(dm_hln),
        "p_value": float(p), "df": int(df),
    }


# -----------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# -----------------------------------------------------------------------------


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjusted q-values.

    Returns q such that q[i] = min_{k>=rank(p[i])} ( n / k * p_{(k)} ),
    clipped to [0, 1]. Enforces monotonicity in the usual way.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    # Unadjusted BH: q_(k) = n / k * p_(k); then monotonic cummin from right.
    raw = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity (step-up): q_(k) = min(q_(k), q_(k+1))
    for i in range(n - 2, -1, -1):
        if raw[i + 1] < raw[i]:
            raw[i] = raw[i + 1]
    raw = np.clip(raw, 0.0, 1.0)
    # Unrank.
    out = np.empty(n, dtype=np.float64)
    out[order] = raw
    return out


# -----------------------------------------------------------------------------
# Forecast-evaluation statistics
# -----------------------------------------------------------------------------


def mse_ratio_stat(num_losses: np.ndarray, den_losses: np.ndarray) -> float:
    n = np.asarray(num_losses, dtype=np.float64)
    d = np.asarray(den_losses, dtype=np.float64)
    m = np.isfinite(n) & np.isfinite(d)
    if m.sum() == 0:
        return float("nan")
    num = float(np.mean(n[m]))
    den = float(np.mean(d[m]))
    if den <= 1e-20:
        return float("nan")
    return num / den


def r2_oos_stat(yt: np.ndarray, yp: np.ndarray, rec: np.ndarray) -> float:
    """Campbell-Thompson / Welch-Goyal out-of-sample R^2.

    Note on bootstrap semantics: when this statistic is recomputed on a
    stationary-bootstrap resample, the `rec` array (recursive training-mean
    per test slot) is treated as an observation-level covariate and is
    resampled jointly with yt and yp. This is the standard convention in
    the forecast-evaluation literature (Clark-McCracken 2001 JAE). It
    permits fold-level baselines to mix across bootstrap-block boundaries;
    the resulting replicates are interpretable as R^2_OOS under the joint
    distribution of (yt, yp, rec), which is the quantity whose CI is
    reported.
    """
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(rec)
    if m.sum() == 0:
        return float("nan")
    ss_res = float(np.sum((yt[m] - yp[m]) ** 2))
    ss_bench = float(np.sum((yt[m] - rec[m]) ** 2))
    if ss_bench < 1e-20:
        return 0.0 if ss_res < 1e-20 else float("-inf")
    return 1.0 - ss_res / ss_bench


def dir_acc_stat(sign_matches: np.ndarray) -> float:
    sm = np.asarray(sign_matches, dtype=np.float64)
    if len(sm) == 0:
        return float("nan")
    return float(np.mean(sm))
