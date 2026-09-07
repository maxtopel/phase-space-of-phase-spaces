"""
aoa_v3.predict — AoA-based macro forecasting (paper § Prediction, plan.tex § 13).

Two levels implemented (Level 3 deferred to Paper 2 per plan.tex scope):

  Level 2 (primary): Given a target series nu and a set of leading-series
    attractors {m_i}, build nu's attractor, compute T_{i->nu} for each leading
    series, form sensitivity-ranked features, run ridge regression with
    walk-forward CV.
  Level 1 (fallback): Same feature construction but against pre-built macro
    barycenters B^*_j instead of hand-picked leading series.

Honest Baseline Protocol (plan.tex § Honest Baseline Protocol):
  The predict pipeline evaluates every target against the full 8-item
  benchmark ladder:
    (i)   AR(p*) with BIC-selected p
    (ii)  AO-RW (Phillips) / RW (Okun)
    (iii) Linear ridge on raw input panel  <- PRIMARY denominator
    (iv)  PCA(k) + ridge on raw panel (k matched to AoA feature count)
    (v)   Random Forest on raw panel
    (vi)  Feature-count-matched Gaussian placebo
    (vii) Strong-shrinkage floor (ridge lambda large enough coefficients -> 0)
    (viii) Okun-GDP structural benchmark (Okun target only; caller passes GDP)

Reporting: MSE ratio vs (iii) primary + vs (i); out-of-sample R^2; directional
accuracy; forecast bias; CV-selected lambda; placebo R^2; strong-shrinkage R^2.

CV protocol (plan.tex):
  - Expanding window; initial train max(15y, 40%); re-estimate annually
    (monthly data) / every 4Q (quarterly data); h-1 embargo.
  - Inner 5-fold time-series CV for lambda selection.

No DM-HLN / IAAFT / significance theatre in this module (plan.tex Design
Decision 6); structural-IAAFT lives in the structure scripts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


LagSpec = Literal["ar_p_star", "rw", "ao_rw"]
BenchmarkName = Literal[
    "ar_p_star",
    "ao_rw",
    "rw",
    "linear_ridge_raw",
    "pca_ridge",
    "random_forest_raw",
    "placebo",
    "strong_shrinkage",
    "okun_gdp",
]


# ----------------------------------------------------- CV protocol config


@dataclass(frozen=True)
class CVConfig:
    """Walk-forward CV settings (plan.tex § CV Protocol)."""

    initial_train_frac: float = 0.40    # max(15y, 40%) at caller; we enforce 0.40 floor
    initial_train_min: int = 60         # 5y monthly or 15y quarterly
    re_estimate_every: int = 12         # monthly: every year; quarterly: pass 4
    embargo: int = 0                    # defaults to h-1 at runtime
    inner_cv_folds: int = 5


@dataclass(frozen=True)
class PredictConfig:
    """Prediction pipeline hyperparameters. Paper-binding per CLAUDE.md no-MVP."""

    horizon: int = 6                    # steps ahead
    n_aoa_features: int = 6             # top-k sensitivity-ranked AoA features
    data_freq: Literal["M", "Q"] = "M"  # monthly or quarterly (sets AO-RW window)
    # Grade Fix #4: grid spans 13 decades; adaptive scaling per fold via
    # mean feature variance in _cv_select_lambda_with_flag. Previous 7-point
    # unit-scaled grid pegged at boundary on tiny AoA features (Phillips ~1e-4).
    ridge_lambda_grid: tuple[float, ...] = (
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6
    )
    ridge_lambda_adaptive: bool = True   # scale grid by mean feature variance per fold
    strong_shrinkage_lambda: float = 1e8  # floor case: beta ~ 0
    random_forest_trees: int = 300
    random_forest_min_leaf: int = 5
    random_forest_seed: int = 0           # R16.4: separate from placebo_seed
    placebo_seed: int = 42
    cv: CVConfig = CVConfig()


# --------------------------------------------------- walk-forward iterator


def walk_forward_folds(
    n: int,
    horizon: int,
    initial_train: int,
    re_estimate_every: int,
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for expanding-window walk-forward CV.

    At fold k:
      train = [0 : initial_train + k*re_estimate_every]
      test  = [train_end + embargo + horizon - 1 : train_end + embargo + horizon - 1 + re_estimate_every]
    Bounded by array length.
    """
    if initial_train < 10:
        raise ValueError(f"initial_train={initial_train} too small")
    out: list[tuple[np.ndarray, np.ndarray]] = []
    k = 0
    while True:
        train_end = initial_train + k * re_estimate_every
        if train_end >= n - horizon - embargo:
            break
        test_start = train_end + embargo + horizon - 1
        test_end = min(test_start + re_estimate_every, n)
        if test_start >= test_end:
            break
        out.append((np.arange(train_end), np.arange(test_start, test_end)))
        k += 1
    return out


# --------------------------------------------- metrics (plan.tex § Honest)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(err * err))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classical R^2 = 1 - SS_res / SS_tot(test-window-mean).

    WARNING — time-series gotcha. SS_tot uses `y_true.mean()` which is the
    test-window mean, implicitly giving the naive-mean baseline oracle
    knowledge of the future test-window level. On trending or regime-shifting
    test windows this makes the denominator small and produces spuriously
    large-negative R^2 for any model whose output is anchored to the
    training-window mean. For forecast evaluation, prefer `r2_oos_vs_train_mean`
    (Campbell-Thompson R^2_OOS) which uses the recursive historical training
    mean as the benchmark.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0 if ss_res < 1e-12 else -np.inf
    return 1.0 - ss_res / ss_tot


def r2_oos_vs_train_mean(
    y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float,
) -> float:
    """Campbell-Thompson / Welch-Goyal out-of-sample R^2.

    R^2_{OOS} = 1 - MSE(model) / MSE(train-mean baseline)

    where the baseline forecast at every test point is the training-window
    mean (known at train time, not data-snooped from y_test). This is the
    standard OOS metric in predictive-regression literature; it penalizes
    a constant forecast at the WRONG level rather than rewarding the model
    for having been close to the naive mean's ex-post value.

    When multiple folds exist with different training ends, caller can pass
    a per-fold recursive historical mean via `_r2_oos_piecewise` below.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_bench = float(np.sum((y_true - y_train_mean) ** 2))
    if ss_bench < 1e-12:
        return 0.0 if ss_res < 1e-12 else -np.inf
    return 1.0 - ss_res / ss_bench


def r2_oos_recursive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recursive_train_means: np.ndarray,
) -> float:
    """R^2_OOS using a per-test-point recursive historical mean.

    `recursive_train_means[i]` = mean of all y observed strictly before
    test point i. This matches the Welch-Goyal convention exactly: at each
    test slot the benchmark is the mean known at that point in time.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    bench = np.asarray(recursive_train_means, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_bench = float(np.sum((y_true - bench) ** 2))
    if ss_bench < 1e-12:
        return 0.0 if ss_res < 1e-12 else -np.inf
    return 1.0 - ss_res / ss_bench


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of sign matches on first differences (for non-differenced series,
    compare sign of predicted vs realized deviation from prior value).

    For level series it is more meaningful to diff first; caller passes the
    series to diff. Here we assume y_true and y_pred are already the
    h-step-ahead changes.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def clark_west_stat(
    y_true: np.ndarray,
    y_pred_small: np.ndarray,
    y_pred_large: np.ndarray,
    horizon: int = 1,
) -> tuple[float, float]:
    """Clark-West (2007 J.Econometrics) MSPE-adjusted statistic for nested
    model encompassing tests, with Newey-West HAC variance.

    f_t = (y_t - ŷ_small,t)² − [(y_t - ŷ_large,t)² − (ŷ_small,t - ŷ_large,t)²]
    z = sqrt(T) · mean(f) / SE_NW(f)  ~  N(0, 1) under H0: MSE_large = MSE_small
    alternative H_A: MSE_large < MSE_small (larger model adds incremental content).

    SE is Newey-West (1987) HAC with Bartlett kernel at bandwidth
    max(2(h-1), 1). Overlapping h-step forecast residuals have MA(h-1)
    dependence by construction under H0; bandwidth ≥ h-1 captures all
    non-zero autocovariances, and 2(h-1) provides slack against misspecified
    persistence per Hansen (2009 JBES). Pre-2026-04-24 implementation used
    IID std; see grade Round-4 methodology finding.

    Returns (z-statistic, one-sided p-value). Positive z + small p means the
    larger (nesting) model has genuinely lower MSPE once its extra-parameter
    estimation-error penalty is accounted for.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_pred_small, dtype=np.float64)
    yl = np.asarray(y_pred_large, dtype=np.float64)
    m = np.isfinite(ys) & np.isfinite(yl) & np.isfinite(yt)
    yt, ys, yl = yt[m], ys[m], yl[m]
    T = len(yt)
    if T < 5:
        return float("nan"), float("nan")
    f = (yt - ys) ** 2 - ((yt - yl) ** 2 - (ys - yl) ** 2)
    f_mean = float(np.mean(f))
    # Newey-West HAC variance with Bartlett kernel, bandwidth max(2(h-1), 1).
    L = max(2 * (horizon - 1), 1)
    fc = f - f_mean
    gamma0 = float(np.mean(fc * fc))
    var_nw = gamma0
    for k in range(1, min(L, T - 1) + 1):
        w = 1.0 - k / (L + 1.0)  # Bartlett
        cov_k = float(np.mean(fc[k:] * fc[:-k]))
        var_nw += 2.0 * w * cov_k
    if var_nw <= 1e-20:
        return float("nan"), float("nan")
    f_std_nw = np.sqrt(var_nw)
    z = np.sqrt(T) * f_mean / f_std_nw
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return float(z), float(max(p, 0.0))


def _adf_pvalue_safe(y: np.ndarray) -> float:
    """Augmented Dickey-Fuller p-value on the target series (h-step differences).

    Small p ⇒ series is stationary ⇒ ridge intercept + centered features are
    appropriately specified. Large p ⇒ non-stationary target; ridge with
    intercept is biased and R² may be misleading. Used as a diagnostic
    (Criterion 7 level-4), not as a gate that blocks execution.

    Returns NaN on any failure (missing statsmodels, too few points, etc.).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        y = np.asarray(y, dtype=np.float64).ravel()
        y = y[np.isfinite(y)]
        if len(y) < 20:
            return float("nan")
        stat, p, *_ = adfuller(y, autolag="AIC")
        return float(p)
    except Exception:
        return float("nan")


def dir_acc_binomial_p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-sided binomial p-value for directional accuracy vs 0.5.

    Grade Fix #5: Okun exhibited dir_acc = 0.23 — symptomatic of systematic
    sign inversion (e.g. feature sign error, target-eigenvalue sign flip).
    A binomial two-sided test tells us whether the deviation from chance
    is statistically real so we can flag inversion versus noise.
    Uses scipy.stats.binomtest when available; falls back to normal
    approximation when scipy is unavailable or n is large.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return 1.0
    matches = int(np.sum(np.sign(yt) == np.sign(yp)))
    try:
        from scipy.stats import binomtest
        return float(binomtest(matches, n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        p_hat = matches / n
        z = (p_hat - 0.5) / np.sqrt(0.25 / n)
        from math import erf, sqrt
        return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))))


# -------------------------------------------------- ridge with CV-lambda


def _ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Ordinary ridge closed-form with scalar intercept.

    (X'X + lam I)^{-1} X'y for the centered problem; intercept = y_mean - beta'x_mean.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    n_feat = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(n_feat)
    b = Xc.T @ yc
    beta = np.linalg.solve(A, b)
    intercept = y_mean - float(beta @ x_mean)
    return beta, intercept


def _ridge_predict(X: np.ndarray, beta: np.ndarray, intercept: float) -> np.ndarray:
    return X @ beta + intercept


def _cv_select_lambda_with_flag(
    X: np.ndarray,
    y: np.ndarray,
    grid: tuple[float, ...],
    n_folds: int,
    embargo: int = 0,
    adaptive: bool = True,
) -> tuple[float, bool]:
    """Inner time-series CV for lambda selection (expanding window).

    Applies `embargo` inside the inner CV so autocorrelated state doesn't
    leak across the seam (R02.3). Returns (lambda, used_fallback_flag):
    if n_train is too small for a meaningful CV, returns the grid median
    and flags the fallback for caller-side audit (R02.1).

    Grade Fix #4: when `adaptive=True`, scale the grid by the mean feature
    variance so small-magnitude AoA features (O(1e-4) or smaller) get
    proportionally small lambda candidates. Scale factor ~ mean(var(X)):
    ridge penalty lambda * ||beta||^2 matches data term when lambda scales
    with ||X||^2. Fall back to unscaled grid when scale is near-zero.
    """
    n = len(y)
    if n < n_folds + 5:
        return float(grid[len(grid) // 2]), True
    scaled_grid: tuple[float, ...]
    if adaptive:
        var_mean = float(np.mean(np.var(X, axis=0)))
        scale = var_mean if var_mean > 1e-20 else 1.0
        scaled_grid = tuple(float(lam * scale) for lam in grid)
    else:
        scaled_grid = tuple(float(lam) for lam in grid)
    fold_size = n // (n_folds + 1)
    best_lam = scaled_grid[0]
    best_mse = np.inf
    for lam in scaled_grid:
        fold_mses = []
        for k in range(n_folds):
            train_end = fold_size * (k + 1)
            test_start = train_end + embargo
            test_end = min(test_start + fold_size, n)
            if train_end < 5 or test_end <= test_start:
                continue
            Xtr, ytr = X[:train_end], y[:train_end]
            Xte, yte = X[test_start:test_end], y[test_start:test_end]
            beta, icept = _ridge_fit(Xtr, ytr, lam)
            fold_mses.append(mse(yte, _ridge_predict(Xte, beta, icept)))
        if not fold_mses:
            continue
        avg = float(np.mean(fold_mses))
        if avg < best_mse:
            best_mse = avg
            best_lam = lam
    return float(best_lam), False


def _cv_select_lambda(
    X: np.ndarray,
    y: np.ndarray,
    grid: tuple[float, ...],
    n_folds: int,
    embargo: int = 0,
) -> float:
    """Backwards-compatible wrapper returning only the selected lambda."""
    lam, _ = _cv_select_lambda_with_flag(X, y, grid, n_folds, embargo=embargo)
    return lam


def ridge_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Optional[PredictConfig] = None,
    lam: Optional[float] = None,
    sign_gauge_flip: bool = False,
) -> dict:
    """Walk-forward ridge evaluation with optional CV-λ per fold and per-fold
    diagnostics for honest audit of R²<0 outcomes.

    When ``sign_gauge_flip=True`` each fold's β is sign-normalized to the
    training fold's directional accuracy: DMAP eigenvectors are defined up to
    sign (gauge freedom), so if LAPACK hands us ``−ψ^i_k`` vs. ``+ψ^j_l`` the
    ridge-fit β can land at an MSE-optimal but sign-inverted local point.
    Flipping β on each fold by the sign of ``dir_acc_train − 0.5`` (at binomial
    p < 0.05) is a gauge-fix, not new algorithm — it picks the sign convention
    consistent with the training data, no test leakage.

    Returns a dict of pooled metrics plus ``diagnostics`` with per-fold λ,
    ridge coefficient stats, feature magnitude percentiles, fold R²_OOS,
    condition numbers, and flip counter (for grading Criteria 8, 9, 11).
    """
    cfg = cfg or PredictConfig()
    n = len(y)
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else cfg.horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(
        n, cfg.horizon, initial_train, cfg.cv.re_estimate_every, embargo
    )
    if not folds:
        raise ValueError(
            f"insufficient data for walk-forward: n={n}, horizon={cfg.horizon}, "
            f"initial_train={initial_train}"
        )
    y_true_all, y_pred_all, lams_used, test_idx_all = [], [], [], []
    recursive_means_all = []
    n_fallback = 0
    n_boundary_pegged = 0
    n_sign_flipped = 0
    # Per-fold diagnostic traces.
    fold_lambdas: list[float] = []
    fold_r2_oos: list[float] = []
    fold_coef_max_abs: list[float] = []
    fold_coef_mean_abs: list[float] = []
    fold_cond_number: list[float] = []
    fold_feat_p50: list[float] = []
    fold_feat_p90: list[float] = []
    fold_n_train: list[int] = []
    # Grid boundaries for the boundary-peg counter (fixed grid; per-fold
    # adaptive scaling multiplies these, so we compare on the unscaled ratio).
    grid_min = float(min(cfg.ridge_lambda_grid))
    grid_max = float(max(cfg.ridge_lambda_grid))
    for tr_idx, te_idx in folds:
        test_idx_all.append(te_idx)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        if lam is not None:
            fold_lam = lam
            used_fallback = False
        else:
            fold_lam, used_fallback = _cv_select_lambda_with_flag(
                Xtr, ytr, cfg.ridge_lambda_grid, cfg.cv.inner_cv_folds,
                embargo=embargo, adaptive=cfg.ridge_lambda_adaptive,
            )
        if used_fallback:
            n_fallback += 1
        # Boundary-pegging: check if selected λ is within 10x of grid edge
        # (accounting for adaptive variance scaling).
        if cfg.ridge_lambda_adaptive:
            var_mean = float(np.mean(np.var(Xtr, axis=0)))
            scale = var_mean if var_mean > 1e-20 else 1.0
            unscaled = fold_lam / scale if scale > 0 else fold_lam
        else:
            unscaled = fold_lam
        if unscaled <= grid_min * 10.0 or unscaled >= grid_max / 10.0:
            n_boundary_pegged += 1
        beta, icept = _ridge_fit(Xtr, ytr, fold_lam)
        # Sign-gauge diagnostic: count folds where training dir_acc is
        # significantly below chance (binomial p < 0.05). This is a reported
        # diagnostic — we do NOT alter β. DMAP eigenvectors are ± arbitrary
        # by gauge freedom; empirically, flipping β on sign-flagged folds did
        # not generalize on held-out folds (the sign relationship is not
        # stable across the fold horizon, which is itself a finding). Flip-
        # if-requested via sign_gauge_flip=True; default False for honest
        # reporting.
        if len(Xtr) >= 20:
            yp_train = _ridge_predict(Xtr, beta, icept)
            if np.std(yp_train) > 1e-12 and np.std(ytr) > 1e-12:
                da_train = float(np.mean(np.sign(ytr) == np.sign(yp_train)))
                p_train = dir_acc_binomial_p(ytr, yp_train)
                if da_train < 0.5 and p_train < 0.05:
                    n_sign_flipped += 1
                    if sign_gauge_flip:
                        beta = -beta
                        icept = float(2.0 * ytr.mean() - icept)
        yp_te = _ridge_predict(Xte, beta, icept)
        y_pred_all.append(yp_te)
        y_true_all.append(yte)
        # Campbell-Thompson recursive mean: at each test point t, use the
        # mean of y[:t] (expanding window, per-test-point), NOT the
        # fold-train-window mean (which coincides with ridge's intercept
        # and produces exact R²_OOS = 0 on any β→0 collapse — R14 finding).
        rec_mean_per_test = np.array(
            [float(y[:int(ti)].mean()) if int(ti) > 0 else float(ytr.mean())
             for ti in te_idx],
            dtype=np.float64,
        )
        recursive_means_all.append(rec_mean_per_test)
        lams_used.append(fold_lam)
        # Per-fold diagnostics.
        fold_lambdas.append(float(fold_lam))
        fold_n_train.append(int(len(tr_idx)))
        fold_r2_oos.append(
            r2_oos_recursive(yte, yp_te, rec_mean_per_test)
        )
        fold_coef_max_abs.append(float(np.max(np.abs(beta))) if beta.size else 0.0)
        fold_coef_mean_abs.append(float(np.mean(np.abs(beta))) if beta.size else 0.0)
        # Condition number of (X_tr^T X_tr): indicator of ridge conditioning.
        try:
            Xc = Xtr - Xtr.mean(axis=0)
            _, s, _ = np.linalg.svd(Xc, full_matrices=False)
            fold_cond_number.append(
                float(s[0] / s[-1]) if s[-1] > 1e-20 else float("inf")
            )
        except Exception:
            fold_cond_number.append(float("nan"))
        abs_X = np.abs(Xtr) if Xtr.size else np.array([0.0])
        fold_feat_p50.append(float(np.percentile(abs_X, 50)))
        fold_feat_p90.append(float(np.percentile(abs_X, 90)))
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    rec_means = np.concatenate(recursive_means_all)
    test_idx = np.concatenate(test_idx_all) if test_idx_all else np.array([], dtype=int)
    return {
        "mse": mse(yt, yp),
        "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec_means),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp),
        "cv_lambda_median": float(np.median(lams_used)),
        "n_fallback_folds": int(n_fallback),
        "n_boundary_pegged_folds": int(n_boundary_pegged),
        "n_sign_flipped_folds": int(n_sign_flipped),
        "n_folds": len(folds),
        "n_test": len(yt),
        "y_true": yt,
        "y_pred": yp,
        "recursive_train_means": rec_means,
        "test_indices": test_idx,
        "diagnostics": {
            "fold_lambdas": fold_lambdas,
            "fold_r2_oos": fold_r2_oos,
            "fold_coef_max_abs": fold_coef_max_abs,
            "fold_coef_mean_abs": fold_coef_mean_abs,
            "fold_cond_number": fold_cond_number,
            "fold_feat_p50": fold_feat_p50,
            "fold_feat_p90": fold_feat_p90,
            "fold_n_train": fold_n_train,
        },
    }


# ---------------------------------------- benchmark ladder implementations


def ar_p_star_forecast(
    y: np.ndarray,
    horizon: int,
    p_grid: tuple[int, ...] = tuple(range(1, 13)),
    cfg: Optional[PredictConfig] = None,
) -> dict:
    """AR(p*) with BIC-selected p at each fold. Standard univariate macro baseline."""
    cfg = cfg or PredictConfig()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(n, horizon, initial_train, cfg.cv.re_estimate_every, embargo)
    if not folds:
        raise ValueError("insufficient data for AR(p*) walk-forward")
    yt_all, yp_all, p_used, rec_all = [], [], [], []
    for tr_idx, te_idx in folds:
        ytr = y[tr_idx]
        p_star = _bic_select_p(ytr, p_grid)
        yt_all.append(y[te_idx])
        yp_all.append(_ar_predict(y, int(tr_idx[-1]) + 1, te_idx, p_star, horizon))
        p_used.append(p_star)
        rec_all.append(np.array([float(y[:int(ti)].mean()) if int(ti) > 0 else float(ytr.mean()) for ti in te_idx], dtype=np.float64))
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    rec = np.concatenate(rec_all)
    return {
        "mse": mse(yt, yp),
        "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp),
        "p_star_median": int(np.median(p_used)),
        "n_folds": len(folds),
        "y_true": yt,
        "y_pred": yp,
        "recursive_train_means": rec,
    }


def _bic_select_p(y: np.ndarray, p_grid: tuple[int, ...]) -> int:
    """BIC-select AR lag order on a training window."""
    from statsmodels.tsa.ar_model import AutoReg

    best_p, best_bic = p_grid[0], np.inf
    for p in p_grid:
        if p >= len(y) // 3:
            continue
        try:
            res = AutoReg(y, lags=p, old_names=False).fit()
            if res.bic < best_bic:
                best_bic = float(res.bic)
                best_p = p
        except Exception:
            continue
    return int(best_p)


def _ar_predict(
    y_full: np.ndarray,
    train_end: int,
    test_indices: np.ndarray,
    p: int,
    horizon: int,
) -> np.ndarray:
    """Rolling-origin AR(p) h-step forecast at each test index.

    For each test observation at absolute time `t`, fit AR on y[:t - horizon + 1]
    (all data strictly before the prediction horizon window) and emit the
    h-step-ahead forecast. This avoids the R03.1 HIGH bug where a single
    flat forecast was broadcast across the whole test window.

    Refits per test-t (correct protocol). With 12 test obs/fold × 20 folds
    that's ~240 AR fits per target — ~1-2s in statsmodels at n_train~500.
    """
    from statsmodels.tsa.ar_model import AutoReg

    out = np.empty(len(test_indices), dtype=np.float64)
    for i, t in enumerate(test_indices):
        origin = int(t) - horizon + 1
        if origin < p + 5:
            # Not enough training data — fall back to the mean of y[:train_end].
            out[i] = float(y_full[:train_end].mean())
            continue
        y_tr = y_full[:origin]
        try:
            res = AutoReg(y_tr, lags=p, old_names=False).fit()
            # h-step-ahead forecast from origin; take the final slot.
            fc = res.predict(start=origin, end=origin + horizon - 1)
            out[i] = float(fc[-1])
        except Exception:
            out[i] = float(y_tr[-1])  # RW fallback on fit failure
    return out


def random_walk_forecast(y: np.ndarray, horizon: int, cfg: Optional[PredictConfig] = None) -> dict:
    """Random walk: y_hat(t+h) = y(t - horizon) (rolling-origin)."""
    cfg = cfg or PredictConfig()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(n, horizon, initial_train, cfg.cv.re_estimate_every, embargo)
    yt_all, yp_all, rec_all = [], [], []
    for tr_idx, te_idx in folds:
        preds = np.empty(len(te_idx))
        for i, t in enumerate(te_idx):
            origin = int(t) - horizon
            preds[i] = float(y[max(origin, 0)])
        yt_all.append(y[te_idx])
        yp_all.append(preds)
        rec_all.append(np.array([float(y[:int(ti)].mean()) if int(ti) > 0 else float(y[tr_idx].mean()) for ti in te_idx], dtype=np.float64))
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    rec = np.concatenate(rec_all)
    return {
        "mse": mse(yt, yp), "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp), "n_folds": len(folds),
        "y_true": yt, "y_pred": yp, "recursive_train_means": rec,
    }


def ao_rw_forecast(y: np.ndarray, horizon: int, cfg: Optional[PredictConfig] = None) -> dict:
    """Atkeson-Ohanian RW for inflation.

    Canonical Atkeson-Ohanian (2001): pi_hat(t+h) = mean of last 4 QUARTERS
    of inflation. That's 12 months for monthly series, 4 quarters for
    quarterly. We dispatch via `cfg.data_freq`. For annualized yoy inflation
    the AO-RW forecast is effectively the last-year average of the series.
    """
    cfg = cfg or PredictConfig()
    ao_window = 12 if cfg.data_freq == "M" else 4
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(n, horizon, initial_train, cfg.cv.re_estimate_every, embargo)
    yt_all, yp_all, rec_all = [], [], []
    for tr_idx, te_idx in folds:
        # Rolling-origin: for each test slot, use the window anchored at
        # origin = t - horizon + 1.
        preds = np.empty(len(te_idx))
        for i, t in enumerate(te_idx):
            origin = int(t) - horizon + 1
            w = y[max(0, origin - ao_window) : origin]
            preds[i] = float(w.mean()) if len(w) else float(y[tr_idx[-1]])
        yt_all.append(y[te_idx])
        yp_all.append(preds)
        rec_all.append(np.array([float(y[:int(ti)].mean()) if int(ti) > 0 else float(y[tr_idx].mean()) for ti in te_idx], dtype=np.float64))
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    rec = np.concatenate(rec_all)
    return {
        "mse": mse(yt, yp), "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp), "n_folds": len(folds),
        "ao_window": int(ao_window),
        "y_true": yt, "y_pred": yp, "recursive_train_means": rec,
    }


def random_forest_forecast(
    X: np.ndarray,
    y: np.ndarray,
    horizon: int,
    cfg: Optional[PredictConfig] = None,
) -> dict:
    """Random forest on raw features, walk-forward."""
    from sklearn.ensemble import RandomForestRegressor

    cfg = cfg or PredictConfig()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(n, horizon, initial_train, cfg.cv.re_estimate_every, embargo)
    yt_all, yp_all, rec_all = [], [], []
    for tr_idx, te_idx in folds:
        rf = RandomForestRegressor(
            n_estimators=cfg.random_forest_trees,
            min_samples_leaf=cfg.random_forest_min_leaf,
            random_state=cfg.random_forest_seed,  # R16.4: decoupled from placebo_seed
        )
        rf.fit(X[tr_idx], y[tr_idx])
        yt_all.append(y[te_idx])
        yp_all.append(rf.predict(X[te_idx]))
        rec_all.append(np.array([float(y[:int(ti)].mean()) if int(ti) > 0 else float(y[tr_idx].mean()) for ti in te_idx], dtype=np.float64))
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    rec = np.concatenate(rec_all)
    return {
        "mse": mse(yt, yp), "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp), "n_folds": len(folds),
        "y_true": yt, "y_pred": yp, "recursive_train_means": rec,
    }


def placebo_forecast(
    n_features: int,
    y: np.ndarray,
    horizon: int,
    cfg: Optional[PredictConfig] = None,
) -> dict:
    """Feature-count-matched Gaussian placebo: random features, same CV pipeline."""
    cfg = cfg or PredictConfig()
    rng = np.random.default_rng(cfg.placebo_seed)
    X = rng.standard_normal((len(y), n_features))
    return ridge_walk_forward(X, y, cfg=cfg)


def strong_shrinkage_forecast(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Optional[PredictConfig] = None,
) -> dict:
    """Ridge at lambda so large coefficients collapse — the 'no-feature' floor."""
    cfg = cfg or PredictConfig()
    return ridge_walk_forward(X, y, cfg=cfg, lam=cfg.strong_shrinkage_lambda)


def pca_ridge_forecast(
    X_raw: np.ndarray,
    y: np.ndarray,
    k: int,
    cfg: Optional[PredictConfig] = None,
) -> dict:
    """PCA(k) + ridge, per-fold PCA fit to avoid OOS structure leakage.

    R11.2 fix: principal axes are refit inside each fold using only training
    data, then both train and test are transformed by that fold's PCA.
    """
    from sklearn.decomposition import PCA

    cfg = cfg or PredictConfig()
    y = np.asarray(y, dtype=np.float64).ravel()
    X_raw = np.asarray(X_raw, dtype=np.float64)
    n = len(y)
    k_eff = min(k, X_raw.shape[1])
    embargo = cfg.cv.embargo if cfg.cv.embargo > 0 else cfg.horizon - 1
    initial_train = max(int(cfg.cv.initial_train_frac * n), cfg.cv.initial_train_min)
    folds = walk_forward_folds(n, cfg.horizon, initial_train, cfg.cv.re_estimate_every, embargo)
    if not folds:
        raise ValueError("insufficient data for PCA-ridge walk-forward")
    yt_all, yp_all, lams_used, rec_all, n_fallback = [], [], [], [], 0
    for tr_idx, te_idx in folds:
        pca = PCA(n_components=k_eff)
        Xtr = pca.fit_transform(X_raw[tr_idx])
        Xte = pca.transform(X_raw[te_idx])
        fold_lam, used_fallback = _cv_select_lambda_with_flag(
            Xtr, y[tr_idx], cfg.ridge_lambda_grid, cfg.cv.inner_cv_folds,
            adaptive=cfg.ridge_lambda_adaptive,
        )
        if used_fallback:
            n_fallback += 1
        beta, icept = _ridge_fit(Xtr, y[tr_idx], fold_lam)
        yp_all.append(_ridge_predict(Xte, beta, icept))
        yt_all.append(y[te_idx])
        lams_used.append(fold_lam)
        rec_all.append(np.array([float(y[:int(ti)].mean()) if int(ti) > 0 else float(y[tr_idx].mean()) for ti in te_idx], dtype=np.float64))
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    rec = np.concatenate(rec_all)
    return {
        "mse": mse(yt, yp), "r2": r2(yt, yp),
        "r2_oos": r2_oos_recursive(yt, yp, rec),
        "directional_accuracy": directional_accuracy(yt, yp),
        "bias": forecast_bias(yt, yp),
        "cv_lambda_median": float(np.median(lams_used)),
        "n_fallback_folds": int(n_fallback),
        "n_folds": len(folds),
        "y_true": yt, "y_pred": yp, "recursive_train_means": rec,
    }


# -------------------------------------------- AoA sensitivity features


def sensitivity_ranked_features(
    sensitivity_matrices: dict[str, np.ndarray],
    n_features: int,
) -> list[tuple[str, int, int, float]]:
    """Rank the (leading-series, k, l) triples by |S_{i->nu}^{k,l}| and pick top-n.

    `sensitivity_matrices` maps leading-series name -> S matrix (K_i × K_nu).
    Returns list of (series_name, k, l, score) tuples sorted by score desc.
    """
    items: list[tuple[str, int, int, float]] = []
    for name, S in sensitivity_matrices.items():
        S = np.asarray(S)
        for k in range(S.shape[0]):
            for l in range(S.shape[1]):
                items.append((name, int(k), int(l), float(abs(S[k, l]))))
    items.sort(key=lambda t: -t[3])
    return items[:n_features]


# --------------------------------------------- top-level evaluate_target


def _build_ar_lag_matrix(y_level: np.ndarray, n_samples: int, p_lags: int) -> np.ndarray:
    """Build (n_samples, p_lags) matrix of lagged y_level values.

    For sample index t (0-based into the length-n_samples window),
    row t = [y_level[t], y_level[t-1], ..., y_level[t-p_lags+1]],
    with reflection-padding for t < p_lags - 1 (keeps rows finite;
    CV-ridge absorbs the edge artifact into the intercept).
    """
    y_level = np.asarray(y_level, dtype=np.float64).ravel()
    # y_level length must be >= n_samples. Take the last n_samples entries
    # as the "current" time index 0..n_samples-1 to stay calendar-aligned.
    if len(y_level) < n_samples:
        raise ValueError(f"y_level too short: {len(y_level)} < {n_samples}")
    start = len(y_level) - n_samples
    out = np.empty((n_samples, p_lags), dtype=np.float64)
    for t in range(n_samples):
        abs_t = start + t
        for k in range(p_lags):
            idx = abs_t - k
            out[t, k] = y_level[max(idx, 0)]
    return out


def evaluate_target(
    y: np.ndarray,
    X_aoa: np.ndarray,
    X_raw: np.ndarray,
    name: str,
    horizon: int,
    structural_benchmark_x: Optional[np.ndarray] = None,
    cfg: Optional[PredictConfig] = None,
    y_level: Optional[np.ndarray] = None,
    ar_aug_lags: int = 12,
) -> dict:
    """Run the full benchmark ladder on a single (target, horizon) problem.

    Parameters
    ----------
    y : (T,) target series aligned at horizon h
    X_aoa : (T, n_aoa) AoA sensitivity-ranked features (Level 2/1 output)
    X_raw : (T, p) raw input panel (same panel PCA-ridge and RF use)
    name : target name (for logging)
    horizon : steps ahead
    structural_benchmark_x : optional regressor for benchmark (viii);
        for Okun target, pass real-GDP log-differences.
    cfg : PredictConfig
    y_level : optional (>=T,) underlying level series for AR-augmented ridge.
        If provided, adds two benchmarks: `ar_lags_only` and
        `ar_aug_aoa` (AR lags concatenated with X_aoa). The encompassing
        test compares `ar_aug_aoa` against `ar_lags_only` to quantify
        incremental predictive content of AoA features over persistence
        (Clark-West 2007, Giacomini-White 2006).
    ar_aug_lags : number of lagged y_level values to include in the
        augmented ridge (default 12 = 1y of monthly history).

    Returns dict keyed by benchmark name → metric dict.
    """
    from dataclasses import replace

    cfg = cfg or PredictConfig()
    cfg = replace(cfg, horizon=horizon)  # R03.2: OCP-safe field override

    out: dict[str, dict] = {}

    # Sign-gauge diagnostic (not applied) on the AoA ridge: counts folds where
    # training dir_acc is significantly below chance. See ridge_walk_forward
    # for why we don't act on it by default — the sign relationship is not
    # stable across folds on some cells (honest finding, not a bug to paper
    # over).
    out["aoa_ridge"] = ridge_walk_forward(X_aoa, y, cfg=cfg)
    out["ar_p_star"] = ar_p_star_forecast(y, horizon=horizon, cfg=cfg)

    # Encompassing test: AR(p)-lags-only ridge vs AR-lags augmented with AoA.
    # This is the literature-standard test for incremental predictive content
    # (Clark-West 2007 J. Econometrics; Giacomini-White 2006 Econometrica).
    if y_level is not None:
        n = len(y)
        X_arlags = _build_ar_lag_matrix(y_level, n, ar_aug_lags)
        X_aug = np.concatenate([X_arlags, X_aoa], axis=1)
        out["ar_lags_only"] = ridge_walk_forward(X_arlags, y, cfg=cfg)
        out["ar_aug_aoa"] = ridge_walk_forward(X_aug, y, cfg=cfg)
    out["rw"] = random_walk_forecast(y, horizon=horizon, cfg=cfg)
    # AO-RW for inflation targets only — caller can post-filter.
    if name.startswith("phillips") or "cpi" in name.lower() or "infl" in name.lower():
        out["ao_rw"] = ao_rw_forecast(y, horizon=horizon, cfg=cfg)

    out["linear_ridge_raw"] = ridge_walk_forward(X_raw, y, cfg=cfg)
    out["pca_ridge"] = pca_ridge_forecast(X_raw, y, k=cfg.n_aoa_features, cfg=cfg)
    out["random_forest_raw"] = random_forest_forecast(X_raw, y, horizon=horizon, cfg=cfg)
    out["placebo"] = placebo_forecast(cfg.n_aoa_features, y, horizon=horizon, cfg=cfg)
    # R11.3: compute strong-shrinkage on BOTH feature sets for comparability.
    out["strong_shrinkage_aoa"] = strong_shrinkage_forecast(X_aoa, y, cfg=cfg)
    out["strong_shrinkage_raw"] = strong_shrinkage_forecast(X_raw, y, cfg=cfg)

    if structural_benchmark_x is not None:
        out["okun_gdp"] = ridge_walk_forward(
            structural_benchmark_x.reshape(-1, 1), y, cfg=cfg
        )

    # MSE ratios (plan.tex primary denominator = linear_ridge_raw).
    denom_primary = max(out["linear_ridge_raw"]["mse"], 1e-12)
    denom_ar = max(out["ar_p_star"]["mse"], 1e-12)
    for k in out:
        out[k]["mse_ratio_vs_linear_ridge_raw"] = out[k]["mse"] / denom_primary
        out[k]["mse_ratio_vs_ar"] = out[k]["mse"] / denom_ar

    # R11.1: explicit falsifiable gates for the paper-level claim.
    # R10 panel Step 14: add `aoa_dir_acc_edge_vs_ar` — raw dir_acc = 1.0 is a
    # target-monotonicity artifact (every baseline also hits 1.0 on a
    # monotonically-trending test window). The EDGE over AR is the honest signal.
    # Grade Fix #5: binomial test on dir_acc detects systematic sign inversion
    # (e.g. Okun h=6 dir_acc = 0.23 at p~1e-19 below chance indicates a sign
    # error in feature construction, not noise). `aoa_dir_acc_inverted` fires
    # when dir_acc < 0.5 with binomial p < 0.05 — a fixable pipeline bug, not
    # a valid forecast result.
    aoa_dir_acc = out["aoa_ridge"]["directional_accuracy"]
    aoa_dir_p = dir_acc_binomial_p(
        out["aoa_ridge"]["y_true"], out["aoa_ridge"]["y_pred"]
    )
    out["_gates"] = {
        "aoa_mse_ratio_vs_linear_ridge_raw": out["aoa_ridge"]["mse"] / denom_primary,
        "aoa_mse_ratio_vs_ar": out["aoa_ridge"]["mse"] / denom_ar,
        "aoa_beats_linear_ridge_raw": out["aoa_ridge"]["mse"] < out["linear_ridge_raw"]["mse"],
        "aoa_beats_ar_p_star": out["aoa_ridge"]["mse"] < out["ar_p_star"]["mse"],
        "aoa_beats_placebo_r2": out["aoa_ridge"]["r2"] > out["placebo"]["r2"],
        "aoa_beats_strong_shrinkage_r2": out["aoa_ridge"]["r2"] > out["strong_shrinkage_aoa"]["r2"],
        "aoa_directional_accuracy_above_055": aoa_dir_acc > 0.55,
        "aoa_dir_acc_edge_vs_ar": (
            aoa_dir_acc - out["ar_p_star"]["directional_accuracy"]
        ),
        "aoa_r2_above_pca_ridge_minus_1se": out["aoa_ridge"]["r2"] > out["pca_ridge"]["r2"],
        "aoa_dir_acc_binomial_p": aoa_dir_p,
        "aoa_dir_acc_inverted": bool(aoa_dir_acc < 0.5 and aoa_dir_p < 0.05),
        # Campbell-Thompson R^2_OOS (recursive training-mean benchmark).
        # This is the forecast-evaluation-literature standard and is the
        # honest measure of skill on trending/level-shifting targets.
        "aoa_r2_oos": out["aoa_ridge"].get("r2_oos", np.nan),
        "aoa_r2_oos_above_zero": bool(out["aoa_ridge"].get("r2_oos", -np.inf) > 0.0),
        # Multi-null R² comparison (Criterion 8): AoA residual against three
        # separate benchmarks so reader can reconcile the "R² negative but
        # MSE-ratio favorable" paradox. Each is 1 - MSE(aoa)/MSE(benchmark);
        # positive ⇒ AoA beats that benchmark on pooled MSE.
        "aoa_r2_vs_demeaned_y": out["aoa_ridge"]["r2"],
        "aoa_r2_vs_rw": 1.0 - out["aoa_ridge"]["mse"] / max(out["rw"]["mse"], 1e-12),
        "aoa_r2_vs_ar": 1.0 - out["aoa_ridge"]["mse"] / max(out["ar_p_star"]["mse"], 1e-12),
        "aoa_r2_vs_linear_raw": 1.0 - out["aoa_ridge"]["mse"] / max(out["linear_ridge_raw"]["mse"], 1e-12),
        # MSE↔R² sign-consistency assertion: if AoA's MSE exceeds linear-raw's
        # MSE then AoA's R² (any null) must be below linear-raw's R² on the
        # same null. A violation indicates pooling bug or SS_tot mismatch.
        "aoa_mse_r2_consistency_vs_linear_raw": bool(
            (out["aoa_ridge"]["mse"] > out["linear_ridge_raw"]["mse"]) ==
            (out["aoa_ridge"]["r2"] < out["linear_ridge_raw"]["r2"])
        ),
        # Per-fold R² distribution summary (Criterion 8 level-4).
        "aoa_fold_r2_oos_min": float(
            min(out["aoa_ridge"]["diagnostics"]["fold_r2_oos"])
        ) if out["aoa_ridge"].get("diagnostics") else float("nan"),
        "aoa_fold_r2_oos_max": float(
            max(out["aoa_ridge"]["diagnostics"]["fold_r2_oos"])
        ) if out["aoa_ridge"].get("diagnostics") else float("nan"),
        "aoa_fold_r2_oos_frac_positive": float(
            np.mean([r > 0 for r in out["aoa_ridge"]["diagnostics"]["fold_r2_oos"]])
        ) if out["aoa_ridge"].get("diagnostics") else float("nan"),
        # Gauge-flip and numerics diagnostics (Criterion 11 + 7 level-4).
        "aoa_n_sign_flipped_folds": int(
            out["aoa_ridge"].get("n_sign_flipped_folds", 0)
        ),
        "aoa_n_boundary_pegged_folds": int(
            out["aoa_ridge"].get("n_boundary_pegged_folds", 0)
        ),
        "aoa_target_adf_p": _adf_pvalue_safe(y),
        "aoa_target_stationary_at_005": bool(_adf_pvalue_safe(y) < 0.05),
    }
    # Encompassing test gate: did AoA features add incremental value over
    # AR-only persistence? Compare MSE of ar_aug_aoa vs ar_lags_only.
    if "ar_aug_aoa" in out and "ar_lags_only" in out:
        mse_aug = out["ar_aug_aoa"]["mse"]
        mse_ar  = out["ar_lags_only"]["mse"]
        out["_gates"]["ar_aug_mse_ratio_vs_ar_lags_only"] = mse_aug / max(mse_ar, 1e-12)
        out["_gates"]["aoa_adds_value_over_ar_lags"] = mse_aug < mse_ar
        out["_gates"]["ar_aug_r2_oos"] = out["ar_aug_aoa"].get("r2_oos", np.nan)
        out["_gates"]["ar_lags_only_r2_oos"] = out["ar_lags_only"].get("r2_oos", np.nan)
        out["_gates"]["aoa_delta_r2_oos_vs_ar_lags"] = (
            float(out["ar_aug_aoa"].get("r2_oos", np.nan))
            - float(out["ar_lags_only"].get("r2_oos", np.nan))
        )

    out["_meta"] = {
        "target": name,
        "horizon": horizon,
        "n_observations": int(len(y)),
        "n_aoa_features": X_aoa.shape[1],
        "n_raw_features": X_raw.shape[1],
        "primary_denominator": "linear_ridge_raw",
        "data_freq": cfg.data_freq,
    }
    return out
