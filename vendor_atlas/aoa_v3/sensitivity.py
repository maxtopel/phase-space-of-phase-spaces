"""
aoa_v3.sensitivity — Sensitivity tensor from paper § 4.

Given two metric-measure spaces (M_i with slow modes {Φ^i_k}, eigenvalues {λ^i_k}
and M_j with slow modes {Φ^j_l}, eigenvalues {λ^j_l}) and a GW transport plan
T_{ij} from M_i to M_j, the sensitivity of mode l on M_j to mode k on M_i is:

    S_{i -> j}^{k,l} = (λ^j_l / Σ_l λ^j_l) · (T_{ij} Φ^i_k) · Φ^j_l

where · is the vector dot product and T_{ij} @ Φ^i_k transports the k-th
slow mode of the source attractor into the target's support space.

Key properties (paper § 4):
  - Asymmetric: S_{i->j}^{k,l} != S_{j->i}^{l,k} in general — a leading
    indicator's modes explain a lagging indicator's modes, but the reverse
    need not hold (paper example: US interest rates → Canadian default rates).
  - Weighted by target-eigenvalue mass (λ^j_l / Σ_l λ^j_l): the more variance
    Φ^j_l carries on M_j, the more it counts in the sensitivity.
  - Per-mode-pair entries form the tensor α with entries α_{i,j,k,l}.

This module exposes:
  - `sensitivity(T, phi_i, lam_i, phi_j, lam_j)`: single-pair (k, l) matrix.
  - `sensitivity_tensor(plans, modes, eigvals)`: full tensor across a set of
    (M_i, M_j) pairs.
  - `granger_asymmetry(x, y, max_lag)`: linear Granger-F-statistic asymmetry
    benchmark for the paper § 4 nonlinear vs. linear information-flow test
    (plan.tex Nonlinearity Demonstrations §5.2 item 2).

Paper-binding conventions:
  - T has shape (n_i, n_j), is a COUPLING (total mass = 1; row sums = marginals
    p_i, col sums = marginals p_j).
  - Φ^i is a (n_i, K_i) matrix whose columns are mode vectors on M_i's support.
  - λ^i is a (K_i,) vector of paired eigenvalues.
  - All modes/eigvals are the nontrivial ones (trivial λ_0 = 1 dropped upstream
    via aoa_v3.dmap / aoa_v3.aoa).

LaTeX → NumPy crosswalk:
  Paper writes "(T_{ij} Φ^i_k)_l = Σ_m T_{m,l} Φ^i_{k,m}". That's a left-multiply
  by T^T when Φ^i_k is a column vector, implemented here as `phi_i.T @ T` so row k
  of the result is the transported k-th mode of source i (length n_j).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------- validation


def _require_shape_match(
    T: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    coupling_atol: float = 1e-6,
) -> None:
    if T.ndim != 2:
        raise ValueError(f"T must be 2-D, got shape {T.shape}")
    if phi_i.ndim != 2 or phi_j.ndim != 2:
        raise ValueError("phi_i and phi_j must be 2-D (n, K)")
    n_i, n_j = T.shape
    if phi_i.shape[0] != n_i:
        raise ValueError(
            f"phi_i has n={phi_i.shape[0]} but T expects n_i={n_i}"
        )
    if phi_j.shape[0] != n_j:
        raise ValueError(
            f"phi_j has n={phi_j.shape[0]} but T expects n_j={n_j}"
        )
    # T must be a coupling (sum = 1); a row-stochastic map (sum = n_i) would
    # silently corrupt the transported-mode scale (R01-MEDIUM).
    total = float(T.sum())
    if abs(total - 1.0) > coupling_atol:
        raise ValueError(
            f"T is not a coupling (total mass = {total:.6g}); "
            "expected sum ~ 1.0. Renormalize via T / T.sum() upstream."
        )
    # Modes must not include the trivial constant eigenvector (R13-LOW):
    # if any column is constant, the pushed-forward mode collapses to the
    # target marginal and silently corrupts downstream sensitivity.
    for name, phi in (("phi_i", phi_i), ("phi_j", phi_j)):
        stds = phi.std(axis=0)
        if np.any(stds < 1e-12):
            raise ValueError(
                f"{name} contains a constant column (std < 1e-12); "
                "trivial eigenvector must be dropped upstream"
            )


# ---------------------------------------------------------- core tensor


def sensitivity(
    T: np.ndarray,
    phi_i: np.ndarray,
    lam_i: np.ndarray,        # noqa: ARG001  (kept for API symmetry; unused in formula)
    phi_j: np.ndarray,
    lam_j: np.ndarray,
) -> np.ndarray:
    """Sensitivity tensor slice for a single source/target pair.

    Computes the (K_i, K_j) matrix S with
        S[k, l] = (λ^j_l / Σ_l λ^j_l) * (T phi_i_k) . phi_j_l
    using float64 arithmetic.

    `lam_i` is accepted but not used — the formula weights only by target
    eigenvalue mass. We keep it in the API so callers can pass symmetric
    data without re-slicing; future extensions (e.g. symmetric normalization)
    can use it.

    Returns
    -------
    np.ndarray, shape (K_i, K_j)
    """
    _require_shape_match(T, phi_i, phi_j)
    lam_j = np.asarray(lam_j, dtype=np.float64).ravel()
    total = lam_j.sum()
    if total <= 0:
        raise ValueError(f"sum of target eigenvalues is non-positive: {total}")
    weights = lam_j / total  # (K_j,)
    # Transport each source mode to target support: (K_i, n_j) = (K_i, n_i) @ (n_i, n_j)
    transported = phi_i.T @ T  # (K_i, n_j)
    # Dot with target modes: (K_i, K_j) = (K_i, n_j) @ (n_j, K_j)
    dot = transported @ phi_j
    return dot * weights[np.newaxis, :]


def sensitivity_tensor(
    plans: dict[tuple[int, int], np.ndarray],
    modes: list[np.ndarray],
    eigvals: list[np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute S^{k,l} for every (i, j) pair with a plan provided.

    `plans` maps (i, j) -> T_{ij} (n_i × n_j transport plan).
    `modes[i]` is the (n_i, K_i) mode matrix for attractor i.
    `eigvals[i]` is the (K_i,) eigenvalue vector.

    Returns a dict keyed by the same (i, j) with values the sensitivity
    slice (K_i, K_j).
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for (i, j), T in plans.items():
        out[(i, j)] = sensitivity(T, modes[i], eigvals[i], modes[j], eigvals[j])
    return out


def sensitivity_asymmetry(
    S_ij: np.ndarray,
    S_ji: np.ndarray,
    method: str = "frobenius",
) -> float:
    """Summary scalar for the paper § 4 directional-information-flow claim.

    Parameters
    ----------
    S_ij, S_ji : np.ndarray
        Sensitivity slices. S_ji must have transposed shape relative to S_ij.
    method : {"frobenius", "max", "gauge_invariant"}
        - "frobenius" (default): ||S_ij - S_ji.T||_F / max(||S_ij||_F, ||S_ji||_F).
          Rotation-stable in magnitude across mode columns.
        - "max": max-abs-entry variant (the v1 metric). Rotation-sensitive.
        - "gauge_invariant": compares Gram matrices
          ||S_ij S_ij^T - (S_ji.T)(S_ji.T)^T||_F / peak. Invariant under per-
          mode sign flips (DMAP eigenvectors are defined up to sign).

    All three are scale-normalized so the output lies in [0, ~2]; larger values
    signal stronger directional flow.
    """
    if S_ij.shape[0] != S_ji.shape[1] or S_ij.shape[1] != S_ji.shape[0]:
        raise ValueError(
            f"asymmetry requires S_ij {S_ij.shape} and S_ji {S_ji.shape} to have transposed shape"
        )
    diff = S_ij - S_ji.T  # (K_i, K_j)
    if method == "max":
        scale = max(np.abs(S_ij).max(), np.abs(S_ji).max(), 1e-12)
        return float(np.abs(diff).max() / scale)
    if method == "frobenius":
        scale = max(np.linalg.norm(S_ij), np.linalg.norm(S_ji), 1e-12)
        return float(np.linalg.norm(diff) / scale)
    if method == "gauge_invariant":
        G_ij = S_ij @ S_ij.T
        G_ji_T = S_ji.T @ S_ji  # (K_i, K_i) Gram of S_ji.T (same shape as G_ij)
        gdiff = G_ij - G_ji_T
        scale = max(np.linalg.norm(G_ij), np.linalg.norm(G_ji_T), 1e-12)
        return float(np.linalg.norm(gdiff) / scale)
    raise ValueError(
        f"unknown method {method!r}; choose 'frobenius', 'max', or 'gauge_invariant'"
    )


# ---------------------------------- linear Granger benchmark for §5.2


def granger_asymmetry(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 5,
) -> dict:
    """Linear Granger-causality F-statistic in both directions + asymmetry.

    Uses a SINGLE specified lag `lag` (not a max-F-across-lags selector, which
    has multiple-comparisons bias — R02 Step 12 panel). Caller picks a lag in
    advance; on toy systems we use lag = 5 matching paper § 4's plain-lag
    convention. For real macro data, AIC-select externally and pass the result.

    Returns a dict with keys:
      - granger_F_x_to_y, granger_F_y_to_x : F-stat at `lag`
      - asymmetry: |F_{x→y} − F_{y→x}| / max(F_{x→y}, F_{y→x})
      - p_x_to_y, p_y_to_x: p-values at `lag`
      - lag: the single lag used
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(f"x, y length mismatch: {len(x)} vs {len(y)}")
    if len(x) < lag * 5:
        raise ValueError(
            f"series too short for lag={lag}; need >= {lag * 5} rows"
        )

    def _single_lag_F_p(data):
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=[lag], verbose=False)
        stats, _models = res[lag]
        F, p, _, _ = stats["ssr_ftest"]
        return float(F), float(p)

    # "x causes y" -> [y, x]; "y causes x" -> [x, y]
    F_x_to_y, p_x_to_y = _single_lag_F_p(np.column_stack([y, x]))
    F_y_to_x, p_y_to_x = _single_lag_F_p(np.column_stack([x, y]))
    peak = max(F_x_to_y, F_y_to_x, 1e-12)
    asymmetry = float(abs(F_x_to_y - F_y_to_x) / peak)
    return {
        "granger_F_x_to_y": F_x_to_y,
        "granger_F_y_to_x": F_y_to_x,
        "asymmetry": asymmetry,
        "p_x_to_y": p_x_to_y,
        "p_y_to_x": p_y_to_x,
        "lag": int(lag),
    }
