"""
aoa_v3.viz — Attractor rendering (paper § 5 Structure section).

For each law, render a 2D + 3D DMAP-eigencoord trajectory of the primary
target series, with time-coloring and regime-anchor annotations. Okun also
carries a hysteresis inset: expansion vs. contraction phase ellipses in
(psi_1, psi_2) with rotation / elongation statistics.

This module is the plotting primitive. Script `02_render_attractors.py` wraps
it with paper-binding inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


@dataclass(frozen=True)
class AttractorRenderConfig:
    dpi: int = 300
    scatter_size: float = 12.0
    trajectory_alpha: float = 0.35
    cmap: str = "viridis"
    figsize_2d: tuple[float, float] = (6.0, 5.0)
    figsize_3d: tuple[float, float] = (6.0, 5.0)
    figsize_combined: tuple[float, float] = (13.0, 5.5)


def _ellipse_stats(pts: np.ndarray) -> dict:
    """Fit a 1-sigma covariance ellipse on (N, 2) points.

    Returns center, major/minor axis lengths, rotation angle (radians),
    and median geodesic path length (here = median cumulative arc length).
    """
    if len(pts) < 3:
        return {
            "major": 0.0, "minor": 0.0, "angle_rad": 0.0,
            "center_x": float(np.nan), "center_y": float(np.nan),
            "geodesic_median": 0.0,
        }
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_axis = 2.0 * float(np.sqrt(max(eigvals[0], 0)))
    minor_axis = 2.0 * float(np.sqrt(max(eigvals[1], 0)))
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    geodesic = float(np.median(np.cumsum(steps))) if len(steps) else 0.0
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "major": major_axis,
        "minor": minor_axis,
        "angle_rad": angle,
        "geodesic_median": geodesic,
    }


def render_attractor(
    coords: np.ndarray,
    times: pd.DatetimeIndex,
    title: str,
    out_path: Path,
    anchors: Optional[dict[str, pd.Timestamp]] = None,
    cfg: Optional[AttractorRenderConfig] = None,
) -> dict:
    """Render a 2-panel figure (2D + 3D) of a DMAP-embedded attractor trajectory.

    Parameters
    ----------
    coords : (n_t, d) ndarray — at least 3 eigencoord columns
    times : DatetimeIndex of length n_t
    title : figure title (law name)
    out_path : PDF output path
    anchors : dict mapping label → timestamp (annotates 2D panel)
    cfg : styling

    Returns metadata dict (time range, eigencoord magnitudes, anchor pixel positions).
    """
    cfg = cfg or AttractorRenderConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(
            f"coords must be (n, >=3) for 2D+3D rendering; got {coords.shape}"
        )
    times = pd.DatetimeIndex(times)
    if len(times) != coords.shape[0]:
        raise ValueError(
            f"times ({len(times)}) and coords ({coords.shape[0]}) length mismatch"
        )

    t_num = (times - times[0]).total_seconds().values
    t_num = t_num / max(t_num[-1], 1.0)

    fig = plt.figure(figsize=cfg.figsize_combined)
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    sc2 = ax2d.scatter(
        coords[:, 0], coords[:, 1],
        c=t_num, cmap=cfg.cmap, s=cfg.scatter_size,
    )
    ax2d.plot(
        coords[:, 0], coords[:, 1],
        color="black", alpha=cfg.trajectory_alpha, linewidth=0.6,
    )
    ax2d.set_xlabel(r"$\psi_1$")
    ax2d.set_ylabel(r"$\psi_2$")
    ax2d.set_title(f"{title} — 2D DMAP")
    cbar = plt.colorbar(sc2, ax=ax2d, shrink=0.8)
    cbar.set_label("time")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels([times[0].strftime("%Y-%m"), times[-1].strftime("%Y-%m")])

    ax3d.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=t_num, cmap=cfg.cmap, s=cfg.scatter_size,
    )
    ax3d.plot(
        coords[:, 0], coords[:, 1], coords[:, 2],
        color="black", alpha=cfg.trajectory_alpha, linewidth=0.6,
    )
    ax3d.set_xlabel(r"$\psi_1$")
    ax3d.set_ylabel(r"$\psi_2$")
    ax3d.set_zlabel(r"$\psi_3$")
    ax3d.set_title(f"{title} — 3D DMAP")

    anchor_meta: list[dict] = []
    if anchors:
        for label, ts in anchors.items():
            ts = pd.Timestamp(ts)
            if ts < times[0] or ts > times[-1]:
                warnings.warn(
                    f"anchor {label}@{ts.date()} outside render window "
                    f"[{times[0].date()}, {times[-1].date()}]; skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            idx = int(np.argmin(np.abs((times - ts).total_seconds())))
            ax2d.annotate(
                label,
                xy=(coords[idx, 0], coords[idx, 1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8},
                color="red",
            )
            anchor_meta.append({
                "label": label,
                "date": ts.strftime("%Y-%m-%d"),
                "psi1": float(coords[idx, 0]),
                "psi2": float(coords[idx, 1]),
            })

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": str(out_path),
        "n_points": int(coords.shape[0]),
        "time_range": [times[0].strftime("%Y-%m-%d"), times[-1].strftime("%Y-%m-%d")],
        "psi_range": [
            [float(coords[:, 0].min()), float(coords[:, 0].max())],
            [float(coords[:, 1].min()), float(coords[:, 1].max())],
            [float(coords[:, 2].min()), float(coords[:, 2].max())],
        ],
        "anchors": anchor_meta,
    }


def _reduce_angle_mod_pi(angle_rad: float) -> float:
    """Wrap angle difference into (-π/2, π/2]. Ellipses are π-periodic
    (major-axis orientation is equivalent under 180° rotation), so the
    smallest equivalent signed rotation between two major axes lives in
    that interval (R12 panel).
    """
    return ((angle_rad + np.pi / 2) % np.pi) - np.pi / 2


def _bootstrap_ellipse_cis(
    pts: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Bootstrap 95% CIs for ellipse major/minor/angle_rad (R02 panel)."""
    if len(pts) < 3:
        return {"major_ci95": [0.0, 0.0], "minor_ci95": [0.0, 0.0],
                "angle_ci95_rad": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    majors, minors, angles = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pts), len(pts))
        s = _ellipse_stats(pts[idx])
        majors.append(s["major"])
        minors.append(s["minor"])
        angles.append(s["angle_rad"])
    return {
        "major_ci95": [float(np.percentile(majors, 2.5)),
                       float(np.percentile(majors, 97.5))],
        "minor_ci95": [float(np.percentile(minors, 2.5)),
                       float(np.percentile(minors, 97.5))],
        "angle_ci95_rad": [float(np.percentile(angles, 2.5)),
                           float(np.percentile(angles, 97.5))],
    }


def render_okun_hysteresis(
    coords: np.ndarray,
    times: pd.DatetimeIndex,
    nber_recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    out_path: Path,
    cfg: Optional[AttractorRenderConfig] = None,
    n_bootstrap: int = 1000,
) -> dict:
    """Okun-specific inset: fit 1-sigma ellipses to expansion vs. contraction
    DMAP-coord clouds and compare rotation / elongation.

    NBER recession windows are half-open: [start, end). The complement within
    `times` is expansion. Includes bootstrap 95% CIs on major, minor, angle.
    Rotation-delta between expansion/contraction is reduced mod π per R12.
    """
    cfg = cfg or AttractorRenderConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] < 2:
        raise ValueError(
            f"Okun hysteresis needs at least 2 DMAP coords; got {coords.shape}"
        )
    times = pd.DatetimeIndex(times)

    is_contraction = np.zeros(len(times), dtype=bool)
    for start, end in nber_recession_periods:
        mask = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))
        is_contraction |= mask
    is_expansion = ~is_contraction

    exp_pts = coords[is_expansion, :2]
    con_pts = coords[is_contraction, :2]

    exp_stats = _ellipse_stats(exp_pts)
    con_stats = _ellipse_stats(con_pts)
    exp_ci = _bootstrap_ellipse_cis(exp_pts, n_boot=n_bootstrap, seed=0)
    con_ci = _bootstrap_ellipse_cis(con_pts, n_boot=n_bootstrap, seed=1)

    fig, ax = plt.subplots(figsize=cfg.figsize_2d)
    ax.scatter(exp_pts[:, 0], exp_pts[:, 1], s=cfg.scatter_size, color="tab:blue",
               alpha=0.6, label=f"expansion (n={len(exp_pts)})")
    ax.scatter(con_pts[:, 0], con_pts[:, 1], s=cfg.scatter_size, color="tab:red",
               alpha=0.6, label=f"contraction (n={len(con_pts)})")
    for stats_dict, color in ((exp_stats, "tab:blue"), (con_stats, "tab:red")):
        if stats_dict["major"] > 0:
            ell = Ellipse(
                (stats_dict["center_x"], stats_dict["center_y"]),
                width=stats_dict["major"],
                height=stats_dict["minor"],
                angle=np.degrees(stats_dict["angle_rad"]),
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
            )
            ax.add_patch(ell)
    ax.set_xlabel(r"$\psi_1$")
    ax.set_ylabel(r"$\psi_2$")
    ax.set_title("Okun hysteresis — expansion vs. contraction")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    # Π-reduced rotation delta (R12).
    angle_delta_rad = _reduce_angle_mod_pi(
        exp_stats["angle_rad"] - con_stats["angle_rad"]
    )
    return {
        "path": str(out_path),
        "expansion": {"n": int(len(exp_pts)), **exp_stats, **exp_ci},
        "contraction": {"n": int(len(con_pts)), **con_stats, **con_ci},
        "rotation_delta_deg": float(np.degrees(abs(angle_delta_rad))),
        "rotation_delta_signed_deg": float(np.degrees(angle_delta_rad)),
        "elongation_ratio_expansion": (
            exp_stats["major"] / max(exp_stats["minor"], 1e-12)
        ),
        "elongation_ratio_contraction": (
            con_stats["major"] / max(con_stats["minor"], 1e-12)
        ),
    }


def render_spectrum_overlay(
    dmap_eigenvalues: np.ndarray,
    pca_eigenvalues: np.ndarray,
    title: str,
    out_path: Path,
    cfg: Optional[AttractorRenderConfig] = None,
) -> dict:
    """DMAP vs. PCA eigenvalue decay (log-scale, normalized to leading λ).

    R13 panel: DMAP kernel-eigenvalues and PCA variance-eigenvalues live on
    incommensurable scales, so we plot `λ_k / λ_1` for each — the decay shape
    is what's compared. A slower DMAP decay relative to PCA is the nonlinear-
    compression signature (paper §5 Nonlinearity Demonstrations item 2).
    """
    cfg = cfg or AttractorRenderConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dmap_eigenvalues = np.asarray(dmap_eigenvalues, dtype=np.float64)
    pca_eigenvalues = np.asarray(pca_eigenvalues, dtype=np.float64)

    dmap_abs = np.abs(dmap_eigenvalues)
    pca_abs = np.abs(pca_eigenvalues)
    dmap_norm = dmap_abs / max(dmap_abs[0], 1e-30) if len(dmap_abs) else dmap_abs
    pca_norm = pca_abs / max(pca_abs[0], 1e-30) if len(pca_abs) else pca_abs

    fig, ax = plt.subplots(figsize=cfg.figsize_2d)
    k_dmap = np.arange(1, len(dmap_norm) + 1)
    k_pca = np.arange(1, len(pca_norm) + 1)
    ax.semilogy(k_dmap, dmap_norm, "o-", label="DMAP")
    ax.semilogy(k_pca, pca_norm, "s--", label="PCA")
    ax.set_xlabel("index k")
    ax.set_ylabel(r"$|\lambda_k| / |\lambda_1|$")
    ax.set_title(f"{title} — DMAP vs PCA normalized spectrum")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(out_path),
        "dmap_normalized_top6": [float(x) for x in dmap_norm[:6]],
        "pca_normalized_top6": [float(x) for x in pca_norm[:6]],
    }
