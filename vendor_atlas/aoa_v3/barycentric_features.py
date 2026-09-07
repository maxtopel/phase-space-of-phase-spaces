"""aoa_v3.barycentric_features — Level-1 prediction via GW barycenters.

Implements paper §Prediction Level 1 (the primary paper claim):

  1. Group leading series into domain-coherent clusters.
  2. For each cluster, build the GW barycenter B_c of its member
     attractors (paper §3.3).
  3. Compute ONE GW transport T: B_c → target-attractor (per cluster),
     not per individual series.
  4. Sensitivity tensor S^{k,l}_{B_c → ν} on barycentric modes.
  5. Features are functions of (barycentric state, target state,
     sensitivity weights) per cluster — one feature block per cluster,
     not per individual series.

The barycenter is the "attractor of attractors" of the cluster:
the Fréchet mean in GW space that captures the shared geometric
dynamics of all members while remaining invariant to member-specific
noise. Features derived from it are predictions against the cluster's
typical dynamics, not against any specific series.

Implementation pattern:
  build_cluster_barycenter(cluster_series_coords) -> (B, T_plans, Phi_B, evals_B)
  barycentric_state_at_time(coord_i(t), T_{i->B}, Phi_B) -> bary coord
  sensitivity_tensor(T_{B->target}, Phi_B, Phi_target) -> S^{k,l}

Unit-tested: with a single cluster member, B reduces to the member
attractor (up to n_s FPS subsample), recovering Level-2 pairwise coupling
as a degenerate case.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from aoa_v3.barycenter import (
    BarycenterConfig, gw_barycenter, member_transport_plans,
)
from aoa_v3.dmap import dmap_dense
from aoa_v3.gw import GWConfig, gw_distance


def dmap_on_cost_matrix(
    B_cost: np.ndarray,
    n_components: int = 6,
    k_adaptive: int = 10,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run DMAP on a barycenter intra-distance matrix.

    The barycenter is a set of n_s abstract support points with known
    pairwise distances (B_cost). We embed them via DMAP using the cost
    matrix directly as pairwise distances, yielding per-support-point
    diffusion coordinates Phi^B and eigenvalues lambda^B that can be
    used in the sensitivity tensor.

    Internally, we construct a dummy point cloud in R^{n_s} whose
    pairwise Euclidean distances match B_cost (via classical MDS),
    then hand it to dmap_dense. This preserves the kernel-bandwidth
    logic and normalization of the standard DMAP pipeline.
    """
    B_cost = np.asarray(B_cost, dtype=np.float64)
    B_cost = 0.5 * (B_cost + B_cost.T)
    n_s = B_cost.shape[0]

    # Classical MDS embedding: double-center squared-distance matrix, eigendecompose.
    D2 = B_cost ** 2
    J = np.eye(n_s) - np.ones((n_s, n_s)) / n_s
    G = -0.5 * J @ D2 @ J
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # Take positive eigenvalues; drop near-zero/negative (rounding noise).
    pos = eigvals > 1e-12
    mds_dim = min(int(pos.sum()), n_s - 1)
    X_mds = eigvecs[:, :mds_dim] * np.sqrt(np.maximum(eigvals[:mds_dim], 0))

    coords_B, evals_B = dmap_dense(
        X_mds, n_components=n_components,
        alpha=alpha, k_adaptive=k_adaptive, theiler=0,
    )
    return coords_B, evals_B


def build_cluster_barycenter(
    member_cost_matrices: list[np.ndarray],
    n_components: int = 6,
    barycenter_cfg: BarycenterConfig | None = None,
) -> dict:
    """Build GW barycenter of a cluster of member attractors.

    Each member cost matrix C_i is a (n_s_i, n_s_i) pairwise distance on
    the FPS subsample of member i's DMAP coord cloud. The barycenter
    B is (n_s, n_s). Member transport plans T_{i→B} are returned so
    callers can project member coords onto barycentric modes.

    Returns dict with keys:
      B_cost      : (n_s, n_s) barycenter intra-distance matrix
      T_plans     : list of (n_s_i, n_s) transport plans, one per member
      d2_members  : list of GW^2 distances from each member to B
      coords_B    : (n_s, n_components) barycentric DMAP coords
      evals_B     : (n_components,) barycentric eigenvalues
      info        : barycenter-solver info (objective, n_restarts, etc.)
    """
    cfg = barycenter_cfg or BarycenterConfig(
        n_supports=40, n_restarts=5, epsilon=0.01,
        max_iter=500, tol=1e-8,
    )
    B_cost, info = gw_barycenter(member_cost_matrices, cfg=cfg)
    T_plans, d2s = member_transport_plans(member_cost_matrices, B_cost, cfg=cfg)
    coords_B, evals_B = dmap_on_cost_matrix(B_cost, n_components=n_components)
    return {
        "B_cost":     B_cost,
        "T_plans":    T_plans,
        "d2_members": d2s,
        "coords_B":   coords_B,
        "evals_B":    evals_B,
        "info":       info,
    }


def bary_state_from_member(
    coord_i_t: np.ndarray,
    support_coords_i: np.ndarray,
    T_i_to_B: np.ndarray,
    coords_B: np.ndarray,
) -> np.ndarray:
    """Project member i's coord at time t onto the barycentric coord system.

    Steps:
      (1) Find nearest FPS-support point p* in member i's support to
          coord_i_t (Euclidean in i's coord space).
      (2) T_i_to_B[p*, :] is a distribution over barycenter support points
          (the member's transport mass from support point p* to each
          barycenter support point). Normalize to a probability.
      (3) Barycentric state at time t for member i = expectation of
          coords_B under that distribution:
              bary_state(t) = (T_i_to_B[p*, :] / sum) @ coords_B

    Returns (n_components,) barycentric coord.
    """
    coord_i_t = np.asarray(coord_i_t, dtype=np.float64).ravel()
    dists = np.linalg.norm(support_coords_i - coord_i_t[None, :], axis=1)
    p_star = int(np.argmin(dists))
    dist = T_i_to_B[p_star, :]
    s = float(dist.sum())
    if s < 1e-12:
        return np.zeros(coords_B.shape[1])
    probs = dist / s
    return probs @ coords_B


def bary_state_timeseries(
    coords_i_timeseries: np.ndarray,
    support_coords_i: np.ndarray,
    T_i_to_B: np.ndarray,
    coords_B: np.ndarray,
) -> np.ndarray:
    """Vectorized: project a full (T, K_i) time series of member i's coords
    onto the barycentric coord system.

    Returns (T, n_components) barycentric state trajectory.
    """
    coords_i_timeseries = np.asarray(coords_i_timeseries, dtype=np.float64)
    T, _ = coords_i_timeseries.shape
    n_comp = coords_B.shape[1]
    out = np.empty((T, n_comp), dtype=np.float64)
    # Nearest-FPS-support-point assignment per time t.
    d2 = cdist(coords_i_timeseries, support_coords_i)
    p_star = np.argmin(d2, axis=1)            # (T,)
    # Lookup T_i_to_B[p*, :] per time → (T, n_s_B)
    dist = T_i_to_B[p_star]                    # (T, n_s_B)
    s = dist.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    probs = dist / s
    out = probs @ coords_B                     # (T, n_comp)
    return out


def cluster_bary_state(
    members_coords: list[np.ndarray],
    members_support_coords: list[np.ndarray],
    T_plans: list[np.ndarray],
    coords_B: np.ndarray,
) -> np.ndarray:
    """Aggregate barycentric state across cluster members at each time.

    All members' time series must be pre-aligned to a common time index
    (callers: handle alignment via pd.Series.reindex before calling).
    Returns (T, n_components) mean-pooled barycentric state.
    """
    if not members_coords:
        raise ValueError("empty cluster")
    per_member = [
        bary_state_timeseries(c, s, T, coords_B)
        for c, s, T in zip(members_coords, members_support_coords, T_plans)
    ]
    stacked = np.stack(per_member, axis=0)     # (M, T, n_comp)
    return stacked.mean(axis=0)                # (T, n_comp) — cluster mean-pool
