"""Thin wrappers over the vendored Atlas machinery + optimal transport.

Every figure that embeds, diffusion-maps, or GW-aligns goes through here, so the
pipeline is implemented ONCE. The heavy lifting lives in _vendor/{embed,diffuse,
transport}.py (verbatim copies of the Atlas modules).
"""
import numpy as np
from scipy.spatial.distance import cdist

from . import config
config.use_vendor()
import embed as _embed        # noqa: E402  (vendored)
import diffuse as _diffuse    # noqa: E402  (vendored)


# ---------------------------------------------------------------- embedding
def standardize(series):
    s = np.asarray(series, float)
    return (s - s.mean()) / (s.std() + 1e-9)


def delay_embed(series, dim=3, tau=8, standardized=True):
    """Takens delay-coordinate cloud (Atlas embed.make_delay_vectors)."""
    s = standardize(series) if standardized else np.asarray(series, float)
    return _embed.make_delay_vectors(s, dim=dim, tau=tau)


def select_tau(series):
    """First AMI minimum (Atlas embed.select_tau)."""
    return _embed.select_tau(np.asarray(series, float))


# ---------------------------------------------------------------- diffusion maps
def diffusion_coords(X, n_components=5, alpha=1.0):
    """Per-series diffusion map coordinates (Atlas diffuse.dmap_dense).

    Returns (coords, eigenvalues).
    """
    return _diffuse.dmap_dense(X, n_components=n_components, alpha=alpha)


def intra_distance(coords, scale_normalize=True):
    """Intra-cloud Euclidean distance matrix (the mm-space metric for GW)."""
    C = cdist(coords, coords)
    if scale_normalize:
        med = np.median(C[C > 0]) + 1e-12
        C = C / med
    return C


# ---------------------------------------------------------------- transport
def gw_distance(C1, C2, p=None, q=None, loss="square_loss"):
    """Gromov--Wasserstein distance between two metric-measure clouds."""
    import ot
    p = np.ones(len(C1)) / len(C1) if p is None else p
    q = np.ones(len(C2)) / len(C2) if q is None else q
    gw2 = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss)
    return float(np.sqrt(max(gw2, 0.0)))


def gw_pairwise(clouds):
    """Symmetric pairwise GW distance matrix for a list of distance matrices."""
    n = len(clouds)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            G[i, j] = G[j, i] = gw_distance(clouds[i], clouds[j])
    return G


def hellinger_distance(T1, T2):
    """Hellinger distance between two (transport-plan) distributions."""
    return float(np.linalg.norm(np.sqrt(np.maximum(T1, 0)) - np.sqrt(np.maximum(T2, 0))))


def persistent_h1(X, noise_floor=None, life_frac=0.05):
    """Vietoris-Rips persistent homology in dim 1 (life-fraction noise model).

    Returns (dgm, beta_1, max_persistence). beta_1 counts H1 bars with lifetime
    above either an explicit noise_floor, or, lacking that, life_frac * max_life.
    """
    from ripser import ripser
    dgm = ripser(X, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return dgm, 0, 0.0
    life = dgm[:, 1] - dgm[:, 0]
    mx = float(life.max())
    if mx < 1e-12:
        return dgm, 0, mx
    thr = noise_floor if noise_floor is not None else life_frac * mx
    return dgm, int((life > thr).sum()), mx


def audit_persistent_h1(X, noise_frac=0.05):
    """Persistent H1 with the atlas-script-26 noise model.

    Threshold = noise_frac * max(birth_times). This is the audit's exact
    convention; ported verbatim from
    `atlas/research/core/AoA_v3.0/scripts/26_phillips_hysteresis_audit.py`
    function `_persistent_beta1`.

    Returns (dgm, beta_1, max_persistence).
    """
    from ripser import ripser
    if len(X) < 10:
        return np.zeros((0, 2)), 0, 0.0
    dgm = ripser(X, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return dgm, 0, 0.0
    births = dgm[:, 0]; deaths = dgm[:, 1]
    pers = deaths - births
    max_birth = float(np.nanmax(births)) if len(births) else 0.0
    thr = noise_frac * max(max_birth, 1e-12)
    return dgm, int(np.sum(pers > thr)), float(np.nanmax(pers))


def takens_joint_dmap(series_list, n_components=5, alpha=1.0,
                      n_top_coords=3, auto_select=True,
                      dim=8, tau=4):
    """Atlas-script-26 joint Phillips construction (auto-select by default).

    For each scalar `series` in `series_list`: AMI-select tau, FNN-select dim,
    build raw delay vectors, row-normalize, truncate to common length,
    column-concatenate, then DMAP the joint with Theiler window = max(tau).
    Returns the first `n_top_coords` diffusion coordinates.

    Setting `auto_select=False` uses the explicit `dim`/`tau` instead.
    """
    embs, taus = [], []
    for s in series_list:
        s = np.asarray(s, float)
        if auto_select:
            t = _embed.select_tau(s)
            d = _embed.select_dim(s, t)
            dv = _embed.make_raw_delay_vectors(s, d, t)
            dv = _embed.standardize_rows(dv)
            taus.append(t)
        else:
            dv = _embed.make_delay_vectors(s, dim=dim, tau=tau)
            taus.append(tau)
        embs.append(dv)
    n = min(len(e) for e in embs)
    joint = np.concatenate([e[-n:] for e in embs], axis=1)
    coords, _ = _diffuse.dmap_dense(joint, n_components=n_components, alpha=alpha,
                                    theiler=max(taus))
    return coords[:, :n_top_coords]


def hellinger_dmap(plans_flat, n_components=10):
    """Hellinger-metric diffusion map over a stack of flattened transport plans.

    plans_flat : (N, n_s*n_s). Returns (coords, eigenvalues, spectral_gap).
    Mirrors the Atlas interpretability pipeline (sqrt embed -> Hellinger -> DMAP).
    """
    sqrt_p = np.sqrt(np.maximum(plans_flat, 1e-15))
    H = cdist(sqrt_p, sqrt_p)
    bw = np.median(H[H > 0])
    K = np.exp(-H**2 / (2 * bw**2))
    np.fill_diagonal(K, 0.0)
    d = K.sum(1)
    d_inv = 1.0 / np.maximum(d, 1e-10)
    K = K * np.outer(d_inv, d_inv)              # alpha = 1 normalization
    rs = K.sum(1)
    P = K / np.maximum(rs[:, None], 1e-10)
    P = 0.5 * (P + P.T)
    ev, evec = np.linalg.eigh(P)
    idx = np.argsort(ev)[::-1]
    ev, evec = ev[idx], evec[:, idx]
    nc = min(n_components, len(ev) - 1)
    coords = evec[:, 1:nc+1] * ev[1:nc+1]
    gap = float(ev[1] / ev[2]) if len(ev) > 2 else float("nan")
    return coords, ev[1:nc+1], gap
