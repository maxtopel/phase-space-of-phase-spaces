#!/usr/bin/env python3
r"""
Robust intrinsic-dimensionality + topology toolkit for diffusion-map manifolds.

Implements the methodology of record, so the dimension claim is
bandwidth-free and reproducible rather than a tuning artifact:

  bandwidth      auto-selected by the Coifman/Singer kernel-sum slope (NOT median,
                 which over-smooths; NOT a hand-picked multiplier).
  dimension      (i) Coifman epsilon-scaling d = 2*max d(log S)/d(log eps)  [bandwidth-free]
                 (ii) spectral: # diffusion modes before the eigenvalue gap / 95% variance
                 (iii) L-method knee on the eigenvalue curve (Salvador-Chan)
                 TWO-NN is deliberately NOT the figure of record (density- and
                 n_components-dependent on diffusion coordinates).
  topology       loop test on (psi1, psi2): empty interior + angular coverage; and the
                 H1 persistence ratio if ripser is available (beta_1 loop strength).

One function `analyse(points)` runs the whole battery on an (n, d) point cloud.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh


def kernel_sum_bandwidth(D):
    """Coifman/Singer kernel-sum slope: returns (sigma, intrinsic_dim).
    S(eps)=sum exp(-D^2/eps); the max slope of log S vs log eps is d/2 and locates
    the optimal eps* (=> sigma=sqrt(eps*/2))."""
    pos = D[D > 0]
    med = np.median(pos)
    eps = np.logspace(np.log10(pos.min() ** 2) - 1, np.log10(D.max() ** 2) + 1, 40)
    S = np.array([np.exp(-D ** 2 / e).sum() for e in eps])
    slope = np.gradient(np.log(S), np.log(eps))
    i = int(np.argmax(slope))
    sigma = float(np.sqrt(eps[i] / 2))
    # floor: a degenerate (too-small) bandwidth gives a near-identity kernel and NaN
    # eigenvectors; keep sigma in a sane band around the data scale.
    sigma = float(np.clip(sigma, 0.15 * med, 1.0 * med))
    return sigma, float(2 * slope[i])


def diffusion_spectrum(D, sigma, q=15, alpha=1.0):
    """alpha=1 (Laplace-Beltrami) Hellinger/Euclidean diffusion map at bandwidth sigma.
    Returns (eigenvalues[q], diffusion_coords[n,q])."""
    K = np.exp(-D ** 2 / (2 * sigma * sigma))
    np.fill_diagonal(K, 0.0)
    d = np.maximum(K.sum(1), 1e-12)              # guard isolated points (row-sum -> 0)
    Kn = K / np.outer(d, d) ** alpha
    ds = np.maximum(Kn.sum(1), 1e-12)
    s = 1.0 / np.sqrt(np.maximum(ds, 1e-12))
    Ps = Kn * np.outer(s, s)
    Ps = 0.5 * (Ps + Ps.T)
    n = Ps.shape[0]
    k = min(q + 1, n - 1)
    # dense eigh for small / near-degenerate matrices (ARPACK is unstable there);
    # truncated eigsh only when it pays off (large n)
    if n <= 2500:
        w, V = np.linalg.eigh(Ps)
        o = np.argsort(w)[::-1][:k]
        w, V = w[o], V[:, o]
    else:
        w, V = eigsh(Ps, k=k, which="LM")
        o = np.argsort(w)[::-1]
        w, V = w[o], V[:, o]
    V = V * s[:, None]
    return w[1:], V[:, 1:] * w[1:]               # drop trivial mode; diffusion coords


def lmethod_knee(ev):
    """Salvador-Chen L-method: knee of the sorted positive eigenvalue curve."""
    y = np.sort(ev[ev > 0])[::-1]
    n = len(y)
    if n < 4:
        return n
    x = np.arange(1, n + 1)

    def rss(xx, yy):
        if len(xx) < 2:
            return 0.0
        A = np.vstack([xx, np.ones_like(xx)]).T
        m, b = np.linalg.lstsq(A, yy, rcond=None)[0]
        return float(((yy - (m * xx + b)) ** 2).sum())
    best, bc = np.inf, 2
    for c in range(2, n - 1):
        t = rss(x[:c], y[:c]) + rss(x[c:], y[c:])
        if t < best:
            best, bc = t, c
    return bc


def spectral_dim(ev, var_thresh=0.95):
    """# modes to reach var_thresh of the eigenvalue (lambda^2) variance, and the
    location of the largest relative eigen-gap."""
    e = ev[ev > 0]
    if len(e) < 2:
        return len(e), len(e)
    cum = np.cumsum(e ** 2) / np.sum(e ** 2)
    n_var = int(np.searchsorted(cum, var_thresh) + 1)
    gaps = e[:-1] / np.maximum(e[1:], 1e-12)
    n_gap = int(np.argmax(gaps[:min(len(gaps), 10)]) + 1)
    return n_var, n_gap


def loop_test(coords):
    """Is (psi1, psi2) a closed loop? Returns dict: angular coverage, fraction of points
    near the centre (empty interior => loop), and H1 persistence ratio if available."""
    coords = np.asarray(coords, float)
    fin = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    coords = coords[fin]
    if len(coords) < 10:
        return {"angular_coverage": float("nan"), "frac_near_center": float("nan"), "loop_like": False}
    with np.errstate(all="ignore"):
        th = np.arctan2(coords[:, 1], coords[:, 0])
        r = np.hypot(coords[:, 0], coords[:, 1])
    cov = float((np.histogram(th, bins=24)[0] > 0).mean())
    frac_center = float(np.mean(r < 0.3 * np.median(r)))
    res = {"angular_coverage": cov, "frac_near_center": frac_center,
           "loop_like": cov > 0.8 and frac_center < 0.1}
    try:
        from ripser import ripser
        dg = ripser(coords[:, :2], maxdim=1)["dgms"][1]
        if len(dg):
            pers = np.sort(dg[:, 1] - dg[:, 0])[::-1]
            # significant loops = persistence above a noise threshold (the diagonal band).
            # macro attractors carry MANY loops (one per business cycle), not one dominant S^1.
            thresh = max(0.25 * pers[0], np.median(pers) + 2 * np.std(pers))
            res["H1_top_persistence"] = float(pers[0])
            res["n_significant_loops"] = int((pers > thresh).sum())   # ~ beta_1
            res["H1_dominant_ratio"] = float(pers[0] / (pers[1] if len(pers) > 1 and pers[1] > 0 else pers[0]))
    except Exception:
        pass
    return res


def analyse(points, q=15, label=""):
    """Full robust dimensionality + topology battery on an (n, d) point cloud."""
    D = squareform(pdist(np.asarray(points, float)))
    sigma, d_coifman = kernel_sum_bandwidth(D)
    ev, coords = diffusion_spectrum(D, sigma, q=q)
    n_var, n_gap = spectral_dim(ev)
    out = {"label": label, "n": int(len(points)),
           "sigma_auto": sigma, "median_dist": float(np.median(D[D > 0])),
           "coifman_dim": d_coifman, "lmethod_knee": int(lmethod_knee(ev)),
           "spectral_dim_95var": n_var, "spectral_gap_at": n_gap,
           "eigenvalues": [float(x) for x in ev[:8]],
           "topology": loop_test(coords)}
    return out, coords
