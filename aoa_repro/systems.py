"""Reference dynamical systems + invariants, shared across figures.

Used by the pipeline schematic (Fig 1) and the validation figure (Fig 2).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------- systems
def lorenz(T=80.0, dt=0.01, s=10.0, r=28.0, b=8.0/3.0, ic=(1., 1., 1.),
           seed=None, burn=1000):
    """Lorenz attractor trajectory; returns (N,3). seed perturbs the IC.

    `burn` is the number of leading samples discarded as transient. It is
    clipped to at most half the trajectory length so that a short `T` does
    not silently return an empty array.
    """
    ic = np.asarray(ic, float)
    if seed is not None:
        ic = ic + 0.01 * np.random.default_rng(seed).standard_normal(3)

    def f(t, u):
        x, y, z = u
        return [s * (y - x), x * (r - z) - y, x * y - b * z]
    t = np.arange(0, T, dt)
    sol = solve_ivp(f, (0, T), list(ic), t_eval=t, rtol=1e-9, atol=1e-9)
    n = sol.y.shape[1]
    burn = min(int(burn), max(0, n - 100))
    return sol.y.T[burn:]


def rossler(T=400.0, dt=0.05, a=0.2, b=0.2, c=5.7, burn=200):
    """Rossler attractor trajectory; returns (N,3). burn-in samples discarded."""
    def f(t, u):
        x, y, z = u
        return [-y - z, x + a * y, b + z * (x - c)]
    t = np.arange(0, T, dt)
    sol = solve_ivp(f, (0, T), [1., 1., 1.], t_eval=t, rtol=1e-9, atol=1e-9)
    n = sol.y.shape[1]
    burn = min(int(burn), max(0, n - 100))
    return sol.y.T[burn:]


def torus(n=6000, R=2.0, r=0.8, w1=1.0, w2=np.sqrt(2.0), T=200.0):
    """Quasiperiodic 2-torus (incommensurate frequencies) embedded in R^3."""
    t = np.linspace(0, T, n)
    th, ph = w1 * t, w2 * t
    x = (R + r * np.cos(th)) * np.cos(ph)
    y = (R + r * np.cos(th)) * np.sin(ph)
    z = r * np.sin(th)
    return np.column_stack([x, y, z])


def ornstein_uhlenbeck(n=3000, theta=0.5, sigma=1.0, dt=0.05, seed=0):
    """Linear-Gaussian null process (the negative control for chaos tests)."""
    rng = np.random.default_rng(100 + seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i-1] - theta * x[i-1] * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return x


# ---------------------------------------------------------------- invariants
C_FLOOR, C_CEIL = 0.005, 0.25   # the correlation integral scales below saturation


def correlation_dimension(X, npts=1500, seed=0, width=8, max_spread=0.12,
                          return_quality=False, theiler=0):
    """Grassberger--Procaccia correlation dimension D2, fitted on the scaling region.

    The previous version fitted log C(r) against log r over a fixed 10th-45th percentile
    band of the pairwise distances. That band runs past the end of the scaling region and
    into saturation, where the slope is falling toward zero, so the fit was dragged low:
    on the Lorenz attractor it returned 1.77 against a literature D2 of 2.05, while the
    local slope sits on a clean plateau at 1.92 +/- 0.01 between the 3rd and 10th
    percentile.

    D2 only exists where log C(r) is straight. This locates that stretch instead of
    assuming it: the flattest window of `width` consecutive local slopes. The spread
    within that window measures whether a scaling region exists at all, which is a real
    question rather than a formality. On 1600-dimensional Hellinger plan vectors,
    distances concentrate and no plateau forms; the function returns nan there instead of
    reporting the width of the concentration jump as a dimension.

    `theiler` excludes pairs closer than that many samples in TIME (Theiler window,
    \\cite{theiler1986spurious}): temporally adjacent points on one trajectory strand
    are dynamically, not geometrically, close and bias the correlation integral.

    Returns D2, or (D2, spread) when return_quality is set.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if len(X) > npts:
        idx = np.sort(rng.choice(len(X), npts, replace=False))
        X = X[idx]
    if theiler > 0:
        D = squareform(pdist(X))
        dt = np.abs(idx[:, None] - idx[None, :])
        d = D[dt > theiler]
        d = d[d > 0]
    else:
        d = pdist(X)
        d = d[d > 0]
    if len(d) < 100:
        return (float("nan"), float("inf")) if return_quality else float("nan")
    rs = np.logspace(np.log10(np.percentile(d, 0.5)),
                     np.log10(np.percentile(d, 90)), 40)
    C = np.array([(d < rr).mean() for rr in rs])
    m = C > 1e-5
    if m.sum() < width + 2:
        return (float("nan"), float("inf")) if return_quality else float("nan")
    sl = np.gradient(np.log(C[m]), np.log(rs[m]))
    # The saturation tail is also flat, and flatter than the true scaling region, so
    # "flattest window" alone selects it: on the Okun attractor it chose 39-69% of pairs
    # and returned 1.07 against a Coifman dimension of 2.40. The scaling region of a
    # correlation integral lives at small r, before C(r) begins to saturate, so the
    # search is confined there.
    Cm = C[m]
    ok = [j for j in range(len(sl) - width) if Cm[j + width] <= C_CEIL and Cm[j] >= C_FLOOR]
    if not ok:
        return (float("nan"), float("inf")) if return_quality else float("nan")
    i = min(ok, key=lambda j: sl[j:j + width].std())
    d2, spread = float(sl[i:i + width].mean()), float(sl[i:i + width].std())
    if spread > max_spread:                 # no straight stretch: D2 is not defined here
        d2 = float("nan")
    return (d2, spread) if return_quality else d2


# beta_1 / persistent-H1 helper lives in aoa_repro.pipeline.persistent_h1 --
# it's the single canonical implementation used by Components 2 and 4.
