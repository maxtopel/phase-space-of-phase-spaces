#!/usr/bin/env python3
r"""
Robust dimensionality + topology of the macro-law attractors (Phillips, Okun, Solow).

The three canonical laws are joint two-variable systems whose trajectories trace
attractors (Phillips: the unemployment-inflation hysteresis loop, etc.). We analyse
each with the same bandwidth-free battery as the universal PSoPS (aoa_dimension.analyse):
Coifman intrinsic dim, spectral mode count, L-method knee, and the (psi1,psi2) loop /
H1 test. This is the clean, interpretable counterpart to the universal corpus -- the
laws are where the low-dimensional loop topology is sharpest.

Each law's attractor is a joint delay embedding of its standardized variables
[x_t, y_t, x_{t-tau}, y_{t-tau}] (Takens on the coupled system); the raw (x,y) cloud
is analysed too for the bare loop topology.

Writes artifacts/law_dimensionality.json (consumed by scripts/fig7_law_attractors.py,
which renders the paper figure fig_laws.pdf) and a supplement spectrum figure.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import config, figstyle as F
F.use()
from aoa_repro import plotting as _P
_P.set_style()   # paper-wide figure style (Computer Modern, sizes, palette)
from scipy.signal import detrend as _lin_detrend
import pipeline.dimension as AD
import validation.laws_pipeline as L

OUT = os.path.join(_ROOT, "artifacts")
FIG = str(config.FIG_OUT)
TAU = 4


def cyclical(c):
    """Cyclical component: remove the (possibly exponential, hence linear-in-log) growth
    trend so the embedding reflects DYNAMICS, not the trivial common trend. Raw log-levels
    (Solow output & productivity) co-trend and would otherwise look 1-D/collinear."""
    c = np.asarray(c, float)
    m = np.isfinite(c)
    out = np.full_like(c, np.nan)
    out[m] = _lin_detrend(c[m], type="linear")
    return out


def standardize(*cols):
    return [(cyclical(c) - np.nanmean(cyclical(c))) / (np.nanstd(cyclical(c)) + 1e-12) for c in cols]


def joint_embed(X, Y, tau=TAU, dim=2):
    """Joint Takens embedding of the coupled (X,Y) system."""
    n = len(X)
    rows = []
    for t in range((dim - 1) * tau, n):
        row = []
        for k in range(dim):
            row += [X[t - k * tau], Y[t - k * tau]]
        rows.append(row)
    return np.array(rows)


def main():
    laws = {"Phillips": L.phillips_cloud(), "Solow": L.solow_cloud(), "Okun": L.okun_cloud()}
    results = {}
    for name, (d, Xc, Yc, xl, yl) in laws.items():
        X, Y = standardize(np.asarray(Xc, float), np.asarray(Yc, float))
        m = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[m], Y[m]
        emb = joint_embed(X, Y)
        res, coords = AD.analyse(emb, label=name)
        # bare 2-D cloud topology (the literal Phillips-type loop)
        raw = np.c_[X, Y]
        res["raw2d_topology"] = AD.loop_test(raw)
        results[name] = res
        t = res["raw2d_topology"]
        print(f"[{name:9s}] n={res['n']:4d}  Coifman dim={res['coifman_dim']:.2f}  "
              f"L-knee={res['lmethod_knee']}  | raw2d: empty-interior={t['frac_near_center']:.2f} "
              f"significant loops (~beta1)={t.get('n_significant_loops','?')}", flush=True)
    json.dump(results, open(os.path.join(OUT, "law_dimensionality.json"), "w"), indent=2)

    # supplement figure: per-law eigenvalue spectrum only (the paper figure
    # fig_laws.pdf, including these spectra as row D, is drawn by
    # scripts/fig7_law_attractors.py from the JSON this script writes)
    F.use()
    # bootstrapped values, so the figure and Table I quote the same measurement
    _bp = os.path.join(OUT, "table1_bootstrap.json")
    BOOT = json.load(open(_bp)) if os.path.exists(_bp) else {}

    fig = plt.figure(figsize=(7.1, 1.85))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.09, right=0.98, top=0.86, bottom=0.28)
    for j, name in enumerate(F.LAWS):
        r = results[name]
        b_ = BOOT.get(name, {})
        ax2 = fig.add_subplot(gs[0, j])
        ev = np.array(r["eigenvalues"])
        ax2.plot(np.arange(1, len(ev) + 1), ev, "-o", color=F.BLUE, ms=3)
        F.finish(ax2, "Diffusion mode index",
                 r"Eigenvalue $\lambda$" if j == 0 else None, nticks=4)
        if j == 0:
            F.group(ax2, "A", dx=-0.26, dy=1.04)
        F.sub(ax2, j, dx=-0.12, dy=1.02)
        if b_:
            ax2.text(0.97, 0.94, f"$d = {b_['coifman_dim']:.2f}$",
                     transform=ax2.transAxes, ha="right", va="top", fontsize=7,
                     color=F.INK)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_law_dimensionality_supplement.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
