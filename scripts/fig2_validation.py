#!/usr/bin/env python3
"""Reproduce Fig 2 -- validation on known dynamical systems.

Draws: row A, the three reference attractors (Lorenz / Rossler / 2-torus);
panel B, the GW distance matrix of 5 Lorenz vs 5 Ornstein-Uhlenbeck
realizations (cross/within ratio ~1.6).

Prints (quoted in the text, not drawn): the D2 invariant check under the
paper's own protocol (AMI tau, FNN dim, Theiler window; delay reconstruction
vs true state), and the feature-based twelve-family / macro ARI numbers from
validation/denoising.py that the discrimination paragraph quotes.

Run:  python3 scripts/fig2_validation.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import figstyle as F
F.use()

from aoa_repro import systems, pipeline, plotting as P, config

P.set_style()


def gw_separation(n_per=5, npts=200):
    """5 Lorenz vs 5 OU; GW on raw scale-normalized delay clouds."""
    clouds = []
    for k in range(n_per):
        dv = pipeline.delay_embed(systems.lorenz(seed=k)[:, 0], dim=3, tau=8)
        idx = np.linspace(0, len(dv) - 1, npts).astype(int)
        clouds.append(pipeline.intra_distance(dv[idx], scale_normalize=True))
    for k in range(n_per):
        dv = pipeline.delay_embed(systems.ornstein_uhlenbeck(seed=k), dim=3, tau=8)
        idx = np.linspace(0, len(dv) - 1, npts).astype(int)
        clouds.append(pipeline.intra_distance(dv[idx], scale_normalize=True))
    G = pipeline.gw_pairwise(clouds)
    grp = np.array([0] * n_per + [1] * n_per)
    within = np.median([G[i, j] for i in range(2*n_per) for j in range(i+1, 2*n_per) if grp[i] == grp[j]])
    cross = np.median([G[i, j] for i in range(2*n_per) for j in range(i+1, 2*n_per) if grp[i] != grp[j]])
    return grp, float(cross / within), G


def ari_by_metric():
    """Real family-recovery ARI per metric from the Atlas synthetic gate."""
    d = config.CHECKPOINTS / "family_discovery" / "synthetic_gate_v4.json"
    import json
    ps = json.load(open(d))["per_seed"]
    out = {}
    for m in ["ari_m1", "ari_m4", "ari_m5"]:
        v = [s[m] for s in ps if m in s]
        out[m] = (float(np.mean(v)), float(min(v)), float(max(v)))
    return out


def main():
    F.use()
    L, Ro, To = systems.lorenz(), systems.rossler(), systems.torus()
    grp, ratio, G = gw_separation()
    print(f"GW cross/within ratio = {ratio:.2f}")

    # family recovery by metric, and on macro. One panel replaces the two MDS
    # scatters: twelve overlapping colours never read at this size, and the claim
    # they support is a single number each.
    import validation.denoising as D
    (d1d, d1r, d4, d5), dlab = D.features_dysts()
    (m1d, m1r, m4), mlab = D.features_macro()
    dys_ari = D.ward_ari(d5, dlab)
    mac_ari = D.ward_ari(m1d, mlab)
    ari = ari_by_metric()
    print(f"dysts ARI (m5) = {dys_ari:.2f}; macro ARI (m1-DMAP) = {mac_ari:.2f}")
    print("ARI by metric:", {k: round(v[0], 3) for k, v in ari.items()})

    # D2 under the paper's own protocol: AMI tau, FNN dim, Theiler window.
    # The comparison of record is delay reconstruction vs the TRUE state under
    # one estimator: that isolates what Takens promises from fit convention.
    from scipy.spatial import cKDTree

    def fnn_dim(x, tau, Rtol=15.0, eta=0.01):
        for dim in range(2, 9):
            dv = pipeline.delay_embed(x, dim=dim, tau=tau)
            dv1 = pipeline.delay_embed(x, dim=dim + 1, tau=tau)
            n = min(len(dv1), len(dv)); dv, dv1 = dv[:n], dv1[:n]
            t = cKDTree(dv); d, j = t.query(dv, k=2)
            nn, dn = j[:, 1], d[:, 1]; ok = dn > 0
            if np.mean(np.abs(dv1[ok, -1] - dv1[nn[ok], -1]) / dn[ok] > Rtol) <= eta:
                return dim
        return 8

    def d2_stats(X, theiler, npts=3000):
        vals = [systems.correlation_dimension(X, npts=npts, seed=s, theiler=theiler)
                for s in range(5)]
        return float(np.mean(vals)), float(np.std(vals))

    d2 = {}
    for name, X, tau in (("Lorenz", systems.lorenz(T=400.0), 16),
                         ("Rossler", systems.rossler(T=2000.0), 27)):
        dim = fnn_dim(X[:, 0], tau)
        dv = pipeline.delay_embed(X[:, 0], dim=dim, tau=tau)
        d2[name] = {"delay": d2_stats(dv, theiler=dim * tau),
                    "true": d2_stats(X, theiler=dim * tau), "dim": dim, "tau": tau}
        print(f"D2 {name}: delay(AMI tau={tau}, FNN dim={dim}, Theiler) = "
              f"{d2[name]['delay'][0]:.2f}+/-{d2[name]['delay'][1]:.2f}  vs  "
              f"true state {d2[name]['true'][0]:.2f}+/-{d2[name]['true'][1]:.2f}")

    fig = plt.figure(figsize=(7.1, 2.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 0.22, 0.82],
                          left=0.005, right=0.925, top=0.86, bottom=0.10,
                          wspace=0.30)

    def attractor(cell, X, sub_idx, name, color, view, letter=None):
        ax = fig.add_subplot(cell, projection="3d")
        P.render_attractor_3d(ax, X, color=color, view=view)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_visible(False)
            a._axinfo["grid"].update(color=F.CONTEXT, linewidth=0.3)
            a.line.set_color(F.INK)
        ax.set_xlabel(r"$x_t$", labelpad=-9)
        ax.set_ylabel(r"$x_{t-\tau}$", labelpad=-9)
        ax.set_zlabel(r"$x_{t-2\tau}$", labelpad=-9)
        F.law_title(ax, name)                       # identity is not inferable from position
        if letter:
            F.group(ax, letter, dx=-0.06, dy=0.94)
        F.sub(ax, sub_idx, dx=0.08, dy=0.90)
        return ax

    # cmr10 has no o-umlaut, so the figure spells Rossler plainly; the caption
    # carries the accent. Invariants are quoted in the text, not drawn here.
    attractor(gs[0, 0], L, 0, "Lorenz", F.BLUE, (22, -60), letter="A")
    attractor(gs[0, 1], Ro, 1, "Rossler", F.VERMILLION, (28, -50))
    attractor(gs[0, 2], To, 2, "2-torus", F.GREEN, (32, -60))

    # --- B: the GW distance matrix (the chaos-vs-stochastic-null contrast) --
    axB = fig.add_subplot(gs[0, 4])
    im = axB.imshow(G, cmap=F.CMAP_SEQ, aspect="equal")
    n2 = len(grp) // 2
    for pos in (n2 - 0.5,):
        axB.axhline(pos, color="w", lw=0.9)
        axB.axvline(pos, color="w", lw=0.9)
    axB.set_xticks([n2 / 2 - 0.5, n2 * 1.5 - 0.5]); axB.set_xticklabels(["Lorenz", "OU"])
    axB.set_yticks([n2 / 2 - 0.5, n2 * 1.5 - 0.5])
    axB.set_yticklabels(["Lorenz", "OU"], rotation=90, va="center")
    axB.tick_params(length=0, labelsize=7)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)
    cb.set_label("GW distance", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    F.panel(axB, "B", dx=-0.22, dy=1.06)

    # No Hellinger-vs-Frobenius family panels: that claim failed verification
    # (ARI <= 0.13 at every operating point; see validation/dysts_families.py).

    from aoa_repro import config
    out = str(config.FIG_OUT / "fig2_validation.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=400)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
