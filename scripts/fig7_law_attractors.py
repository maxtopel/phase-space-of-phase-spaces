#!/usr/bin/env python3
r"""
The single law figure fig_laws.pdf (Results sec.~\ref{subsec:law-attractors}),
four rows by three law columns:

Row A: the canonical law variables as a raw joint cloud from FRED.
Row B: the same joint delay embedding reduced linearly by PCA (the contrast
       that isolates the reduction).
Row C: the same embedding reduced by the diffusion map, where the loops resolve.
Row D: each law's Hellinger-diffusion eigenvalue spectrum, annotated with the
       bootstrapped Coifman dimension from artifacts/table1_bootstrap.json
       (row-D inputs from artifacts/law_dimensionality.json, written by
       validation/law_dimensionality.py, so figure and Table I quote one number).

Phillips (UNRATE, core pi_YoY) monthly, Solow (log GDPC1, log OPHNFB) quarterly,
Okun (Delta UNRATE, GDP YoY) quarterly. rho/PR from atlas nonlinearity.json.

Run: python3 scripts/fig7_law_attractors.py  ->  ../figures/fig_laws.pdf
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, json
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import config, figstyle as F
F.use()
from aoa_repro import config, dataio, pipeline

FS = 9
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FRED = config.DATA_ROOT / "fred"
FIG_OUT = str(config.FIG_OUT / "fig_laws.pdf")


def _align_monthly(da, a, db, b):
    kb = {(d.year, d.month): v for d, v in zip(db, b)}
    out = [(d, av, kb[(d.year, d.month)]) for d, av in zip(da, a) if (d.year, d.month) in kb]
    return ([o[0] for o in out], np.array([o[1] for o in out]), np.array([o[2] for o in out]))


def _align_quarterly(da, a, db, b):
    def q(d): return (d.year, (d.month - 1) // 3)
    kb = {q(d): v for d, v in zip(db, b)}
    out = [(d, av, kb[q(d)]) for d, av in zip(da, a) if q(d) in kb]
    return ([o[0] for o in out], np.array([o[1] for o in out]), np.array([o[2] for o in out]))


def phillips_cloud():
    du, u = dataio.load_dated_series(FRED / "fred_UNRATE.csv")
    dc, c = dataio.load_dated_series(FRED / "fred_CPILFESL.csv")
    pi_d, pi = dc[12:], 100 * (np.log(c[12:]) - np.log(c[:-12]))
    d, U, P = _align_monthly(du, u, pi_d, pi)
    return d, U, P, "Unemployment (%)", r"Core inflation $\pi_{\mathrm{YoY}}$ (%)"


def solow_cloud():
    dg, g = dataio.load_dated_series(FRED / "fred_GDPC1.csv")
    do, o = dataio.load_dated_series(FRED / "fred_OPHNFB.csv")
    d, G, O = _align_quarterly(dg, g, do, o)
    return d, np.log(G), np.log(O), r"$\log$ Real GDP", r"$\log$ Labour productivity"


def okun_cloud():
    dg, g = dataio.load_dated_series(FRED / "fred_GDPC1.csv")
    du, u = dataio.load_dated_series(FRED / "fred_UNRATE.csv")
    # quarterly UNRATE
    qd, qu = {}, {}
    for d, v in zip(du, u):
        qd.setdefault((d.year, (d.month-1)//3), []).append(v)
    qkeys = sorted(qd); U = np.array([np.mean(qd[k]) for k in qkeys])
    # GDP YoY
    d, G, _ = _align_quarterly(dg, g, dg, g)
    Gyoy = 100*(np.log(g[4:]) - np.log(g[:-4]))
    gy_d = dg[4:]
    def q(dd): return (dd.year, (dd.month-1)//3)
    kg = {q(dd): v for dd, v in zip(gy_d, Gyoy)}
    out_d, dU, GR = [], [], []
    Uq = {k: np.mean(qd[k]) for k in qkeys}
    prevk = None
    for k in qkeys:
        if k in kg and prevk is not None and prevk in Uq:
            out_d.append(k[0] + k[1]*0.25); dU.append(Uq[k]-Uq[prevk]); GR.append(kg[k])
        prevk = k
    return np.array(out_d), np.array(dU), np.array(GR), r"$\Delta$ Unemployment (%)", r"GDP growth, YoY (%)"


def years_of(d):
    if len(d) and hasattr(d[0], "year"):
        return np.array([dd.year + (dd.month-1)/12 for dd in d])
    return np.asarray(d, float)


def _boot_loops(X, Y, n_boot=40, frac=0.85, seed=0):
    """Bootstrap beta_1 (number of significant persistent loops) of the cyclical-detrended
    (x,y) attractor: subsample frac of the points without replacement, recount loops, repeat.
    Returns (mean, sd) over resamples -- the proper-architecture loop count with uncertainty."""
    import pipeline.dimension as AD
    from scipy.signal import detrend as _ld
    x = np.asarray(X, float); y = np.asarray(Y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    x = _ld(x, type="linear"); y = _ld(y, type="linear")
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    raw = np.c_[x, y]
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(raw), int(frac * len(raw)), replace=False))
        t = AD.loop_test(raw[idx])
        counts.append(t.get("n_significant_loops", np.nan))
    counts = np.array(counts, float); counts = counts[np.isfinite(counts)]
    return (float(np.mean(counts)), float(np.std(counts))) if len(counts) else (float("nan"), 0.0)


def main():
    nl = json.load(open(config.RESULTS_V3 / "nonlinearity.json"))
    nl = {row["law"]: row for row in nl}
    # beta_1 from the new cyclical-detrended per-law loop test (bootstrapped here), NOT the
    # stale homology_barycenter field, which under-counted business-cycle loops.
    ld_path = os.path.join(_ROOT, "artifacts", "law_dimensionality.json")
    ld = {k.lower(): v for k, v in json.load(open(ld_path)).items()}
    panels = [("phillips", "Phillips", phillips_cloud()),
              ("okun", "Okun", okun_cloud()),
              ("solow", "Solow", solow_cloud())]
    # shared absolute-year color scale across all panels (same colour = same year)
    allyr = np.concatenate([years_of(p[2][0]) for p in panels])
    gmin, gmax = float(allyr.min()), float(allyr.max())

    F.use()
    # single graphic, four rows: raw plane / PCA / diffusion map / spectrum.
    # Row D reads artifacts/law_dimensionality.json + table1_bootstrap.json
    # so figure and Table I quote the same measurement.
    _bp = os.path.join(_ROOT, "artifacts", "table1_bootstrap.json")
    BOOT = json.load(open(_bp)) if os.path.exists(_bp) else {}
    fig = plt.figure(figsize=(7.1, 6.75))
    gs = fig.add_gridspec(4, 3, left=0.09, right=0.855, top=0.955, bottom=0.055,
                          hspace=0.62, wspace=0.38, height_ratios=[1.05, 1, 1, 0.66])
    row_axes = {"A": [], "B": [], "C": [], "D": []}

    def _bare3d(ax, sub_idx):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_visible(False)
            a._axinfo["grid"].update(color=F.CONTEXT, linewidth=0.3)
            a.line.set_color(F.INK); a.line.set_linewidth(0.6)
        ax.view_init(elev=22, azim=-60)
        ax._sub_idx = sub_idx

    for j, (key, title, (d, X, Y, xl, yl)) in enumerate(panels):
        yr = years_of(d)
        X = np.asarray(X, float); Y = np.asarray(Y, float)

        # Row A: the raw joint plane
        axA = fig.add_subplot(gs[0, j])
        axA.plot(X, Y, color=F.CONTEXT, lw=0.3, alpha=0.5, zorder=1)
        axA.scatter(X, Y, c=yr, s=8, cmap=F.CMAP_SEQ, vmin=gmin, vmax=gmax,
                    linewidths=0, zorder=2)
        F.finish(axA, xl, yl, nticks=4)
        F.law_title(axA, title)
        axA._sub_idx = j; row_axes["A"].append(axA)

        # Row B: the SAME delay embedding reduced linearly, by PCA. Without this the
        # claim that the geometry is nonlinear is asserted rather than shown: the
        # comparison has to hold the embedding fixed and vary only the reduction.
        emb = pipeline.takens_joint_raw([X, Y]) if hasattr(pipeline, "takens_joint_raw") \
            else np.column_stack([X[:-8], Y[:-8], X[8:], Y[8:]])
        E = (emb - emb.mean(0)) / (emb.std(0) + 1e-12)
        _, _, Vt = np.linalg.svd(E - E.mean(0), full_matrices=False)
        pca3 = (E - E.mean(0)) @ Vt[:3].T
        axB = fig.add_subplot(gs[1, j], projection="3d")
        cyr_p = np.linspace(yr.min(), yr.max(), len(pca3))
        axB.scatter(pca3[:, 0], pca3[:, 1], pca3[:, 2], c=cyr_p, s=5, cmap=F.CMAP_SEQ,
                    vmin=gmin, vmax=gmax, linewidths=0, alpha=0.85, depthshade=False)
        axB.set_xlabel("PC 1", labelpad=-9); axB.set_ylabel("PC 2", labelpad=-9)
        axB.set_zlabel("PC 3", labelpad=-9)
        _bare3d(axB, j)
        row_axes["B"].append(axB)

        # Row C: the nonlinear reduction of the same embedding
        psi = pipeline.takens_joint_dmap([X, Y], n_components=5, n_top_coords=3)
        axC = fig.add_subplot(gs[2, j], projection="3d")
        cyr = np.linspace(yr.min(), yr.max(), len(psi))
        axC.scatter(psi[:, 0], psi[:, 1], psi[:, 2], c=cyr, s=5, cmap=F.CMAP_SEQ,
                    vmin=gmin, vmax=gmax, linewidths=0, alpha=0.85, depthshade=False)
        axC.set_xlabel(r"$\Psi_1$", labelpad=-9); axC.set_ylabel(r"$\Psi_2$", labelpad=-9)
        axC.set_zlabel(r"$\Psi_3$", labelpad=-9)
        _bare3d(axC, j)
        row_axes["C"].append(axC)

        # rho / PR / beta_1 are tabulated in Table law-geometry; annotating them
        # under every panel duplicated the table and crowded the figure.
        r = nl[key]
        rho = r["rho_geodesic_over_euclidean"]; PR = r["participation_ratio"]
        b1_mean, b1_sd = _boot_loops(X, Y, seed=j)
        print(f"{title}: n={len(X)} rho={rho:.2f} PR={PR:.2f} "
              f"beta1={b1_mean:.1f}+/-{b1_sd:.1f}", flush=True)

        # Row D: the Hellinger-diffusion eigenvalue spectrum, annotated with the
        # bootstrapped Coifman dimension so figure and Table I quote one number.
        axD = fig.add_subplot(gs[3, j])
        ev = np.array(ld[key]["eigenvalues"])
        axD.plot(np.arange(1, len(ev) + 1), ev, "-o", color=F.BLUE, ms=3)
        F.finish(axD, "Diffusion mode index",
                 r"Eigenvalue $\lambda$" if j == 0 else None, nticks=4)
        b_ = BOOT.get(title, {})
        if b_:
            axD.text(0.97, 0.94, f"$d = {b_['coifman_dim']:.2f}$",
                     transform=axD.transAxes, ha="right", va="top", fontsize=7,
                     color=F.INK)
        axD._sub_idx = j; row_axes["D"].append(axD)

    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(gmin, gmax), cmap=F.CMAP_SEQ)
    sm.set_array([])
    cax = fig.add_axes([0.935, 0.30, 0.013, 0.62])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("Year", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)

    # aligned letters: one x for all rows, subpanel tags at each panel corner
    fig.canvas.draw()
    xL = 0.012
    # one x per COLUMN for the roman tags, anchored on row A, so (i)/(ii)/(iii)
    # stack vertically no matter how each row's axes are inset
    colx = {a._sub_idx: a.get_position().x0 + 0.002 for a in row_axes["A"]}
    for letter in ("A", "B", "C", "D"):
        axes = row_axes[letter]
        ytop = max(a.get_position().y1 for a in axes)
        fig.text(xL, ytop + 0.006, letter, fontsize=10, fontweight="bold",
                 color=F.INK, ha="left", va="bottom")
        for a in axes:
            b = a.get_position()
            fig.text(colx[a._sub_idx], b.y1 + 0.003, f"({F.ROMANS[a._sub_idx]})",
                     fontsize=8.5, color=F.INK, ha="left", va="bottom")
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig("/tmp/fig_laws.png", dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.normpath(FIG_OUT), flush=True)


if __name__ == "__main__":
    main()
