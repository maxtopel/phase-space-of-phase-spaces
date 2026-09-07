#!/usr/bin/env python3
"""Reproduce Component 4 (macro laws) -- live per-law joint attractors built
from raw FRED data, annotated with the published nonlinearity diagnostics.

Each panel is the joint cloud of the law's canonical variables, drawn
directly from the FRED CSVs (`data/fred/` in this repo):
   Phillips : (UNRATE, pi_YoY) monthly,    1957-2025
   Solow    : (log GDPC1, log OPHNFB) quarterly, 1947-2025
   Okun     : (Delta UNRATE, GDP YoY growth) quarterly, 1948-2025

Trajectories are colored by year. The ρ_geodesic/Euclidean, PR, and β_1
annotations underneath each panel are loaded live from atlas
`nonlinearity.json` (the atlas pipeline at law scope) -- the geometric
diagnostic numbers are the headline content; the panels are the
geometric carriers.

Writes figures/fig_macro_laws.pdf. Replaces the previous version which
rasterized atlas script-02 PDFs.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from aoa_repro import config, dataio, plotting as P
P.set_style()


# -------------------------------------------------------------- helpers
def _align_monthly(d_a, a, d_b, b):
    """Inner-join two monthly series on (year, month). Returns aligned arrays + dates."""
    key_a = {(d.year, d.month): v for d, v in zip(d_a, a)}
    out_d, out_a, out_b = [], [], []
    for d, v in zip(d_b, b):
        k = (d.year, d.month)
        if k in key_a:
            out_d.append(d); out_a.append(key_a[k]); out_b.append(v)
    return np.array(out_d), np.array(out_a, float), np.array(out_b, float)


def _align_quarterly(d_a, a, d_b, b):
    """Inner-join two quarterly series on (year, quarter)."""
    def q(d): return (d.year, (d.month - 1) // 3)
    key_a = {q(d): v for d, v in zip(d_a, a)}
    out_d, out_a, out_b = [], [], []
    for d, v in zip(d_b, b):
        if q(d) in key_a:
            out_d.append(d); out_a.append(key_a[q(d)]); out_b.append(v)
    return np.array(out_d), np.array(out_a, float), np.array(out_b, float)


# -------------------------------------------------------------- laws
FRED = config.DATA_ROOT / "fred"

def phillips_cloud():
    du, u = dataio.load_dated_series(FRED / "fred_UNRATE.csv")
    dc, c = dataio.load_dated_series(FRED / "fred_CPILFESL.csv")
    pi_d, pi = dc[12:], 100 * (np.log(c[12:]) - np.log(c[:-12]))
    d, U, P = _align_monthly(du, u, pi_d, pi)
    return d, U, P, "UNRATE (%)", r"$\pi_{\mathrm{YoY}}^{\mathrm{core}}$ (%)"

def solow_cloud():
    dg, g = dataio.load_dated_series(FRED / "fred_GDPC1.csv")
    do, o = dataio.load_dated_series(FRED / "fred_OPHNFB.csv")
    d, G, O = _align_quarterly(dg, g, do, o)
    return d, np.log(G), np.log(O), r"$\log$ GDPC1", r"$\log$ OPHNFB (labour productivity)"

def okun_cloud():
    dg, g = dataio.load_dated_series(FRED / "fred_GDPC1.csv")
    du, u = dataio.load_dated_series(FRED / "fred_UNRATE.csv")
    # quarterly UNRATE = mean over the three months of each quarter
    by_q = {}
    for di, ui in zip(du, u):
        q = (di.year, (di.month - 1) // 3)
        by_q.setdefault(q, []).append(ui)
    dq_u = sorted(by_q.keys())
    u_q = np.array([np.mean(by_q[k]) for k in dq_u], float)
    # GDPC1 quarterly growth (YoY %)
    G = g
    Gyoy = 100 * (np.log(G[4:]) - np.log(G[:-4]))
    Gdates = dg[4:]
    # align Gyoy and u_q quarterly
    g_idx = {(d.year, (d.month - 1) // 3): v for d, v in zip(Gdates, Gyoy)}
    out_d, U, GR = [], [], []
    for k in dq_u:
        if k in g_idx:
            out_d.append(datetime(k[0], 1 + 3 * k[1], 1))
            U.append(u_q[dq_u.index(k)])
            GR.append(g_idx[k])
    U = np.array(U, float); GR = np.array(GR, float)
    dU = np.concatenate([[0.0], np.diff(U)])
    return np.array(out_d), dU, GR, r"$\Delta$ UNRATE (%, q/q)", r"GDP YoY growth (%)"


def main():
    """Build fig_macro_laws.pdf. Wrapped in a function (2026-09-04): this used
    to run at import time, so importing the *_cloud() helpers from other scripts
    silently regenerated the figure as a side effect."""
    # -------------------------------------------------------------- nonlinearity metrics
    nl = json.load(open(config.RESULTS_V3 / "nonlinearity.json"))
    M = {row["law"]: row for row in nl}


    # -------------------------------------------------------------- figure
    fig = plt.figure(figsize=(7.1, 3.1))
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.88, bottom=0.18, wspace=0.32)

    panels = [("phillips", "Phillips", phillips_cloud()),
              ("solow",    "Solow",    solow_cloud()),
              ("okun",     "Okun",     okun_cloud())]

    for j, (key, title, (d, x, y, xlab, ylab)) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        years = np.array([di.year + (di.month - 1) / 12 for di in d])
        sc = ax.scatter(x, y, c=years, s=8, cmap="viridis", linewidths=0, alpha=0.85)
        ax.plot(x, y, color="#9aa6b2", lw=0.25, alpha=0.45, zorder=0)
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.margins(0.02)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        r = M[key]
        rho = r["rho_geodesic_over_euclidean"]
        PR  = r["participation_ratio"]
        b1_b = r.get("homology_barycenter", {}).get("beta_1", "--")
        b1_p = r.get("homology_pca_projection", {}).get("beta_1", "--")
        n = len(x)
        ax.set_title(f"({chr(ord('a')+j)}) {title}  ($n={n}$)", fontsize=8.4, pad=2)
        ax.text(0.5, -0.32, rf"$\rho={rho:.2f}\quad\mathrm{{PR}}={PR:.2f}\quad"
                           rf"\beta_1^{{\mathrm{{PSoPS}}}}={b1_b}$ (PCA $\beta_1={b1_p}$)",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color=P.INK)
        print(f"{title:9s}  n={n}  rho={rho:.2f}  PR={PR:.2f}  beta1_AoA={b1_b}  beta1_PCA={b1_p}")

    # small colorbar for year
    cax = fig.add_axes([0.99, 0.20, 0.008, 0.55])
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=years.min(), vmax=years.max())
    cbar = mpl.colorbar.ColorbarBase(cax, cmap="viridis", norm=norm, orientation="vertical")
    cbar.set_label("year", fontsize=6.5); cbar.ax.tick_params(labelsize=5.5)

    fig.suptitle("Macroeconomic laws as joint attractors from raw FRED data",
                 fontsize=9.5, weight="bold", y=0.99)
    fig.text(0.5, 0.02,
             r"$\rho$, PR, $\beta_1$ from atlas nonlinearity.json (law-scoped script 02, $N\!\approx\!60$).",
             ha="center", va="bottom", fontsize=6.2, color=P.GREY, style="italic")

    P.savefig(fig, "fig_macro_laws")
    print("wrote figures/fig_macro_laws.pdf")


if __name__ == "__main__":
    main()
