#!/usr/bin/env python3
"""Reproduce Component 5 (Nonlinearity) -- the PSoPS basis is not linearly
recoverable.

Panel (a): DMAP-vs-PCA eigenvalue decay, computed live from the Takens
           embedding of core CPI log-returns (data/fred/fred_CPILFESL.csv).
Panel (b): IAAFT phase-randomized surrogates destroy 24-44% of d_eff across the
           six laws (numbers from the convergence_battery eps008 results).

Writes figures/fig_nonlinearity.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import config, dataio, pipeline, plotting as P
from aoa_repro import figstyle as F
F.use()


def dmap_vs_pca_spectrum(series, dim=8, tau=4, npts=300, n_components=8):
    """Live Takens-embed -> DMAP eigenvalues vs PCA eigenvalues on the same cloud."""
    dv = pipeline.delay_embed(series, dim=dim, tau=tau)
    idx = np.linspace(0, len(dv) - 1, min(npts, len(dv))).astype(int)
    X = dv[idx]
    # DMAP
    _, evs = pipeline.diffusion_coords(X, n_components=n_components, alpha=1.0)
    evs = np.abs(np.asarray(evs, float))
    evs = evs / (evs.max() + 1e-12)
    # PCA on the same delay-cloud (centered)
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    pca_vals = (s ** 2) / max((s ** 2).max(), 1e-12)
    pca_vals = pca_vals[:n_components]
    return evs, pca_vals


# -------------------------------------------------------------- (b) IAAFT
# Per-law fractional d_eff destruction by IAAFT phase-randomized surrogates
# loaded from the convergence-battery iaaft_timeseries result. The value
# we plot is the `relative_change` field, which is (real_d_eff - surr_d_eff)/real_d_eff
# i.e. the fraction of effective dimensionality destroyed by phase randomization.
import glob as _glob
_iaaft_files = sorted(_glob.glob(str(config.CHECKPOINTS / "convergence_battery" /
                                     "iaaft_timeseries_*.json")))
assert _iaaft_files, "no iaaft_timeseries_*.json found in convergence_battery"
iaaft = json.load(open(_iaaft_files[-1]))
IAAFT_PCT, IAAFT_ERR = {}, {}
for law, summ in iaaft["surr_summary"].items():
    rc = float(summ["relative_change"])
    IAAFT_PCT[law] = round(100 * rc)
    # propagate the surrogate spread to the % destroyed: real = mean/(1-rc);
    # error bar = standard error of the mean over the n_reps surrogate d_eff.
    real = summ["mean"] / (1 - rc) if rc < 1 else summ["mean"]
    sem = float(summ["std"]) / np.sqrt(max(int(summ.get("n_reps", 1)), 1))
    IAAFT_ERR[law] = 100 * sem / real if real > 0 else 0.0
# Every row the paper quotes must come from the artifact. An earlier version fell back
# to a hardcoded 44% for Rates, which would silently reproduce a paper number from a run
# that did not contain it.
assert "Rates" in IAAFT_PCT, f"iaaft artifact has no Rates row: {sorted(IAAFT_PCT)}"
print(f"IAAFT d_eff destruction (loaded from {_iaaft_files[-1].split('/')[-1]}):")
for k, v in IAAFT_PCT.items():
    print(f"  {k:10s} {v:>3d}%")


# -------------------------------------------------------------- compose
fig = plt.figure(figsize=(7.1, 3.2))
gs = fig.add_gridspec(1, 2, left=0.05, right=0.99, top=0.86, bottom=0.16, wspace=0.16)

# (a) DMAP-vs-PCA spectrum -- LIVE Takens embed of core CPI (CPILFESL)
dc, c = dataio.load_dated_series(config.DATA_ROOT / "fred" / "fred_CPILFESL.csv")
# work on log-returns to suppress trend
y = np.diff(np.log(c))
evs, pca_vals = dmap_vs_pca_spectrum(y, dim=8, tau=4, npts=300)
print(f"DMAP eigenvalues (normalized): {np.round(evs, 4)}")
print(f"PCA  eigenvalues (normalized): {np.round(pca_vals, 4)}")
ax = fig.add_subplot(gs[0, 0])
ks = np.arange(1, len(evs) + 1)
ax.semilogy(ks, evs, "o-", color=F.BLUE, lw=1.0, ms=5, label="DMAP", clip_on=False)
ax.semilogy(ks[:len(pca_vals)], pca_vals + 1e-16, "s--", color=F.VERMILLION, lw=1.0,
            ms=5, label="PCA", clip_on=False)
ax.set_xlabel("eigenvalue index $k$", fontsize=8)
ax.set_ylabel(r"$|\lambda_k|/\lambda_1$", fontsize=8)
ax.tick_params(labelsize=7)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False,
          edgecolor=F.INK, framealpha=1.0, borderpad=0.35)
F.panel(ax, "A", dx=-0.14, dy=1.02)

# (b) IAAFT destruction bar
ax = fig.add_subplot(gs[0, 1])
laws = list(IAAFT_PCT.keys())
vals = [IAAFT_PCT[k] for k in laws]
errs = [IAAFT_ERR[k] for k in laws]
# colour encodes the surrogate-band test, not an arbitrary threshold: a case is
# highlighted when the real d_eff lies outside the one-sided 99% band of its 10
# surrogate d_eff values (z > 2.326). Under this rule GDP (z = 2.1) is the only
# case inside the band; the other five all have z >= 4.3.
zsc = {}
for law, summ in iaaft["surr_summary"].items():
    rc = float(summ["relative_change"])
    real = summ["mean"] / (1 - rc) if rc < 1 else summ["mean"]
    zsc[law] = (real - summ["mean"]) / max(float(summ["std"]), 1e-12)
print("surrogate-band z-scores:", {k: round(v, 2) for k, v in zsc.items()})
cols = [F.VERMILLION if zsc[k] > 2.326 else F.MUTED for k in laws]
bars = ax.bar(range(len(laws)), vals, yerr=errs, color=cols, edgecolor="none",
              width=0.65, error_kw=dict(lw=0.9, capsize=3.5, ecolor=F.INK))
ax.set_xticks(range(len(laws))); ax.set_xticklabels(laws, fontsize=7)
ax.set_ylabel(r"% of $d_{\mathrm{eff}}$ destroyed by IAAFT", fontsize=7.5)
ax.set_ylim(0, 58); ax.axhline(0, color="grey", lw=0.4)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
F.panel(ax, "B", dx=-0.14, dy=1.02)
P.savefig(fig, "fig_nonlinearity")
print("wrote figures/fig_nonlinearity.pdf")
