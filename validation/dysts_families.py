#!/usr/bin/env python3
r"""
PSoPS on a controlled zoo of chaotic systems (Results sec.~\ref{sec:validation}).

Note: build the Phase Space of Phase Spaces for a dozen attractor *families*
from `dysts` and test whether the transport-plan geometry groups realizations
by family. MEASURED RESULT (2026-09-04): it does not -- ARI_H <= 0.13 for eps
in [0.05, 0.2] under both global and per-matrix cost normalization, with
ARI_F similar. The paper accordingly does NOT claim transport-plan family
recovery; the Hellinger-vs-Frobenius choice is carried by the resolution
argument and the macro-corpus near-duplicate evidence instead.

Pipeline for clean low-dimensional simulated systems:
  x-observable -> Takens delay embedding (RAW; the per-series diffusion map is a
  denoising step for noisy real data and is SKIPPED here, since it washes out the
  discriminating geometry of clean attractors) -> FPS to n_s supports -> cost
  matrix -> GW barycenter -> per-attractor transport plan T_i -> Hellinger-DMAP
  on {T_i} -> PSoPS coordinates.

eps and n_s are tuned for THIS use case (the macro 0.008/40 operating point does
not transfer and does not even converge on raw delay-coordinate matrices). We
sweep eps and report; the figure uses the best.

Deps: dysts + atlas aoa_v3 + scikit-learn + scipy + matplotlib.
Run:     python3 validation/dysts_families.py
Outputs: prints sweep + ARI; writes ../figures/fig_dysts_families.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, sys
import numpy as np
from aoa_repro import config
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# vendored atlas modules (see vendor_atlas/); private tree no longer needed
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "vendor_atlas"))
from aoa_v3.embed import embed_series, EmbedConfig
from aoa_v3.barycenter import gw_barycenter, member_transport_plans, BarycenterConfig, fps_landmarks

FS = 9
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------- PARAMETERS
SEED       = 0
FAMILIES   = ["Lorenz", "Rossler", "Chen", "Chua", "Aizawa", "Halvorsen",
              "Thomas", "Dadras", "RabinovichFabrikant", "Rucklidge",
              "Lorenz84", "Bouali2"]
N_REP      = 8
N_PTS      = 2000
NS         = 40            # transport-plan support size
N_LAND     = 3             # landmark attractors per family for the barycenter
PARAM_JIT  = 0.05
EPS_SWEEP  = [0.05, 0.1, 0.2]   # tuned for the raw-Takens scale
MAXIT      = 300
CACHE      = os.path.join(_ROOT, "caches", "cache_dysts_costs.npz")
FIG_OUT    = str(config.FIG_OUT / "fig_dysts_families.pdf")
rng = np.random.default_rng(SEED)


def realization(name):
    import dysts.flows as F
    sys_ = getattr(F, name)()
    for k, v in list(sys_.params.items()):
        if isinstance(v, (int, float)):
            setattr(sys_, k, float(v) * (1.0 + PARAM_JIT * rng.standard_normal()))
    sys_.ic = np.asarray(sys_.ic, float) * (1.0 + 0.05 * rng.standard_normal(np.shape(sys_.ic)))
    tr = sys_.make_trajectory(N_PTS, resample=True)
    if tr is None or len(tr) < 500 or not np.all(np.isfinite(tr)):
        return None
    return tr[:, 0]


def raw_cost(x):
    """RAW Takens delay-coordinate cost matrix (no diffusion map)."""
    dv, dim, tau = embed_series(x, cfg=EmbedConfig())      # standardized delay vectors
    if dv.shape[0] < NS:
        return None
    sel = fps_landmarks(cdist(dv, dv), n_landmarks=NS, seed=0)
    return cdist(dv[sel], dv[sel])


def build_costs():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"loaded {len(z['costs'])} cached raw attractors", flush=True)
        return list(z["costs"]), np.asarray(z["labels"]), list(z["names"])
    costs, labels, names = [], [], []
    for fi, fam in enumerate(FAMILIES):
        got = 0
        for _ in range(N_REP):
            try:
                x = realization(fam)
                if x is None:
                    continue
                C = raw_cost(x)
                if C is None:
                    continue
                costs.append(C); labels.append(fi); names.append(fam); got += 1
            except Exception:
                continue
        print(f"  {fam:20s}: {got}/{N_REP}", flush=True)
    labels = np.array(labels)
    np.savez(CACHE, costs=np.array(costs), labels=labels, names=np.array(names))
    return costs, labels, names


def dmap_on(D, ncomp=6):
    bw = np.median(D[D > 0]) + 1e-12
    K = np.exp(-D**2 / (2*bw**2)); np.fill_diagonal(K, 0.0)
    d = K.sum(1); K = K / np.outer(d, d); rs = K.sum(1); P = K / rs[:, None]
    P = 0.5*(P+P.T); ev, V = np.linalg.eigh(P); idx = np.argsort(ev)[::-1]
    return V[:, idx[1:ncomp+1]] * ev[idx[1:ncomp+1]]


def run_aoa(costs, labels, eps):
    cfg = BarycenterConfig(n_supports=NS, epsilon=eps, n_restarts=3, max_iter=MAXIT, n_workers=8)
    land = [costs[i] for lab in sorted(set(labels)) for i in np.where(labels == lab)[0][:N_LAND]]
    B, _ = gw_barycenter(land, cfg=cfg)
    plans, _ = member_transport_plans(costs, B, cfg=cfg)
    P = np.array([np.asarray(T, float).ravel() / max(np.asarray(T, float).sum(), 1e-30) for T in plans])
    DH = cdist(np.sqrt(np.maximum(P, 0)), np.sqrt(np.maximum(P, 0)))
    DF = cdist(P, P)
    H, Fc = dmap_on(DH), dmap_on(DF)
    k = len(set(labels))
    ariH = adjusted_rand_score(labels, KMeans(k, n_init=10, random_state=0).fit_predict(H))
    ariF = adjusted_rand_score(labels, KMeans(k, n_init=10, random_state=0).fit_predict(Fc))
    n = len(P)
    w = np.median([DH[i, j] for i in range(n) for j in range(i+1, n) if labels[i] == labels[j]])
    c = np.median([DH[i, j] for i in range(n) for j in range(i+1, n) if labels[i] != labels[j]])
    return dict(eps=eps, ratio=c/w, ariH=ariH, ariF=ariF, H=H, Fc=Fc, DH=DH, DF=DF)


def main():
    costs, labels, names = build_costs()
    labels = np.asarray(labels)
    print(f"total {len(costs)} attractors across {len(set(labels))} families "
          f"(RAW Takens, n_s={NS})", flush=True)
    results = []
    for eps in EPS_SWEEP:
        r = run_aoa(costs, labels, eps)
        results.append(r)
        print(f"  eps={eps:5.3f}: Hellinger ratio {r['ratio']:.2f}  ARI_H {r['ariH']:.3f}  ARI_F {r['ariF']:.3f}", flush=True)
    best = max(results, key=lambda r: r["ariH"])
    print(f"\nBEST eps={best['eps']}: ARI_H {best['ariH']:.3f} ARI_F {best['ariF']:.3f} ratio {best['ratio']:.2f}", flush=True)
    # Persist the pairwise plan-distance matrices under both metrics as the
    # record of the (negative) family-recovery measurement; nothing else reads
    # this artifact.
    np.savez(os.path.join(_ROOT, "artifacts", "dysts_plan_metrics.npz"),
             DH=best["DH"], DF=best["DF"], labels=labels,
             names=np.array(names), ariH=best["ariH"], ariF=best["ariF"],
             eps=best["eps"], ratio=best["ratio"])
    _figure(best, labels, names)


def _figure(best, labels, names):
    H, Fc, ariH, ariF = best["H"], best["Fc"], best["ariH"], best["ariF"]
    fams = [names[list(labels).index(i)] for i in sorted(set(labels))]
    cmap = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(7.1, 3.4))
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.99, top=0.9, bottom=0.13, wspace=0.3)

    def emb(ax, X, title):
        for i in sorted(set(labels)):
            m = labels == i
            ax.scatter(X[m, 0], X[m, 1], s=14, color=cmap(i % 20), label=fams[i],
                       edgecolors="k", linewidths=0.2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("PSoPS$_1$"); ax.set_ylabel("PSoPS$_2$"); ax.set_title(title, pad=2)

    a1 = fig.add_subplot(gs[0, 0]); emb(a1, H, rf"(i) Hellinger PSoPS (ARI {ariH:.2f})")
    a2 = fig.add_subplot(gs[0, 1]); emb(a2, Fc, rf"(ii) Frobenius (ARI {ariF:.2f})")  # both ARIs low: see docstring
    a1.legend(fontsize=5.0, ncol=2, loc="best", frameon=False,
              handletextpad=0.2, columnspacing=0.6, borderpad=0.1)
    a3 = fig.add_subplot(gs[0, 2])
    a3.bar([0, 1], [ariH, ariF], color=["#2a7f9e", "#c0c6cc"], width=0.6)
    a3.set_xticks([0, 1]); a3.set_xticklabels(["Hellinger", "Frobenius"])
    a3.set_ylabel("ARI vs family"); a3.set_ylim(0, 1)
    for sp in ("top", "right"):
        a3.spines[sp].set_visible(False)
    a3.set_title("(iii) metric matters", pad=2)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig("/tmp/fig_dysts_families.png", dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.normpath(FIG_OUT))


if __name__ == "__main__":
    main()
