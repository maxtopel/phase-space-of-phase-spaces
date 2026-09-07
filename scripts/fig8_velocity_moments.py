#!/usr/bin/env python3
r"""
Transport-plan velocity moments (Results sec.~\ref{subsec:velocity}).

Beyond detecting regime *steps* (Fig.~\ref{fig:velocity}), we characterise the
transport-plan velocity v(t)=||T_t - T_{t-\Delta}||_F as a stochastic process:
its first four moments show that the dynamical basis is mostly STABLE (small,
low-CV velocity) with heavy-tailed excursions.

Velocity construction is identical to the atlas pipeline (regime_detection_v3):
per law, build a full-sample reference attractor (N_REF=20 FPS supports); on a
6-month grid 1965--2022 take a rolling window, embed (AMI/FNN) -> diffusion map
-> FPS to N_WIN=20 -> entropic GW plan T_t to the reference (eps=0.02); then
v(t)=||T_t-T_{t-\Delta}||_F and Hellinger ||sqrt T_t - sqrt T_{t-\Delta}||_F.

Entropic GW has a stochastic init, so each moment is reported as mean +/- s.d.
over N_REP GW-init seeds. The expensive embed/DMAP/FPS step is deterministic and
cached; only the GW solve is re-run per replicate.

Uses the atlas aoa_v3 modules (data/embed/dmap/gw/barycenter).
Run:     python3 scripts/fig8_velocity_moments.py
Outputs: prints moments; writes ../figures/fig_velocity_moments.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aoa_repro import config, figstyle as F
F.use()
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis

# vendored atlas modules (see vendor_atlas/); private tree no longer needed
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "vendor_atlas"))
from aoa_v3.data import _load_series
from aoa_v3.embed import embed_series, EmbedConfig
from aoa_v3.dmap import dmap_series, DmapConfig
from aoa_v3.gw import gw_distance, GWConfig
from aoa_v3.barycenter import fps_landmarks

# One font, one size, matching the revtex4-2 (Computer Modern) body text.
FS = 9
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------- PARAMETERS
N_REF = 20            # reference attractor support size
N_WIN = 20            # rolling-window support size
GW_EPS = 0.02         # entropic-GW regularization the atlas pipeline
N_COMP = 5            # diffusion components
N_REP = 8             # GW-init replicate seeds (entropic GW is stochastic)
GRID = pd.date_range("1965-01-01", "2022-12-01", freq="6MS").to_list()
LAWS = [{"label": "Phillips", "target": "CPILFESL", "logret": True,  "w": 60},
        {"label": "Okun",     "target": "UNRATE",   "logret": False, "w": 60},
        {"label": "Solow",    "target": "GDPC1",    "logret": True,  "w": 180}]
# (Solow is quarterly: window 120mo => ~40 obs < the 50-point embedding minimum,
#  which is why the atlas regime_detection_v3 has Solow n_valid_plans=0. We use a
#  180mo window so the quarterly series embeds.)
REGIMES = {"Volcker": "1979-10-01", "Great Mod.": "1984-01-01", "Recession": "2007-12-01",
           "Recovery": "2009-06-01", "COVID": "2020-03-01"}
FIG_OUT = str(config.FIG_OUT / "fig_velocity_moments.pdf")


# ----------------------------------------------------------------- pipeline
def _emb(y):
    dv, dim, tau = embed_series(y, cfg=EmbedConfig())
    co, _ = dmap_series(dv, cfg=DmapConfig(n_components=N_COMP, theiler=0), tau=tau)
    return co


def _pre(y, lr):
    return np.diff(np.log(y)) if lr else y


def reference_C(ys, lr):
    co = _emb(_pre(ys.dropna().to_numpy(float), lr))
    D = cdist(co, co)
    sel = fps_landmarks(D, n_landmarks=min(N_REF, len(co)), seed=42)
    return cdist(co[sel], co[sel])


def window_C(ys, t, lr, w):
    """Deterministic part: rolling-window support-distance matrix C_win(t)."""
    yw = ys.loc[t - pd.DateOffset(months=w):t].dropna()
    if len(yw) < 20:
        return None
    try:
        co = _emb(_pre(yw.to_numpy(float), lr))
    except Exception:
        return None
    if co.shape[0] < N_WIN:
        return None
    D = cdist(co, co)
    sel = fps_landmarks(D, n_landmarks=N_WIN, seed=0)
    return cdist(co[sel], co[sel])


def plan(C_win, C_ref):
    try:
        _, T, _ = gw_distance(C_win, C_ref, cfg=GWConfig(epsilon=GW_EPS, max_iter=500, tol=1e-7))
    except Exception:
        return None
    T = np.asarray(T, float); s = T.sum()
    return T / s if np.isfinite(s) and s > 1e-20 else None


def velocity_series(C_wins, C_ref, seed):
    """One replicate: GW plan per grid date (seed varies entropic-GW init),
    then Frobenius and Hellinger velocities of successive plans."""
    np.random.seed(seed)
    Ts = {t: (plan(C, C_ref) if C is not None else None) for t, C in C_wins.items()}
    ds = sorted(Ts)
    vF, vH, dates = [], [], []
    prev = None
    for t in ds:
        if Ts[t] is None:
            continue
        if prev is not None:
            vF.append(float(np.linalg.norm(Ts[t] - prev)))
            vH.append(float(np.linalg.norm(np.sqrt(Ts[t]) - np.sqrt(prev))))
            dates.append(t)
        prev = Ts[t]
    return np.array(dates), np.array(vF), np.array(vH)


def moments(v):
    m = float(np.mean(v)); s = float(np.std(v))
    return dict(mean=m, std=s, cv=s / (m + 1e-30),
                skew=float(skew(v)), kurt=float(kurtosis(v, fisher=True)))


# ----------------------------------------------------------------- run
def main():
    # Computation cache: style-only reruns redraw in a second. Delete the file
    # or pass --recompute to redo the rolling GW solves.
    import sys as _sys, pickle as _pickle
    CCACHE = os.path.join(_ROOT, "caches", "_cache_velocity_moments.pkl")
    if os.path.exists(CCACHE) and "--recompute" not in _sys.argv:
        results = _pickle.load(open(CCACHE, "rb"))
        print(f"loaded cached computation ({CCACHE})")
        _figure(results)
        return
    results = {}
    for law in LAWS:
        print(f"[{law['label']}] loading {law['target']} ...", flush=True)
        ys = _load_series(law["target"]).sort_index()
        C_ref = reference_C(ys, law["logret"])
        C_wins = {t: window_C(ys, t, law["logret"], law["w"]) for t in GRID}
        nvalid = sum(c is not None for c in C_wins.values())
        if nvalid < 10:
            print(f"  SKIP {law['label']}: only {nvalid} valid windows (too short to embed).")
            continue
        # replicate the (stochastic) GW velocity
        reps = [velocity_series(C_wins, C_ref, seed=1000 + r) for r in range(N_REP)]
        dates = reps[0][0]
        VF = np.array([r[1] for r in reps]); VH = np.array([r[2] for r in reps])
        mF = [moments(VF[r]) for r in range(N_REP)]
        agg = {k: (np.mean([m[k] for m in mF]), np.std([m[k] for m in mF]))
               for k in ("mean", "std", "cv", "skew", "kurt")}
        results[law["label"]] = dict(dates=dates, vF=VF.mean(0), vF_sd=VF.std(0),
                                     vH=VH.mean(0), agg=agg, nvalid=nvalid)
        print(f"  valid plans {nvalid}/{len(GRID)}; velocity moments (mean+/-s.d. over {N_REP} GW seeds):")
        for k in ("mean", "std", "cv", "skew", "kurt"):
            print(f"    {k:5s} = {agg[k][0]:.4f} +/- {agg[k][1]:.4f}")

    _pickle.dump(results, open(CCACHE, "wb"))
    print(f"cached computation -> {CCACHE}")
    _figure(results)


def _figure(results):
    laws = list(results.keys())
    # canonical US regime dates marked on every law (fractional year)
    reg = [(f"{k} {pd.Timestamp(v).year}",
            pd.Timestamp(v).year + (pd.Timestamp(v).month - 1) / 12)
           for k, v in REGIMES.items()]
    ecol = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(7.1, 2.7))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.80, bottom=0.17, wspace=0.30)
    for j, law in enumerate(laws):
        R = results[law]
        yrs = np.array([d.year + (d.month - 1) / 12 for d in R["dates"]])
        ax = fig.add_subplot(gs[0, j])
        m, s = R["vF"].mean(), R["vF"].std()
        ax.axhspan(m - s, m + s, color=F.CONTEXT, alpha=0.7, lw=0, zorder=0)
        ax.axhline(m, color=F.MUTED, lw=0.7, ls="--", zorder=1)
        for i, (name, x) in enumerate(reg):
            ax.axvline(x, color=ecol(i), lw=1.0, ls=(0, (4, 2)), alpha=0.9, zorder=2,
                       label=name if j == 0 else None)
        ax.plot(yrs, R["vF"], color=F.BLUE, lw=1.1, zorder=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        F.law_title(ax, law)
        F.panel(ax, ["A", "B", "C"][j], dx=-0.13, dy=1.08)
        ax.set_xlabel("Year")
        if j == 0:
            ax.set_ylabel(r"Velocity $\|T_t-T_{t-\Delta}\|_F$")
    fig.legend(loc="upper center", ncol=len(reg), frameon=True, fancybox=False,
               edgecolor=F.INK, framealpha=1.0, borderpad=0.35, fontsize=8,
               bbox_to_anchor=(0.5, 1.06), columnspacing=0.8, handletextpad=0.4)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig("/tmp/fig_velocity_moments.png", dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.normpath(FIG_OUT))
    # moments for the LaTeX table (printed for transcription into main.tex)
    print("LATEX_MOMENTS")
    for l in laws:
        a = results[l]["agg"]
        print(f"  {l} & {a['mean'][0]:.3f} & {a['cv'][0]:.2f} & {a['skew'][0]:.2f} & {a['kurt'][0]:.2f}")


if __name__ == "__main__":
    main()
