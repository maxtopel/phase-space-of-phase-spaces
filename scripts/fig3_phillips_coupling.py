#!/usr/bin/env python3
r"""
Phillips coupling (Results sec.~\ref{subsec:phillips-loop}).

Do the UNRATE and YoY-inflation attractors *share structure* (predict each
other), and does the business-cycle loop live in their COUPLED geometry?

Method (paper-native; pairwise GW barycenter of inflation & unemployment ONLY,
not the global PSoPS barycenter):
  1. Per series: Takens embed (dim=8, tau=4) -> diffusion map (5 comps) ->
     FPS to n_s=40 supports -> intra-distance matrix C_i.
  2. Pairwise GW barycenter B* of {UNRATE, pi}; per-series plans T_i: C_i->C_{B*}.
  3. Mediated coupling  T_{ij}=T_j T_i^T : sparse (low entropy) => predictive.
  4. Directed sensitivity  S_{i->j} (Eq.~\ref{eq:sensitivity}); low participation
     ratio => concentrated => shared; S_{i->j} != S_{j->i} => lead/lag.
  5. Loops: persistent H1 on UNRATE, pi, and their JOINT delay-DMAP embedding.
Entropic GW has a stochastic barycenter init, so every coupling statistic is
reported as mean +/- std over N_REP replicate seeds. Control: a random walk run
through the identical pipeline (unrelated null).

Figure (page width): A = 3-D attractors (UNRATE / inflation / joint);
B = raw data (Phillips-loop plane + UNRATE & pi timeseries);
C = coupling (plan T_ij + coupling-over-time).

Data:    FRED UNRATE, CPIAUCSL (YoY 12-mo log-return), DATA_ROOT/fred.
Outputs: prints diagnostics; writes ../figures/fig_phillips_coupling.pdf.
Run:     python3 scripts/fig3_phillips_coupling.py
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, sys, csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import config, figstyle as F
F.use()

from aoa_repro import pipeline
config.use_vendor()
import diffuse as _diffuse
import ot

# One font, one size, matching the revtex4-2 (Computer Modern) body text.
FS = 9
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.unicode_minus": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------- PARAMETERS
SEED      = 0
EMBED_DIM = 8          # Takens embedding dimension
EMBED_TAU = 4          # Takens delay
N_COMP    = 5          # diffusion components (modes Phi, eigenvalues lambda)
NS        = 40         # transport-plan support size (paper n_s)
EPS_CANDS = (0.01, 0.05, 0.1, 0.5)     # entropic-GW epsilon candidates (vendored)
PH_NOISE  = 0.05       # persistent-H1 audit noise fraction
N_REP     = 12         # replicate seeds (entropic-GW barycenter init is stochastic)
FIG_OUT   = str(config.FIG_OUT / "fig_phillips_coupling.pdf")
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- data
def load_dated(path):
    dates, vals = [], []
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                d = datetime.strptime(row[0][:10], "%Y-%m-%d"); v = float(row[1])
            except (ValueError, TypeError):
                continue
            dates.append(d); vals.append(v)
    return np.array(dates), np.array(vals, float)


def load_unrate_pi():
    du, u = load_dated(config.DATA_ROOT / "fred" / "fred_UNRATE.csv")
    dc, c = load_dated(config.DATA_ROOT / "fred" / "fred_CPIAUCSL.csv")
    pi_d = dc[12:]; pi_v = np.log(c[12:]) - np.log(c[:-12])
    key_u = {(d.year, d.month): uv for d, uv in zip(du, u)}
    U, PI, YR = [], [], []
    for d, pv in zip(pi_d, pi_v):
        if (d.year, d.month) in key_u:
            U.append(key_u[(d.year, d.month)]); PI.append(pv); YR.append(d.year + d.month/12)
    return np.array(U), np.array(PI), np.array(YR)


# ----------------------------------------------- attractor -> supports, modes
def attractor(series):
    X = pipeline.delay_embed(series, dim=EMBED_DIM, tau=EMBED_TAU)
    coords, evals = pipeline.diffusion_coords(X, n_components=N_COMP)
    sub, idx = _diffuse.subsample_fps(coords, NS)
    C = _diffuse.intra_distance_matrix(sub)
    return C, coords[idx], np.asarray(evals, float), coords


def bary_eps(Cs):
    p = np.ones(NS)/NS
    for eps in EPS_CANDS:
        with np.errstate(all="ignore"):
            CB = ot.gromov.gromov_barycenters(NS, Cs, [p]*len(Cs), p,
                                              [1.0/len(Cs)]*len(Cs), "square_loss",
                                              epsilon=eps, max_iter=200, tol=1e-5)
        if np.all(np.isfinite(CB)) and CB.max() > 0:
            return CB, eps
    return CB, EPS_CANDS[-1]


def gw_plan(C, CB, eps):
    p = np.ones(NS)/NS
    with np.errstate(all="ignore"):
        T = ot.gromov.entropic_gromov_wasserstein(C, CB, p, p, "square_loss",
                                                  epsilon=eps, max_iter=2000, tol=1e-9)
    if not np.all(np.isfinite(T)) or T.sum() <= 0:
        T = ot.gromov.gromov_wasserstein(C, CB, p, p, "square_loss")
    return T/T.sum()


# ----------------------------------------------- diagnostics
def eff_support(T):
    T = np.maximum(T, 0); T = T/T.sum()
    return float(np.exp(-np.sum(T[T > 0]*np.log(T[T > 0]))))

def norm_mi(T):
    T = np.maximum(T, 0); T = T/T.sum()
    p = T.sum(1, keepdims=True); q = T.sum(0, keepdims=True); m = T > 0
    return float(np.sum(T[m]*np.log(T[m]/(p@q)[m]))/np.log(T.shape[0]))

def part_ratio(M):
    a = np.abs(M).ravel(); s = a.sum()
    return float(s*s/(np.sum(a*a)+1e-30))

def sensitivity(Ti, Tj, Phi_i, Phi_j, lam_j):
    Tij = Tj @ Ti.T
    Pti = NS*(Ti.T @ Phi_i); Ptj = NS*(Tj.T @ Phi_j)
    w = lam_j/(lam_j.sum()+1e-30); K = Phi_i.shape[1]
    S = np.zeros((K, K))
    for k in range(K):
        a = Tij @ Pti[:, k]
        for l in range(K):
            S[k, l] = w[l]*np.mean(a*Ptj[:, l])
    return S


def coupling_stats(att_U, att_P, att_R, n_rep=N_REP):
    """Replicate the coupling over n_rep barycenter-init seeds. Returns a dict of
    mean/std for MI(real), MI(null), S participation ratio (real/null), and the
    lead/lag asymmetry, plus the mean real coupling plan for plotting."""
    CU, PhiU, lamU, _ = att_U
    CP, PhiP, lamP, _ = att_P
    CR, PhiR, lamR, _ = att_R
    mis, mirs, prs, prrs, asy, plans = [], [], [], [], [], []
    nU, nP = [], []          # ||S_U->pi|| and ||S_pi->U|| per seed, for the direction test
    for k in range(n_rep):
        # global seeding is deliberate: the vendored GW barycenter reads
        # numpy's global state, so per-replicate determinism requires it
        np.random.seed(1000 + k)               # vary GW-barycenter init
        CB, e = bary_eps([CU, CP]); TU = gw_plan(CU, CB, e); TP = gw_plan(CP, CB, e)
        CBr, er = bary_eps([CU, CR]); TUr = gw_plan(CU, CBr, er); TR = gw_plan(CR, CBr, er)
        Tij = TP @ TU.T; Tij /= Tij.sum()
        Tijr = TR @ TUr.T; Tijr /= Tijr.sum()
        mis.append(norm_mi(Tij)); mirs.append(norm_mi(Tijr)); plans.append(Tij)
        Sup = sensitivity(TU, TP, PhiU, PhiP, lamP)
        Spu = sensitivity(TP, TU, PhiP, PhiU, lamU)
        prs.append(part_ratio(Sup)); prrs.append(part_ratio(sensitivity(TUr, TR, PhiU, PhiR, lamR)))
        asy.append(np.linalg.norm(Sup - Spu.T)/(np.linalg.norm(Sup)+1e-12))
        nU.append(np.linalg.norm(Sup)); nP.append(np.linalg.norm(Spu))
    ms = lambda a: (float(np.mean(a)), float(np.std(a)))
    # direction of explanation: unemployment -> inflation vs the reverse
    from scipy import stats as _st
    rat = np.array(nU) / np.array(nP)
    tt = _st.ttest_rel(nU, nP)
    return {"mi": ms(mis), "mi_null": ms(mirs), "pr": ms(prs), "pr_null": ms(prrs),
            "asym": ms(asy), "dir_ratio": ms(rat.tolist()),
            "dir_all_gt1": bool(rat.min() > 1.0), "dir_p": float(tt.pvalue)}, np.mean(plans, axis=0)


# US recessions (NBER peak->trough, fractional years) for shading
NBER = [(1953.5, 1954.4), (1957.6, 1958.3), (1960.3, 1961.1), (1969.9, 1970.9),
        (1973.9, 1975.2), (1980.0, 1980.5), (1981.5, 1982.9), (1990.6, 1991.2),
        (2001.2, 2001.9), (2007.9, 2009.5), (2020.1, 2020.4)]


def rolling_coupling(U, PI, YR, win=120, step=6):
    ts, mis = [], []
    for s in range(0, len(U)-win, step):
        u, p = U[s:s+win], PI[s:s+win]
        try:
            np.random.seed(s)
            Cu = attractor(u)[0]; Cp = attractor(p)[0]
            CB, eps = bary_eps([Cu, Cp])
            T = gw_plan(Cp, CB, eps) @ gw_plan(Cu, CB, eps).T
            mis.append(norm_mi(T)); ts.append(YR[s+win//2])
        except Exception:
            continue
    return np.array(ts), np.array(mis)


# ----------------------------------------------- run + figure
def main():
    # Cache: style reruns redraw in seconds; delete or --recompute to refit.
    CCACHE = os.path.join(_ROOT, "caches", "_cache_phillips_coupling.npz")
    recompute = "--recompute" in sys.argv or not os.path.exists(CCACHE)
    if not recompute:
        z = np.load(CCACHE, allow_pickle=True)
        U, PI, YR = z["U"], z["PI"], z["YR"]
        coU, coP, joint, Tij = z["coU"], z["coP"], z["joint"], z["Tij"]
        rt, rmi, mir = z["rt"], z["rmi"], float(z["mir"])
        vmin, vmax = float(YR.min()), float(YR.max())
        print(f"loaded cached computation ({CCACHE})")
    else:
        U, PI, YR = load_unrate_pi()
        RW = np.cumsum(rng.standard_normal(len(U)))
        vmin, vmax = float(YR.min()), float(YR.max())
        print(f"UNRATE/pi joint monthly obs: {len(U)}  ({int(YR[0])}..{int(YR[-1])})")

        att_U = attractor(U); att_P = attractor(PI); att_R = attractor(RW)
        coU, coP = att_U[3], att_P[3]
        stats, Tij = coupling_stats(att_U, att_P, att_R)
        import json as _json
        _json.dump(stats, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "artifacts", "phillips_coupling_stats.json"), "w"), indent=2)
        (mi, mis_), (mir, mirs_) = stats["mi"], stats["mi_null"]
        (pr, prs_), (prn, prns_) = stats["pr"], stats["pr_null"]
        asym, asyms_ = stats["asym"]
        print(f"\n=== coupling statistics over {N_REP} replicate barycenter seeds ===")
        print(f"mediated-coupling MI : UNRATE<->pi {mi:.3f}+/-{mis_:.3f} | null {mir:.3f}+/-{mirs_:.3f}")
        print(f"directed S PR        : UNRATE<->pi {pr:.2f}+/-{prs_:.2f} | null {prn:.2f}+/-{prns_:.2f} (of {N_COMP**2})")
        print(f"lead/lag asymmetry   : {asym:.3f}+/-{asyms_:.3f}")
        dr, drs = stats["dir_ratio"]
        print(f"direction ||S_U->pi||/||S_pi->U||: {dr:.3f}+/-{drs:.3f}  "
              f"all seeds > 1: {stats['dir_all_gt1']}  paired-t p = {stats['dir_p']:.2e}")

        joint = pipeline.takens_joint_dmap([U, PI], dim=EMBED_DIM, tau=EMBED_TAU,
                                           n_components=N_COMP, n_top_coords=3)
        lp = lambda c: pipeline.audit_persistent_h1(c[:, :3], noise_frac=PH_NOISE)[1:]
        bU, mU = lp(coU); bP, mP = lp(coP); bJ, mJ = lp(joint)
        print(f"loops beta_1/max-persist: UNRATE {bU}/{mU:.3f}, pi {bP}/{mP:.3f}, JOINT {bJ}/{mJ:.3f}")

        print("computing rolling coupling ...")
        rt, rmi = rolling_coupling(U, PI, YR)
        np.savez(CCACHE, U=U, PI=PI, YR=YR, coU=coU, coP=coP, joint=joint,
                 Tij=Tij, rt=rt, rmi=rmi, mir=mir)
        print(f"cached computation -> {CCACHE}")

    # ----------------------------------------------- figure (3 rows, page width)
    fig = plt.figure(figsize=(7.1, 5.15))   # under the height a page top can hold
    gs = fig.add_gridspec(3, 6, left=0.085, right=0.93, top=0.95, bottom=0.06,
                          hspace=0.45, wspace=1.15)
    def scat3d(cell, c, yr_, tag):
        ax = fig.add_subplot(cell, projection="3d")
        s = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=yr_, s=5, cmap=F.CMAP_SEQ,
                       linewidths=0, alpha=0.85, depthshade=False, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_visible(False)                 # the grey panes are chrome, not data
            a._axinfo["grid"].update(color=F.CONTEXT, linewidth=0.3)
            a.line.set_color(F.INK); a.line.set_linewidth(0.6)
        ax.set_xlabel(r"$\Psi_1$", labelpad=-9); ax.set_ylabel(r"$\Psi_2$", labelpad=-9)
        ax.set_zlabel(r"$\Psi_3$", labelpad=-9)
        ax.view_init(elev=22, azim=-60)
        ax._sub_idx = tag
        return ax, s

    # Row A: 3-D attractors. Identity is not inferable from position, so each keeps a
    # short name. Group letters and subpanel tags are placed AFTER layout, in figure
    # coordinates, so A/B/C share one x position on the far left (see _letters below).
    axA, _ = scat3d(gs[0, 0:2], coU, np.linspace(vmin, vmax, len(coU)), 0)
    F.law_title(axA, "Unemployment")
    ax2, _ = scat3d(gs[0, 2:4], coP, np.linspace(vmin, vmax, len(coP)), 1)
    F.law_title(ax2, "Inflation")
    ax3, sA = scat3d(gs[0, 4:6], joint, YR[-len(joint):], 2)
    F.law_title(ax3, "Joint")
    cax = fig.add_axes([0.945, 0.70, 0.012, 0.22])
    cb = fig.colorbar(sA, cax=cax); cb.set_label("Year", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)

    # Row B: raw data -- Phillips-loop plane + raw timeseries
    axB1 = fig.add_subplot(gs[1, 0:3])
    axB1.plot(U, PI, color=F.CONTEXT, lw=0.3, alpha=0.5, zorder=1)
    axB1.scatter(U, PI, c=YR, s=7, cmap=F.CMAP_SEQ, linewidths=0,
                 vmin=vmin, vmax=vmax, zorder=2)
    F.finish(axB1, "Unemployment (%)", r"Inflation $\pi_{\mathrm{YoY}}$ (log-return)")
    axB1._sub_idx = 0

    axB2 = fig.add_subplot(gs[1, 3:6])
    for (a, b) in NBER:
        axB2.axvspan(a, b, color=F.CONTEXT, alpha=0.8, lw=0)
    lnU, = axB2.plot(YR, U, color=F.BLUE, lw=1.0, label="Unemployment")
    ax2b = axB2.twinx()
    lnP, = ax2b.plot(YR, PI, color=F.VERMILLION, lw=1.0, label="Inflation")
    for a in (axB2, ax2b):
        a.spines["top"].set_visible(False)
        for sp in ("left", "bottom", "right"):
            a.spines[sp].set_color(F.INK); a.spines[sp].set_linewidth(0.6)
    axB2.set_xlabel("Year"); axB2.set_ylabel("Unemployment (%)")
    ax2b.set_ylabel(r"Inflation $\pi_{\mathrm{YoY}}$")
    # legend anchored in the clear band between the 1980 and 2020 spikes
    axB2.legend(handles=[lnU, lnP], loc="upper center", bbox_to_anchor=(0.72, 1.0),
                fontsize=7, frameon=True, fancybox=False, edgecolor=F.INK,
                framealpha=1.0, borderpad=0.35)
    axB2._sub_idx = 1

    # Row C: coupling -- mean plan + coupling over time. C(i) spans the same
    # columns as B(i) so the two rows' panels share width and left edge.
    axC1 = fig.add_subplot(gs[2, 0:3])
    # log10(T/U): uniform coupling sits at 1; a log scale is needed because a
    # few support pairs carry mass decades above uniform.
    U0 = 1.0 / Tij.size
    R = np.clip(Tij / U0, 1e-2, None)
    vhi = float(np.log10(np.percentile(R, 99.5)))
    imC = axC1.imshow(np.log10(R), cmap=F.CMAP_SEQ, aspect="equal",
                      interpolation="nearest", vmin=-2.0, vmax=max(vhi, 0.5))
    axC1.set_anchor("C")      # square panel centered under B(i)
    axC1.set_xticks([]); axC1.set_yticks([])
    for sp in axC1.spines.values():
        sp.set_color(F.INK); sp.set_linewidth(0.6)
    axC1.set_xlabel(r"Unemployment support $a$")
    axC1.set_ylabel(r"Inflation support $b$")
    axC1._sub_idx = 0

    axC2 = fig.add_subplot(gs[2, 3:6])
    for (a, b) in NBER:
        axC2.axvspan(a, b, color=F.CONTEXT, alpha=0.8, lw=0)
    axC2.plot(rt, rmi, color=F.BLUE, lw=1.4, zorder=3, label="Rolling coupling MI")
    axC2.axhline(mir, color=F.VERMILLION, ls="--", lw=0.9, zorder=2,
                 label="Random-walk null")
    axC2.set_ylim(top=float(np.nanmax(rmi)) * 1.22)   # headroom so the box sits above the curve
    axC2.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False,
                edgecolor=F.INK, framealpha=1.0, borderpad=0.35)
    F.finish(axC2, "Year", "Coupling MI")
    axC2._sub_idx = 1

    # ---- aligned letters and tags, in figure coordinates -------------------
    # One x position for every group letter; every subpanel tag sits at the same
    # offset from its own panel's top-left corner.
    fig.canvas.draw()
    # C(i)'s colorbar hugs the (aspect-shrunk) square, clear of C(ii)'s ylabel
    bC1 = axC1.get_position()
    caxC = fig.add_axes([bC1.x1 + 0.012, bC1.y0, 0.012, bC1.height])
    cbC = fig.colorbar(imC, cax=caxC, extend="both")
    dec = np.arange(-2, int(np.ceil(max(vhi, 0.5))) + 1)
    cbC.set_ticks(dec)
    cbC.set_ticklabels([rf"$10^{{{d}}}$" if d else "$1$" for d in dec])
    cbC.set_label(r"$T\,/\,U$", fontsize=7.5)
    cbC.ax.tick_params(labelsize=7); cbC.outline.set_visible(False)

    rows = [("A", [axA, ax2, ax3]), ("B", [axB1, axB2]), ("C", [axC1, axC2])]
    LX = 0.012
    for letter, axes in rows:
        ytop = max(a.get_position().y1 for a in axes)
        fig.text(LX, ytop + 0.018, letter, fontsize=10, fontweight="bold",
                 color=F.INK, ha="left", va="bottom")
    # tags in rows B and C share one x per column, so C(i) (a centered square,
    # narrower than its cell) lines up under B(i) rather than under its own edge
    col_x = {0: axB1.get_position().x0, 1: axB2.get_position().x0}
    for letter, axes in rows:
        for a in axes:
            b = a.get_position()
            x = b.x0 if letter == "A" else col_x[a._sub_idx]
            fig.text(x + 0.002, b.y1 + 0.006, f"({F.ROMANS[a._sub_idx]})",
                     fontsize=8.5, color=F.INK, ha="left", va="bottom")

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig("/tmp/fig_phillips_coupling.png", dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.normpath(FIG_OUT))


if __name__ == "__main__":
    main()
