#!/usr/bin/env python3
r"""
The PSoPS stability figure (fig_aoa_stability.pdf): one 2x2 graphic.

Row A: Frechet-mean scaling of the reference barycenter (Coifman d_eff and
curvature of a fixed 1500-series evaluation panel vs reference size, read from
artifacts/scaling_coifman_planspace.json, produced by validation/scaling.py).
Row B: dimension robustness of the headline estimate on the full Hellinger
coordinates (corpus-size bootstrap with per-point error bars, and three
sampling measures at matched n=3000).

Headline protocol throughout (pipeline/compute_dimensions.py, distributed_n60):
Hellinger coordinates from sqrtplans_dist.dat, no near-duplicate gate, draws
without replacement. Row-B numbers cache to artifacts/robustness_cache.npz;
delete it or pass --recompute to redo the ~15 min of bootstraps.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from aoa_repro import config, figstyle as F
F.use()          # paper-wide figure style
from scipy.spatial.distance import pdist, squareform
import pipeline.dimension as AD
import pipeline.corpus as C

OUT = C.OUT
FIG = str(config.FIG_OUT)
N_S = 40



def main():
    """One stability figure, four panels. Row A: Frechet-mean scaling of the
    reference barycenter (d_eff and curvature of a fixed 1500-series evaluation
    panel vs reference size, read from scaling_coifman_planspace.json). Row B:
    dimension robustness of the headline estimate on the full Hellinger
    coordinates (corpus-size bootstrap and sampling-measure comparison).

    Headline protocol throughout (pipeline/compute_dimensions.py, distributed_n60):
    Hellinger coordinates from sqrtplans_dist.dat, no near-duplicate gate,
    draws without replacement. An earlier version read plans.dat (raw
    couplings, a different metric) on the n>=200 corpus with a dedup gate,
    which is why it plateaued near 6 while the headline reads 7.3.

    Row-B numbers are cached in robustness_cache.npz; delete it or pass
    --recompute to redo the ~15 min of bootstraps. Writes
    figures/fig_aoa_stability.pdf.
    """
    import sys as _sys
    D_REF = 7.30                       # headline, dimensions_bootstrap.json

    # ---- row A source: the scaling sweep artifact --------------------------
    sc_src = None
    for cand in ("scaling_coifman_planspace.json", "scaling_coifman.json"):
        if os.path.exists(os.path.join(OUT, cand)):
            sc_src = cand
            break
    if sc_src is None:
        raise FileNotFoundError(
            f"no scaling_coifman*.json in {OUT}; regenerate with "
            "validation/scaling.py")
    sc = json.load(open(os.path.join(OUT, sc_src)))["scaling"]
    N_ref = np.array([r["N_frechet"] for r in sc])
    d_ref_curve = np.array([r["coifman_dim"] for r in sc])
    curv = np.array([r["curvature"] for r in sc])

    # ---- row B numbers: cached bootstrap on the Hellinger coordinates ------
    CCACHE = os.path.join(OUT, "robustness_cache.npz")
    if os.path.exists(CCACHE) and "--recompute" not in _sys.argv:
        z = np.load(CCACHE, allow_pickle=True)
        Ns, mean_cf, std_cf = z["Ns"], z["mean_cf"], z["std_cf"]
        names = list(z["names"]); means, stds = z["means"], z["stds"]
        print(f"loaded cached robustness numbers ({CCACHE})")
    else:
        n_all = len(np.load(os.path.join(OUT, "corpus_dist.npz"))["idx"])
        mm = np.memmap(os.path.join(OUT, "sqrtplans_dist.dat"), dtype=np.float32,
                       mode="r", shape=(n_all, N_S * N_S))
        v = np.where(np.load(os.path.join(OUT, "plans_valid_dist.npy")))[0]
        rng = np.random.default_rng(7)

        def coifman_of(rows):
            Xn = np.asarray(mm[np.sort(rows)]) / np.sqrt(2.0)
            return AD.kernel_sum_bandwidth(squareform(pdist(Xn)))[1]

        Ns = [500, 1000, 2000, 4000, 8000, 12800]
        draws = {500: 6, 1000: 6, 2000: 6, 4000: 4, 8000: 3, 12800: 2}
        mean_cf, std_cf = [], []
        for N in Ns:
            reps = [coifman_of(rng.choice(v, N, replace=False)) for _ in range(draws[N])]
            mean_cf.append(np.mean(reps)); std_cf.append(np.std(reps))
            print(f"  N={N:6d}  d={mean_cf[-1]:.2f}+/-{std_cf[-1]:.2f}", flush=True)
        mean_cf, std_cf = np.array(mean_cf), np.array(std_cf)

        con = np.asarray(np.load(os.path.join(OUT, "corpus_dist.npz"),
                                 allow_pickle=True)["concept"])[v]
        NS_DRAW, NDR = 3000, 6

        def scheme_rows(name, r):
            if name == "Natural":
                return v[r.choice(len(v), NS_DRAW, replace=False)]
            if name == "Concept-capped":
                caps, seen = [], {}
                for k in r.permutation(len(v)):
                    c = con[k]; seen[c] = seen.get(c, 0)
                    if seen[c] < 350:
                        seen[c] += 1; caps.append(k)
                return v[np.array(caps[:NS_DRAW])]
            uc = np.unique(con); per = max(1, NS_DRAW // len(uc)); bal = []
            for c in uc:
                ic = np.where(con == c)[0]
                bal.extend(r.choice(ic, min(per, len(ic)), replace=False))
            return v[np.array(bal)]

        names = ["Natural", "Concept-capped", "Concept-balanced"]
        means, stds = [], []
        for name in names:
            reps = [coifman_of(scheme_rows(name, np.random.default_rng(100 + t)))
                    for t in range(NDR)]
            means.append(np.mean(reps)); stds.append(np.std(reps))
            print(f"  {name:16s} d={means[-1]:.2f}+/-{stds[-1]:.2f}", flush=True)
        means, stds = np.array(means), np.array(stds)
        np.savez(CCACHE, Ns=np.array(Ns), mean_cf=mean_cf, std_cf=std_cf,
                 names=np.array(names), means=means, stds=stds)
        print(f"cached robustness numbers -> {CCACHE}")

    # ---- the figure: 2 x 2, one dotted 7.3 reference in every d_eff panel --
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.99, top=0.955, bottom=0.115,
                          wspace=0.26, hspace=0.55)

    axAi = fig.add_subplot(gs[0, 0])
    axAi.plot(N_ref, d_ref_curve, "-o", color=F.BLUE)
    axAi.axhline(D_REF, color=F.MUTED, ls=":", lw=0.8)
    axAi.set_xscale("log")
    axAi.set_ylim(0, D_REF * 1.35)
    F.finish(axAi, "Number of Reference Series", r"$d_{\mathrm{eff}}$ (Coifman)")
    axAi._sub_idx = 0

    axAii = fig.add_subplot(gs[0, 1])
    axAii.plot(N_ref, curv, "-o", color=F.VERMILLION)
    axAii.set_xscale("log")
    axAii.set_ylim(1, max(curv) * 1.12)
    F.finish(axAii, "Number of Reference Series", r"Curvature Ratio $\rho$")
    axAii._sub_idx = 1

    axBi = fig.add_subplot(gs[1, 0])
    axBi.errorbar(Ns, mean_cf, yerr=std_cf, fmt="-o", color=F.BLUE, lw=1.1,
                  ms=3.5, elinewidth=0.7, capsize=2, ecolor=F.INK)
    axBi.axhline(D_REF, color=F.MUTED, ls=":", lw=0.8)
    axBi.set_xscale("log")
    axBi.set_ylim(0, D_REF * 1.35)
    F.finish(axBi, "Number of Embedded Series", r"$d_{\mathrm{eff}}$ (Coifman)")
    axBi._sub_idx = 0

    axBii = fig.add_subplot(gs[1, 1])
    axBii.bar(range(len(means)), means, yerr=stds, color=F.BLUE, width=0.62,
              error_kw=dict(lw=0.7, capsize=2, ecolor=F.INK))
    axBii.axhline(D_REF, color=F.MUTED, ls=":", lw=0.8)
    axBii.set_xticks(range(len(means)))
    axBii.set_xticklabels(names, fontsize=7.5)
    axBii.set_ylim(0, D_REF * 1.35)
    F.finish(axBii, "Sampling Measure", r"$d_{\mathrm{eff}}$ (Coifman)")
    axBii._sub_idx = 1

    fig.canvas.draw()
    xL = max(min(axAi.get_position().x0, axBi.get_position().x0) - 0.055, 0.005)
    for letter, a in (("A", axAi), ("B", axBi)):
        b = a.get_position()
        fig.text(xL, b.y1 + 0.012, letter, fontsize=10, fontweight="bold",
                 color=F.INK, ha="left", va="bottom")
    for a in (axAi, axAii, axBi, axBii):
        b = a.get_position()
        fig.text(b.x0 + 0.002, b.y1 + 0.006, f"({F.ROMANS[a._sub_idx]})",
                 fontsize=8.5, color=F.INK, ha="left", va="bottom")

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_aoa_stability.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, flush=True)
    # The controls_v2.json falsification tail that used to live here is removed:
    # its distance ratio compared a raw-plan denominator against a Hellinger
    # numerator and the paper's control result is validation/controls.py.


if __name__ == "__main__":
    main()
