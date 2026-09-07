#!/usr/bin/env python3
"""Reproduce the universal-PSoPS figure (fig_aoa.pdf), four panels.

A: UMAP of the per-series PSoPS coordinates, the 17,025 x 10 Hellinger-dMap
   coordinates the paper is built on (nystrom_281k_coords.npz, exhaustive set),
   colored by concept. The honest read is a CONNECTED manifold with
   concept-enriched regions, not separable islands. The same script computes
   the 10-way KMeans ARI of these coordinates against the concept labels.
B: the directed coupling network of the 30 conceptual @timescale barycenters
   (multiscale_453907 checkpoint), laid out BY DIRECTION: x = net incoming
   coupling, so net sources sit left of zero and net sinks right. Includes the
   direction-shuffled null test and the rate-node rank test printed to stdout.
C: the Nystrom-extended corpus over the exhaustive corpus in the same 3-D
   coordinates (nystrom_281k_coords.npz).
D: nearest-neighbour distance CDFs, the quantitative form of C.

Writes figures/fig_aoa.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
import numpy as np
import matplotlib.pyplot as plt

from aoa_repro import config, plotting as P
P.set_style()
SEED = config.SEED
CACHE = config.COMPUTATION / "caches" / "_cache_umap_coords.npz"


# -------------------------------------------------------------- (a) UMAP
# The object under the microscope is the paper's own coordinate set: the
# 17,025 x 10 PSoPS coordinates of the exhaustive corpus.
OUT = config.COMPUTATION / "artifacts"
zc0 = np.load(OUT / "nystrom_281k_coords.npz")
Xco = np.asarray(zc0["exhaustive"], np.float64)
zd = np.load(OUT / "corpus_dist.npz", allow_pickle=True)
valid = np.sort(np.where(np.load(OUT / "plans_valid_dist.npy"))[0])
IND = zd["indicator"].astype(str)[valid]
print(f"PSoPS coordinates: {Xco.shape}, {len(IND)} concept labels")

if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    emb = z["emb"]; IND = z["ind"].astype(str)
    print("loaded cached UMAP", emb.shape)
else:
    import umap
    print("running UMAP on", Xco.shape, "...")
    emb = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=SEED,
                    n_components=2).fit_transform(Xco)
    np.savez(CACHE, emb=emb, ind=IND)
    print("cached UMAP ->", CACHE)

# 10-way clustering of the SAME coordinates against the concept labels: the
# named concepts are not discovered clusters of the geometry.
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
km = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit(Xco)
ari_coords = float(adjusted_rand_score(IND, km.labels_))
print(f"KMeans-10 ARI of PSoPS coordinates vs concept labels: {ari_coords:.4f}")
json.dump({"K": 10, "ari_vs_concept_coords": ari_coords, "n": int(len(IND))},
          open(OUT / "auto_coords.json", "w"), indent=2)


# -------------------------------------------------------------- (b) network
try:
    ms_json = next(config.CHECKPOINTS.glob("multiscale_*453907*.json"))
except StopIteration:
    raise FileNotFoundError(
        f"multiscale_*453907*.json not found under {config.CHECKPOINTS}; "
        "it ships in data/atlas_anchors/checkpoints/") from None
ms = json.load(open(ms_json))
nodes = sorted(ms["barycenters"].keys())          # 30 indicator@timescale labels
N = len(nodes)
idx = {n: i for i, n in enumerate(nodes)}
S = np.full((N, N), np.nan)
for k, v in ms["coupling_matrix"].items():
    if "->" not in k:
        continue
    src, tgt = k.split("->")
    if src in idx and tgt in idx:
        S[idx[src], idx[tgt]] = v.get("mean_S", np.nan)
S_sym = np.where(np.isfinite(S) & np.isfinite(S.T), 0.5 * (S + S.T), np.nan)
S_max = float(np.nanmax(S_sym))
indicators = [n.split("@")[0] for n in nodes]
timescales = [n.split("@")[1] for n in nodes]
print(f"barycenter network: {N} nodes ({len(set(indicators))} indicators, "
      f"{len(set(timescales))} timescales); max coupling S_max={S_max:.3f}")


# -------------------------------------------------------------- figure
import os
from scipy.spatial import cKDTree
from aoa_repro import figstyle as F
F.use()

zc = np.load(OUT / "nystrom_281k_coords.npz")
E_ex, X_ext = zc["exhaustive"].astype(np.float64), zc["extended"].astype(np.float64)
rng_c = np.random.default_rng(0)

fig = plt.figure(figsize=(7.1, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], left=0.055, right=0.99,
                      top=0.955, bottom=0.075, wspace=0.24, hspace=0.30)

# ---- A: UMAP of the PSoPS coordinates, concept-colored -----------------------
axA = fig.add_subplot(gs[0, 0])
GROUPS = ["interest_rate", "gdp", "inflation", "exchange_rate",
          "labor", "unemployment", "credit", "capital", "trade", "tfp"]
other = ~np.isin(IND, GROUPS)
axA.scatter(emb[other, 0], emb[other, 1], s=0.9, c=F.CONTEXT, alpha=0.45,
            linewidths=0, rasterized=True)
for g in GROUPS:
    m = IND == g
    if m.sum():
        axA.scatter(emb[m, 0], emb[m, 1], s=1.4, c=F.CONCEPT_COLORS[g], alpha=0.7,
                    linewidths=0, rasterized=True,
                    label=g.replace("_", " "))
axA.set_xticks([]); axA.set_yticks([])
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    axA.spines[sp].set_color(F.INK); axA.spines[sp].set_linewidth(0.6)
axA.set_xlabel(r"UMAP$_1$"); axA.set_ylabel(r"UMAP$_2$")
axA.legend(loc="upper left", fontsize=6.2, frameon=True, fancybox=False,
           edgecolor=F.INK, framealpha=1.0, borderpad=0.35,
           handletextpad=0.25, labelspacing=0.22, markerscale=2.6)

# ---- B: directed coupling, laid out by direction ---------------------------
# Position IS the finding: x = net incoming coupling puts sources left of zero
# and sinks right; if transmission is real, arrows flow left to right.
axB = fig.add_subplot(gs[0, 1])
out_c = np.nansum(S, axis=1)
in_c = np.nansum(S, axis=0)
net = out_c - in_c
tot = np.nansum(S_sym, axis=1)
size = 22 + 70 * (tot - tot.min()) / max(tot.max() - tot.min(), 1e-9)
xB, yB = -net, tot          # sources (net>0) on the LEFT, flow reads left->right
# nudge apart vertically where nodes would overprint
order_y = np.argsort(yB)
for a_i in range(1, N):
    i1, i0 = order_y[a_i], order_y[a_i - 1]
    if abs(yB[i1] - yB[i0]) < 0.12 and abs(xB[i1] - xB[i0]) < 0.25:
        yB[i1] = yB[i0] + 0.14
thr_dir = np.nanpercentile(S[np.isfinite(S)], 70)      # top 30% of directed pairs
from matplotlib.patches import FancyArrowPatch
n_lr = n_rl = 0
for a_i in range(N):
    for b_i in range(N):
        if a_i == b_i or not np.isfinite(S[a_i, b_i]) or S[a_i, b_i] < thr_dir:
            continue
        lw = 0.25 + 1.0 * (S[a_i, b_i] - thr_dir) / max(np.nanmax(S) - thr_dir, 1e-9)
        axB.add_patch(FancyArrowPatch((xB[a_i], yB[a_i]), (xB[b_i], yB[b_i]),
                                      arrowstyle="-|>", mutation_scale=5, lw=lw,
                                      color="#b8c0c8", alpha=0.35,
                                      shrinkA=4, shrinkB=4, zorder=1))
        n_lr += xB[a_i] < xB[b_i]; n_rl += xB[a_i] > xB[b_i]
print(f"directional layout: {n_lr} arrows flow source->sink (left->right), {n_rl} against")

# Null: swap S_ij <-> S_ji per pair, recompute layout and flow fraction.
# Recomputing the layout matters: the source/sink axis is itself built from S.
def _flow_frac(Sm):
    outn = np.nansum(Sm, 1) - np.nansum(Sm, 0)
    xb = -outn
    thr = np.nanpercentile(Sm[np.isfinite(Sm)], 70)
    lr = rl = 0
    for a3 in range(N):
        for b3 in range(N):
            if a3 == b3 or not np.isfinite(Sm[a3, b3]) or Sm[a3, b3] < thr:
                continue
            lr += xb[a3] < xb[b3]; rl += xb[a3] > xb[b3]
    return lr / max(lr + rl, 1)

obs_frac = n_lr / max(n_lr + n_rl, 1)
rngp = np.random.default_rng(7)
null_fracs = []
iu = np.triu_indices(N, 1)
for _ in range(2000):
    Sp = S.copy()
    flip = rngp.random(len(iu[0])) < 0.5
    for (a3, b3, fl) in zip(iu[0], iu[1], flip):
        if fl:
            Sp[a3, b3], Sp[b3, a3] = Sp[b3, a3], Sp[a3, b3]
    null_fracs.append(_flow_frac(Sp))
null_fracs = np.array(null_fracs)
pval = float((null_fracs >= obs_frac).mean())
print(f"flow-asymmetry test: observed {obs_frac:.3f} vs null "
      f"{null_fracs.mean():.3f}+/-{null_fracs.std():.3f}, p = {pval:.4f} "
      f"({(null_fracs >= obs_frac).sum()}/2000 null draws reach it)")

# And the rate-complex statement: where do the interest-rate nodes rank by net
# outgoing coupling, and how likely is that concentration by chance?
net_rank = np.argsort(-net)          # descending net-out
ir_ranks = sorted(int(np.where(net_rank == i3)[0][0]) + 1
                  for i3 in range(N) if indicators[i3] == "interest_rate")
print(f"interest-rate nodes rank {ir_ranks} of {N} by net outgoing coupling")
from math import comb
k_ir = len(ir_ranks); top = max(ir_ranks)
p_comb = comb(N - k_ir, top - k_ir) / comb(N, top)
print(f"P(all {k_ir} rate nodes in top {top} by chance) = {p_comb:.2e}")

for i2, n2 in enumerate(nodes):
    col = F.CONCEPT_COLORS.get(indicators[i2], F.MUTED)
    axB.scatter(xB[i2], yB[i2], s=size[i2], c=col, marker="o",
                edgecolors="k", linewidths=0.5, zorder=2)
axB.axvline(0.0, color=F.MUTED, ls=":", lw=0.8, zorder=0)
axB.text(0.02, 0.99, "net sources", transform=axB.transAxes, fontsize=7,
         color=F.INK, ha="left", va="top", style="italic")
axB.text(0.98, 0.99, "net sinks", transform=axB.transAxes, fontsize=7,
         color=F.INK, ha="right", va="top", style="italic")
F.finish(axB, r"Net incoming coupling $\sum_j S_{j\to i}-\sum_j S_{i\to j}$",
         "Total coupling", nticks=4)

# ---- C: the Nystrom extension lands on the object --------------------------
axC = fig.add_subplot(gs[1, 0], projection="3d")
Ee, Xe = E_ex[:, :3], X_ext[:, :3]
CLIP = 1.0
lim = [(np.percentile(Ee[:, k], CLIP), np.percentile(Ee[:, k], 100 - CLIP))
       for k in range(3)]
lim = [(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)) for lo, hi in lim]
Es = Ee[rng_c.choice(len(Ee), 9000, replace=False)]
Xs = Xe[rng_c.choice(len(Xe), 50000, replace=False)]
axC.computed_zorder = False
axC.scatter(*Es.T, s=1.6, c=F.CONTEXT, alpha=0.6, lw=0, rasterized=True,
            zorder=1, label="Exhaustive corpus")
axC.scatter(*Xs.T, s=1.0, c=F.VERMILLION, alpha=0.30, lw=0, rasterized=True,
            zorder=2, label="Nystrom-extended")
axC.view_init(18, -55)
axC.set_xlim(lim[0]); axC.set_ylim(lim[1]); axC.set_zlim(lim[2])
axC.set_xticks([]); axC.set_yticks([]); axC.set_zticks([])
for a3 in (axC.xaxis, axC.yaxis, axC.zaxis):
    a3.pane.set_visible(False)
    a3._axinfo["grid"].update(color=F.CONTEXT, linewidth=0.3)
    a3.line.set_color(F.INK); a3.line.set_linewidth(0.6)
axC.set_xlabel(r"$\Psi_1$", labelpad=-9); axC.set_ylabel(r"$\Psi_2$", labelpad=-9)
axC.set_zlabel(r"$\Psi_3$", labelpad=-9)
leg = axC.legend(loc="upper right", fontsize=6.2, frameon=True, fancybox=False,
                 edgecolor=F.INK, framealpha=1.0, borderpad=0.35, markerscale=4.0)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

# ---- D: nearest-neighbour CDF (the quantitative form of C) -----------------
axD = fig.add_subplot(gs[1, 1])
tree = cKDTree(E_ex)
d_new = tree.query(X_ext[rng_c.choice(len(X_ext), 40000, replace=False)], k=1)[0]
d_self = tree.query(E_ex[rng_c.choice(len(E_ex), 9000, replace=False)], k=2)[0][:, 1]
for d, col, lab in ((d_self, F.BLUE, "Exhaustive to exhaustive"),
                    (d_new, F.VERMILLION, "Extended to exhaustive")):
    sarr = np.sort(d[d > 0])
    axD.step(sarr, np.arange(1, len(sarr) + 1) / len(sarr), color=col, lw=1.2,
             where="post", label=lab)
axD.axvline(np.percentile(d_self, 95), color=F.INK, ls=":", lw=0.8)
axD.set_xscale("log")
axD.set_xlim(left=1e-4)
axD.set_ylim(0, 1.02)
axD.legend(loc="upper left", fontsize=6.2, frameon=True, fancybox=False,
           edgecolor=F.INK, framealpha=1.0, borderpad=0.35)
F.finish(axD, "Nearest-neighbour distance", "Cumulative fraction", nticks=4)

# ---- aligned panel letters in figure coordinates ---------------------------
fig.canvas.draw()
# one x per column so A sits directly above C and B directly above D
xL = max(min(axA.get_position().x0, axC.get_position().x0) - 0.035, 0.005)
xR = max(min(axB.get_position().x0, axD.get_position().x0) - 0.035, 0.005)
for letter, a, xcol in (("A", axA, xL), ("B", axB, xR),
                        ("C", axC, xL), ("D", axD, xR)):
    b = a.get_position()
    fig.text(xcol, b.y1 + 0.006, letter, fontsize=10,
             fontweight="bold", color=F.INK, ha="left", va="bottom")

out = os.path.join(str(config.FIG_OUT), "fig_aoa.pdf")
fig.savefig(out, dpi=400, bbox_inches="tight")
print("wrote", out)
