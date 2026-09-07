#!/usr/bin/env python3
"""
Fig 1 - The PSoPS pipeline (clean conceptual schematic).

Self-contained: no Atlas data tree; only numpy + matplotlib (plus
aoa_repro.config for the output path). Everything is generated locally from a
seeded Lorenz system so the figure reproduces identically on any machine and
is easy to edit.

Design constraints (from review):
  * one row spanning the full text width (revtex4-2 figure*)
  * all five panels are IDENTICAL squares (same rendered size)
  * Computer-Modern serif, one uniform font size, to match body text
  * only left+bottom axis lines (no boxes); EVERY panel has axis labels
  * no overlapping text (arrow labels shifted clear of the next panel's ylabel)
  * (e) is a high-D-looking PSoPS: one connected, branched component

Stages, left to right:
  (a) observable x(t)         scalar timeseries
  (b) delay embedding         reconstructed attractor cloud   [Takens]
  (c) diffusion maps          eigen-coordinates                [dmap]
  (d) transport plan T_i      attractor -> shared barycenter   [Gromov-Wasserstein]
  (e) phase space of phase spaces one point per series             [Hellinger DMAP]

Run:  python3 scripts/fig1_pipeline.py
Out:  ../figures/fig1_pipeline.pdf  (+ /tmp preview PNG)
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import numpy as np
from aoa_repro import config
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import PowerNorm

# --------------------------------------------------------------------------- paths
HERE = _ROOT
FIG_OUT = str(config.FIG_OUT)
os.makedirs(FIG_OUT, exist_ok=True)

# --------------------------------------------------------------------------- style
INK   = "#1f2933"
TEAL  = "#2a7f9e"
GREY  = "#8a949e"
FS    = 9               # one size for ALL text, matching 10pt body closely
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": FS,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
rng = np.random.default_rng(7)

# =========================================================================== data
def lorenz(T=80.0, dt=0.01, burn=2000, s=10.0, r=28.0, b=8.0 / 3.0):
    n = int(T / dt) + burn
    xyz = np.empty((n, 3)); xyz[0] = (1.0, 1.0, 1.0)
    for i in range(n - 1):
        x, y, z = xyz[i]
        xyz[i + 1] = xyz[i] + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
    return xyz[burn:]

XYZ = lorenz()
x = XYZ[:, 0]
t = np.arange(len(x)) * 0.01

def delay_embed(sig, dim=3, tau=8):
    n = len(sig) - (dim - 1) * tau
    return np.column_stack([sig[i * tau:i * tau + n] for i in range(dim)])

dvecs = delay_embed(x, dim=3, tau=8)

# ---- (c) cheap diffusion map on a subsample of the delay cloud --------------
idx = np.linspace(0, len(dvecs) - 1, 700).astype(int)
Y = dvecs[idx]; Y = (Y - Y.mean(0)) / Y.std(0)
D2 = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
K = np.exp(-D2 / (np.median(D2) * 0.5))
d = K.sum(1); Kn = K / np.sqrt(d[:, None] * d[None, :])
P = Kn / Kn.sum(1)[:, None]
w, V = np.linalg.eigh((P + P.T) / 2.0)
order = np.argsort(w)[::-1]; psi = V[:, order]
c1, c2 = psi[:, 1], psi[:, 2]

# ---- (d) a structured GW transport plan (single monotone coupling band) -----
n = 40
ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
center = 0.85 * jj + 0.075 * n * np.sin(jj / n * np.pi)
Tplan = np.exp(-((ii - center) ** 2) / (2 * 2.2 ** 2))
Tplan *= (0.7 + 0.3 * rng.random((n, n)))
Tplan /= Tplan.sum()

# ---- (e) PSoPS: one connected, branched object with concept-enriched arms ----
# Generated to read as a 2-D projection of a high-dimensional manifold: a single
# connected component whose curved arms are enriched in one concept each, which
# is the paper's actual finding (enriched regions, not separable clusters).
groups = ["interest rate", "GDP", "inflation", "exchange rate", "credit"]
gcol = {"interest rate": "#d1495b", "GDP": "#2a7f9e", "inflation": "#edae49",
        "exchange rate": "#66a182", "credit": "#8d6cab"}

def arm(ang, length, curve, npts, spread):
    s = np.linspace(0, 1, npts)
    th = ang + curve * s
    step = length / npts
    px = np.cumsum(np.cos(th)) * step + rng.normal(0, spread, npts)
    py = np.cumsum(np.sin(th)) * step + rng.normal(0, spread, npts)
    return px, py

EX, EY, EC = [], [], []
# core of the giant component (mixed membership)
m = 170
EX.append(rng.normal(0, 0.5, m)); EY.append(rng.normal(0, 0.5, m))
EC += [gcol[g] for g in rng.choice(groups, m)]
# colour-coded curved arms radiating from the core (connected, branched)
arms = [("GDP",            0.45,  3.2,  1.7, 210, 0.19),
        ("inflation",      2.30,  2.5, -1.4, 175, 0.16),
        ("interest rate", -1.25,  2.9,  1.1, 195, 0.18),
        ("exchange rate",  1.45,  2.0,  2.6, 150, 0.15),
        ("credit",        -2.45,  2.3, -1.9, 150, 0.16)]
for g, ang, length, curve, npts, spread in arms:
    px, py = arm(ang, length, curve, npts, spread)
    EX.append(px); EY.append(py); EC += [gcol[g]] * npts
ex = np.concatenate(EX); ey = np.concatenate(EY)

# =========================================================================== layout
# Manual equal-size square panels so every box is identical (incl. the heatmap).
W, H = 7.2, 1.95
GAP_IN, MARG_IN = 0.70, 0.06
s_in = (W - 2 * MARG_IN - 4 * GAP_IN) / 5.0          # square side (inches)
wf, hf = s_in / W, s_in / H                          # square in figure fraction
gapf, margf = GAP_IN / W, MARG_IN / W
y0 = 0.26                                            # panel bottom (room below for labels)
xs = [margf + i * (wf + gapf) for i in range(5)]
rects = [[xs[i], y0, wf, hf] for i in range(5)]

fig = plt.figure(figsize=(W, H))

def lb_axes(ax):
    """Keep only the left + bottom axis lines; drop the box and ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK); ax.spines[side].set_linewidth(0.7)
    ax.set_xticks([]); ax.set_yticks([])

def axis_labels(ax, xlab, ylab):
    ax.set_xlabel(xlab, fontsize=FS, color=INK, labelpad=2)
    ax.set_ylabel(ylab, fontsize=FS, color=INK, labelpad=2)

def title(rect, letter, txt):
    xc = rect[0] + rect[2] / 2
    fig.text(xc, y0 + hf + 0.05, f"$\\bf({letter})$ {txt}",
             ha="center", va="bottom", fontsize=FS, color=INK)

# ---- (a) observable -------------------------------------------------------
axa = fig.add_axes(rects[0])
axa.plot(t[:1500], x[:1500], color=INK, lw=0.6)
lb_axes(axa); axis_labels(axa, "time $t$", "$x(t)$")
title(rects[0], "a", "observable")

# ---- (b) delay embedding (3-D attractor) ----------------------------------
axb = fig.add_axes(rects[1], projection="3d")
axb.plot(dvecs[:, 0], dvecs[:, 1], dvecs[:, 2], color=TEAL, lw=0.32, alpha=0.85)
axb.set_xticks([]); axb.set_yticks([]); axb.set_zticks([]); axb.grid(False)
for pane in (axb.xaxis, axb.yaxis, axb.zaxis):
    pane.pane.set_alpha(0.0); pane.line.set_color((0, 0, 0, 0))
axb.set_box_aspect((1, 1, 1)); axb.view_init(elev=24, azim=-58)
# 3-D per-axis labels never sit cleanly in a small cube; name the three delay
# coordinates with one compact label, aligned with the other panels' xlabels.
fig.text(rects[1][0] + rects[1][2] / 2, y0 - 0.052,
         r"$(x_t,\, x_{t-\tau},\, x_{t-2\tau})$",
         ha="center", va="top", fontsize=FS, color=INK)
title(rects[1], "b", "delay embedding")

# ---- (c) diffusion maps ---------------------------------------------------
axc = fig.add_axes(rects[2])
axc.scatter(c1, c2, s=3.0, c=np.arange(len(c1)), cmap="viridis", alpha=0.85, linewidths=0)
lb_axes(axc); axis_labels(axc, r"$\psi_1$", r"$\psi_2$")
title(rects[2], "c", "diffusion maps")

# ---- (d) transport plan (aspect='auto' so the box matches the others) ------
axd = fig.add_axes(rects[3])
axd.imshow(Tplan, cmap="magma", aspect="auto", origin="upper",
           norm=PowerNorm(gamma=0.55, vmin=0, vmax=Tplan.max()))
lb_axes(axd); axis_labels(axd, r"barycenter$_i$", r"attractor$_j$")
title(rects[3], "d", r"transport plan")

# ---- (e) phase space of phase spaces (one connected, branched component) -------
axe = fig.add_axes(rects[4])
axe.scatter(ex, ey, s=3.0, c=EC, alpha=0.85, linewidths=0)
lb_axes(axe); axis_labels(axe, "PSoPS$_1$", "PSoPS$_2$")
axe.margins(0.10)
title(rects[4], "e", "phase space of phase spaces")

# ---- arrows + short transform names (full detail in the caption) ----------
# Each arrow AND its label STOP before the next panel's y-axis label (which sits
# outside the panel, in the gap). We measure the rendered ylabel extent so the
# arrow line never crosses "attractor_j" / "AoA_2" etc. and the text is clamped
# to that free region.
yc = y0 + hf / 2.0
labels = ["Takens", "dMap", "Gromov-\nWasserstein", "Hellinger\ndMap"]
axes_list = [axa, axb, axc, axd, axe]
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for i, lab in enumerate(labels):
    xr = xs[i] + wf                                   # right edge of panel i
    rax = axes_list[i + 1]
    ylab = rax.yaxis.get_label()
    if ylab.get_text():                               # right panel HAS a ylabel
        ext = ylab.get_window_extent(renderer=rnd)
        region_right = inv.transform((ext.x0, 0))[0] - 0.008
    else:                                             # 3-D panel: no ylabel
        region_right = xs[i + 1] - 0.008
    a = FancyArrowPatch((xr + 0.006, yc), (region_right, yc), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK)
    fig.add_artist(a)
    xc_lbl = 0.5 * (xr + region_right)
    txt = fig.text(xc_lbl, yc + 0.035, lab, ha="center", va="bottom",
                   fontsize=FS - 2, color=GREY, style="italic", linespacing=1.0)
    fig.canvas.draw()                                 # clamp right edge into the free region
    tb = txt.get_window_extent(renderer=rnd)
    half = 0.5 * (inv.transform((tb.x1, 0))[0] - inv.transform((tb.x0, 0))[0])
    if xc_lbl + half > region_right:
        txt.set_x(region_right - half)

# --------------------------------------------------------------------------- save
out_pdf = os.path.join(FIG_OUT, "fig1_pipeline.pdf")
out_png = "/tmp/fig1_pipeline_new.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.03)
print("wrote", out_pdf)
print("wrote", out_png)
