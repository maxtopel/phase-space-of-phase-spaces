"""Shared matplotlib style, palettes, and panel helpers.

One place for the paper's visual language so every figure matches.
"""
import os
import matplotlib
if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

# --- palette ----------------------------------------------------------------
INK   = "#1f2a36"      # body text / strong lines
GREY  = "#9aa6b2"      # de-emphasised lines/axes
TEAL  = "#2a7f9e"      # primary accent (pipeline panels, PSoPS category)
CHAOS = "#d1495b"      # nonlinear-chaos category (Lorenz, "above threshold")
NOISE = "#1f6f8b"      # linear-Gaussian null category (Ornstein-Uhlenbeck)
# NOISE is intentionally distinct from TEAL: TEAL is the brand accent, NOISE
# carries categorical meaning ("linear null") in chaos-vs-noise comparisons.

INDICATOR_COLORS = {
    "interest_rate": "#d1495b", "gdp": "#2a7f9e", "inflation": "#edae49",
    "exchange_rate": "#66a182", "labor": "#8d6cab", "unemployment": "#c75dab",
    "credit": "#3b7a57", "trade": "#e07a5f", "capital": "#5b8e7d",
}
OTHER_COLOR = "#d8dee4"


def set_style():
    """Paper-wide figure style.

    mathtext.fontset='cm' matches REVTeX's Computer Modern body text; without it
    matplotlib falls back to DejaVu and the figure visibly clashes with the page.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def render_attractor_3d(ax, X, color=TEAL, lw=0.35, alpha=0.85, view=(22, -60)):
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color=color, lw=lw, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0)
    ax.view_init(*view)


def clean_2d(ax, spine_color=GREY):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(spine_color)


def savefig(fig, name, preview=True):
    """Save a figure to the paper's figures/ dir (PDF), with tight bounding box."""
    config.ensure_dirs()
    pdf = config.FIG_OUT / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    if preview:
        fig.savefig(f"/tmp/{name}.png", dpi=190, bbox_inches="tight", pad_inches=0.05)
    return pdf
