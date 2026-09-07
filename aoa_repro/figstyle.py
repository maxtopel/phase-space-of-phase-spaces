"""Single source of truth for the paper's figure style.

Every figure script imports this and calls `use()` before creating a figure, so the
visual language survives regeneration. The conventions follow Nature's figure layout
(corner panel labels, no titles, minimal chrome, generous label sizes) while keeping
Computer Modern so figures match the REVTeX body text.

RULES ENCODED HERE
    panels      each row/group gets one bold letter A, B, C placed once at the far
                left of the group; subpanels within a group are labelled (i),
                (ii), (iii); the letter is never repeated on every subpanel.
                Two placement mechanisms exist: `group`/`sub` (axes coords, fine
                for simple grids) and post-canvas.draw() fig.text placement in
                FIGURE coords for exact cross-row alignment (used by the
                phillips/figure/robustness/laws scripts; candidate for a
                shared letter_rows() helper)
    titles      none, except to identify the Phillips / Okun / Solow attractors, which
                are not inferable from position (see `law_title`)
    axes        black, left and bottom spines only, 0.6pt, <=5 ticks, no minor ticks
    colour      Okabe-Ito qualitative set, colourblind-safe and well separated;
                context data in light grey so it never competes with the signal
    order       LAWS is the canonical ordering for every figure and table
"""
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt

# --- canonical ordering ------------------------------------------------------
LAWS = ("Phillips", "Okun", "Solow")

# --- Okabe-Ito, colourblind safe ---------------------------------------------
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
INK = "#000000"
CONTEXT = "#D9D9D9"      # background / "other" data, never competes
MUTED = "#8C8C8C"        # reference lines, nulls

LAW_COLORS = {"Phillips": BLUE, "Okun": VERMILLION, "Solow": GREEN}

CONCEPT_COLORS = {
    "interest_rate": PINK, "inflation": ORANGE, "gdp": SKY,
    "exchange_rate": GREEN, "labor": BLUE, "capital": VERMILLION,
    "credit": "#7F3C8D", "unemployment": "#11A579", "tfp": "#E73F74",
    "trade": "#3969AC",
}

CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"       # available for diverging data; current figures use CMAP_SEQ


def use():
    """Apply the paper-wide rcParams. Call before creating any figure."""
    plt.rcParams.update({
        # type: match the REVTeX body text
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # sizes: generous, Nature-like
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        # chrome: recede it
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "axes.grid": False,
        # marks: thin lines, small edgeless markers
        "lines.linewidth": 1.1, "lines.markersize": 3.0,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3, "legend.borderpad": 0.0,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel(ax, tag, dx=-0.16, dy=1.06):
    """Panel label placed OUTSIDE the axes, above and left of the plot area.

    Never inside: on a heatmap or a dense scatter an interior label sits on top of
    the data. Nature places the letter clear of the frame, which is what dx<0 and
    dy>1 in axes coordinates give.
    """
    # Axes3D.text takes (x, y, z, s); its text2D is the 2-D-in-axes-coords entry point
    draw = getattr(ax, "text2D", ax.text)
    draw(dx, dy, tag, transform=ax.transAxes, ha="left", va="bottom",
         fontsize=9, fontweight="bold", color=INK, clip_on=False)


ROMANS = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii")


def group(ax, letter, dx=-0.30, dy=1.06):
    """Bold group letter (A, B, C), placed ONCE on the leftmost panel of a
    row/group, clear of the frame. The letter is never repeated on sibling
    subpanels; those get `sub`."""
    draw = getattr(ax, "text2D", ax.text)
    draw(dx, dy, letter, transform=ax.transAxes, ha="left", va="bottom",
         fontsize=10, fontweight="bold", color=INK, clip_on=False)


def sub(ax, idx, dx=-0.10, dy=1.04):
    """Roman subpanel tag "(i)", "(ii)" from a 0-based index (or a string)."""
    tag = idx if isinstance(idx, str) else f"({ROMANS[idx]})"
    draw = getattr(ax, "text2D", ax.text)
    draw(dx, dy, tag, transform=ax.transAxes, ha="left", va="bottom",
         fontsize=8.5, fontweight="normal", color=INK, clip_on=False)


def law_title(ax, law):
    """The sanctioned exception: identify which attractor a panel shows."""
    ax.set_title(law, fontsize=8.5, color=INK, pad=3)


def finish(ax, xlabel=None, ylabel=None, nticks=5):
    """Apply the axis conventions: black left/bottom spines, few ticks, clear labels."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.6)
    # log axes use LogLocator, which has no nbins; only thin linear ticks
    for axis in (ax.xaxis, ax.yaxis):
        # never override a locator the caller set deliberately (log, or fixed
        # categorical ticks from set_xticks), or categorical labels vanish
        if type(axis.get_major_locator()).__name__ not in (
                "LogLocator", "NullLocator", "FixedLocator"):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nticks))
    return ax


def scatter(ax, x, y, n_hint=None, **kw):
    """Scatter with density-appropriate defaults: small edgeless markers, alpha and
    rasterisation once the cloud is large enough that individual points stop reading."""
    n = n_hint if n_hint is not None else len(x)
    kw.setdefault("s", 2.0 if n > 4000 else (6.0 if n > 500 else 18.0))
    kw.setdefault("alpha", 0.55 if n > 4000 else (0.75 if n > 500 else 0.9))
    kw.setdefault("linewidths", 0.0)
    kw.setdefault("rasterized", n > 5000)
    return ax.scatter(x, y, **kw)


def band(ax, x, lo, hi, color, alpha=0.15):
    """Uncertainty as a filled band rather than error bars."""
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def direct_label(ax, x, y, text, color, dx=0.01, fontsize=7.5):
    """Label a line at its right-hand end instead of using a legend."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=color, fontsize=fontsize, va="center", ha="left")
