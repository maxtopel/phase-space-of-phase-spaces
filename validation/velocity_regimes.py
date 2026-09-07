#!/usr/bin/env python3
"""Reproduce Component 6 (velocity regime detection) -- transport-plan velocity
steps mark canonical regime transitions.

Reads the F-001 structured finding (1000-shuffle permutation null + BH-FDR at
q=0.05) directly from the FINDINGS.jsonl artifact: 6 of 7 macroeconomic indicators show
significant velocity steps with peak sigma = 4.9 (Solow, 1978-05). Plots a single panel that
puts each detection on a calendar timeline against the regime backdrop and
annotates the named events (Volcker, Draghi, COVID, ...).

Writes figures/fig_velocity.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from aoa_repro import config, plotting as P
P.set_style()


# -------------------------------------------------------------- load F-001
F001 = None
for line in open(config.ATLAS_AOA / "FINDINGS.jsonl"):
    d = json.loads(line)
    if d.get("id") == "F-001":
        F001 = d; break
assert F001 is not None, "F-001 not found in FINDINGS.jsonl"
results = F001["claim_structured"]["results"]
print("F-001 indicators detected:", list(results.keys()))


# -------------------------------------------------------------- transform
INDICATORS = ["Solow", "Unemployment", "Phillips", "Full", "Inflation", "GDP", "Interest"]
DATES = []; SIG = []; LABELS = []; REGS = []; PASS = []
for ind in INDICATORS:
    r = results[ind]
    DATES.append(datetime.strptime(r["date"], "%Y-%m"))
    SIG.append(r["sigma"])
    LABELS.append(ind)
    REGS.append(r.get("regime", ""))
    # F-001 reports 6/7 pass BH-FDR q=0.05; indicators flagged borderline by F-001
    # (e.g. Interest: ECB-only post-1999) are shown as borderline regardless of nominal p.
    PASS.append(r["p"] <= 0.05 and "borderline" not in str(r.get("note", "")).lower())
years = np.array([d.year + (d.month - 1) / 12 for d in DATES])
sig = np.array(SIG); reg = np.array(REGS); passmask = np.array(PASS)


# -------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.1, 3.6))
ax = fig.add_axes([0.07, 0.16, 0.91, 0.74])

# regime backdrop
events = [(1978.0, "Volcker", "#fdebd0"),
          (2007.0, "GFC",     "#f5c6cb"),
          (2012.0, "Draghi",  "#d4edda"),
          (2020.0, "COVID",   "#d6eaf8")]
for x, lbl, fc in events:
    ax.add_patch(Rectangle((x - 0.6, 0), 1.2, 6.0, fc=fc, ec="none", alpha=0.55, zorder=0))
    ax.text(x, 5.7, lbl, ha="center", va="top", fontsize=7, color="#444", style="italic")

# null threshold line at sigma=2 (~ p=0.05 two-sided)
ax.axhline(2.0, ls="--", color=P.GREY, lw=0.5, zorder=1)
ax.text(2024, 2.06, r"$\sigma=2$ ($p\sim0.05$)", fontsize=6, color=P.GREY, ha="right", va="bottom")

# events
ax.vlines(years, 0, sig, color="#4a5662", lw=0.6, zorder=2)
ax.scatter(years[passmask],  sig[passmask],  s=180 * sig[passmask] / sig.max(),
           c=P.CHAOS, edgecolors="k", linewidths=0.5, zorder=3,
           label=f"significant ({passmask.sum()}/7)")
ax.scatter(years[~passmask], sig[~passmask], s=180 * sig[~passmask] / sig.max(),
           c="white",  edgecolors=P.CHAOS, linewidths=0.8, zorder=3,
           label=f"borderline ({(~passmask).sum()}/7)")

for x, y, lbl in zip(years, sig, LABELS):
    ax.text(x + 0.4, y + 0.05, lbl, fontsize=7, va="bottom", ha="left", color=P.INK)

ax.set_xlim(1972, 2024); ax.set_ylim(0, 6.0)
ax.set_xlabel("year of detected velocity step", fontsize=8)
ax.set_ylabel(r"$\sigma$ above permutation null", fontsize=8)
ax.set_title(
    r"Transport-plan velocity steps mark canonical regime transitions "
    r"(6/7 indicators, peak $\sigma=4.9$; 1000-shuffle, BH-FDR $q=0.05$)",
    fontsize=8.5)
ax.legend(loc="upper left", fontsize=6.5, frameon=False)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

P.savefig(fig, "fig_velocity")
print("wrote figures/fig_velocity.pdf")
