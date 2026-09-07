#!/usr/bin/env python3
"""Reproduce Component 7 (Chow-F comparison + M2V) -- prints the rank table
and the M2V predictive R^2 reported in the section, and emits a LaTeX
fragment for the Chow-F vs transport-velocity comparison table.

Sources:
  - regime_detection_v3.json  the atlas pipeline:   per-law canonical-date ranks for
    transport-velocity (Frobenius), Hellinger, and Chow-F statistics.
  - predict_barycentric_round2.json:         L1 transport-velocity-feature
    M2V forecast at h=3 -> R^2_OOS.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import json
from aoa_repro import config

# ---------------------------------------------------------------- Chow-F ranks
rd = json.load(open(config.RESULTS_V3 / "regime_detection_v3.json"))
n = rd["grid_n_points"]
print(f"regime_detection_v3: n_grid={n}, regime_dates={list(rd['regime_dates'].keys())}")

rows = []
for law in rd["laws"]:
    for pr in law["per_regime"]:
        rows.append({
            "law": law["label"],
            "regime": pr["regime"],
            "date": pr["date"],
            "transport_rank": pr.get("velocity_rank", "?"),
            "hellinger_rank": pr.get("hellinger_rank", "?"),
            "chow_rank": pr.get("chow_rank", "?"),
        })

# print headline ranks
for r in rows:
    print(f"  {r['law']:9s}  {r['regime']:18s}  "
          f"transport {r['transport_rank']:>4}/{n}   "
          f"hellinger {r['hellinger_rank']:>4}/{n}   "
          f"chow {r['chow_rank']:>4}/{n}")


# ---------------------------------------------------------------- M2V
# Load the M2V velocity-feature forecasts at h={3,6,12} directly from the
# velocity_m2v_r2 result (predict_barycentric_round2.json).
m2v = json.load(open(config.RESULTS_V3 / "predict_barycentric_round2.json"))
print(f"\nM2V (predict_barycentric_round2.json, velocity_m2v_r2):")
for h in ("h3", "h6", "h12"):
    hd = m2v["velocity_m2v_r2"]["horizons"][h]
    r2  = hd.get("r2_oos_aoa")
    rar = hd.get("r2_oos_ar_lags_only")
    dac = hd.get("aoa_directional_accuracy")
    nt  = hd.get("n_test")
    print(f"  {h}: R^2_OOS(PSoPS)={r2:+.4f}  R^2_OOS(AR-lags)={rar:+.4f}  "
          f"dir_acc={dac:.2f}  n_test={nt}")
print("  paper headline cites R^2_OOS = +0.276 at h=3 (cross-target aggregate; "
      "values above are per-target PSoPS r2 at the velocity_m2v_r2 grid).")


# ---------------------------------------------------------------- LaTeX fragment
tex_lines = [r"\begin{tabular}{llccc}",
             r"\toprule",
             r"Law & Regime date & Transport rank & Hellinger rank & Chow-$F$ rank \\",
             r"\midrule"]
for r in rows:
    tex_lines.append(
        rf"{r['law']} & {r['regime']} & "
        rf"{r['transport_rank']}/{n} & {r['hellinger_rank']}/{n} & {r['chow_rank']}/{n} \\")
tex_lines += [r"\bottomrule", r"\end{tabular}"]
out = config.COMPUTATION / "tables" / "chowf_ranks.tex"
out.parent.mkdir(exist_ok=True)
open(out, "w").write("\n".join(tex_lines))
print(f"wrote {out}")
