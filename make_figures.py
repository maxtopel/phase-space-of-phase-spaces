#!/usr/bin/env python3
"""Rebuild every paper figure in order. One command, no arguments:

    python3 make_figures.py

Each script is run in its own process; a failure stops nothing else. With the
shipped caches present the full set takes a few minutes; pass --recompute to
the individual scripts (see README) to redo the underlying computations.
"""
import subprocess
import sys
import time

FIGURES = [
    ("Fig. 1 pipeline",      "scripts/fig1_pipeline.py"),
    ("Fig. 2 validation",    "scripts/fig2_validation.py"),
    ("Fig. 3 Phillips",      "scripts/fig3_phillips_coupling.py"),
    ("Fig. 4 universal PSoPS", "scripts/fig4_universal_psops.py"),
    ("Fig. 5 stability",     "scripts/fig5_stability.py"),
    ("Fig. 6 nonlinearity",  "scripts/fig6_nonlinearity.py"),
    ("Fig. 7 laws",          "scripts/fig7_law_attractors.py"),
    ("Fig. 8 velocity",      "scripts/fig8_velocity_moments.py"),
    ("Chow-rank table",      "scripts/table4_chow_ranks.py"),
]

def main():
    failures = []
    for label, script in FIGURES:
        t0 = time.time()
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"OK    {label:22s} {script}  ({dt:.0f}s)")
        else:
            failures.append((label, script))
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            print(f"FAIL  {label:22s} {script}  ({dt:.0f}s)")
            for line in tail:
                print(f"      {line}")
    print(f"\n{len(FIGURES) - len(failures)}/{len(FIGURES)} succeeded")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
