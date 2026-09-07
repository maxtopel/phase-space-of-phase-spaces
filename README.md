# Phase spaces of phase spaces

Reproduction code and artifacts for *Phase spaces of phase spaces:
reconstruction of the geometry of the macroeconomy* (M. Topel).

Everything the paper reports can be regenerated from this repository. Code,
small data anchors, and the derived artifacts the paper quotes all ship here.
No private tree is required at runtime.

## Quick start

```bash
pip install -r requirements.txt
python3 make_figures.py        # rebuilds every paper figure in ~1 minute
```

Figures land in `figures/`. Each script also runs standalone, for example
`python3 scripts/fig3_phillips_coupling.py`.

## Layout

```
make_figures.py     one command, rebuilds all figures and the Chow table
scripts/            one script per paper figure or table, named for what it makes
validation/         the checks the paper cites but does not plot
pipeline/           the PSoPS construction itself (corpus rules, barycenter,
                    per-series plans, coordinates, dimension estimation)
aoa_repro/          shared library (config, data IO, figure style)
vendor_atlas/       vendored analysis modules (see NOTICE)
data/               FRED series and result anchors the scripts read
artifacts/          corpus-level artifacts every paper number traces to
caches/             computation caches so figures rebuild in seconds
tables/             generated LaTeX table fragments
```

## What makes what

| Paper object | Script |
|---|---|
| Fig. 1 pipeline schematic | `scripts/fig1_pipeline.py` |
| Fig. 2 validation on known systems | `scripts/fig2_validation.py` |
| Fig. 3 Phillips coupling | `scripts/fig3_phillips_coupling.py` |
| Fig. 4 the universal PSoPS | `scripts/fig4_universal_psops.py` |
| Fig. 5 stability | `scripts/fig5_stability.py` |
| Fig. 6 nonlinearity | `scripts/fig6_nonlinearity.py` |
| Fig. 7 law attractors | `scripts/fig7_law_attractors.py` |
| Fig. 8 velocity moments | `scripts/fig8_velocity_moments.py` |
| Table I dimensions | `scripts/table1_dimensions.py` |
| Table IV Chow ranks | `scripts/table4_chow_ranks.py` |

Every `Source:` pointer in the paper names the script here that regenerates
that result. The construction chain runs stage 1 (raw series to per-series
cost matrices), stage 2 (cost matrices to the universal barycenter and
per-series transport plans, `pipeline/barycenter_plans.py`), stage 3 (plans to
Hellinger diffusion map to PSoPS coordinates, `pipeline/coordinates.py`), and
stage 4 (coordinates to results and figures). The per-law pipelines run all
four stages end to end from the raw CSVs in `data/fred/`.

## What is deliberately not here

- The four transport-plan binaries (`artifacts/*.dat`, ~356 MB) are deposited
  on Zenodo, [doi:10.5281/zenodo.22602656](https://doi.org/10.5281/zenodo.22602656),
  rather than tracked in git. Download them into `artifacts/` to run the
  corpus-level scripts.
- The 2.7 GB stage-1 cost-matrix cache is not distributed. It is derived from
  provider data mined under per-provider terms and is available on reasonable
  request. Everything shipped here is a non-invertible geometric summary from
  which no raw observation values can be recovered.
- Two negative results are documented rather than claimed: transport-plan
  family recovery on the synthetic zoo fails (`validation/dysts_families.py`),
  and out-of-domain controls are not separated from macro data at matched
  solver settings (`validation/controls.py`).

## The PSoPS construction chain

Stage 1 turns raw series into per-series cost matrices (the stage-1 cache is
the on-request item above). Stage 2 fits the universal barycenter and solves
the per-series transport plans (`pipeline/barycenter_plans.py`). Stage 3 turns
plans into Hellinger diffusion-map coordinates (`pipeline/coordinates.py`).
Stage 4 turns coordinates into results and figures (`scripts/`). The per-law
pipelines run all four stages end to end from the raw CSVs in `data/fred/`.

## Recomputing from scratch

With the shipped caches, `make_figures.py` runs in about a minute. To redo the
underlying computations, pass `--recompute` to
`scripts/fig3_phillips_coupling.py`, `scripts/fig5_stability.py`,
`scripts/fig8_velocity_moments.py` or `scripts/fig4_universal_psops.py`
(elsewhere, delete the cache file in `caches/`). Reproducing the headline
dimension (`pipeline/compute_dimensions.py`) builds a dense distance matrix
and needs about 8 GB of RAM. Everything else runs on a laptop.
