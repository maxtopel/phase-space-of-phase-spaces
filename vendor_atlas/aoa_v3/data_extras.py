"""aoa_v3.data_extras — Ad-hoc FRED series loader for expanded experiments.

Loads arbitrary FRED IDs (already in atlas/data/streams/economics/fred/)
and aligns them monthly. Bypasses the LAW_SPECS registry so overnight
experiments can sweep leading-series composition without editing
aoa_v3.data.

Also provides derived/computed series (monetary velocity etc.).
"""

from __future__ import annotations

import pandas as pd

from aoa_v3.data import _load_series, _to_monthly, AggMethod


# ----------------------------------------------------- derived series
def velocity_m2() -> pd.Series:
    """M2 velocity: V_M2 = nominal GDP / M2.

    nominal GDP = GDPC1 × (GDPDEF / 100).  Returns monthly series
    covering the full overlap of GDPC1, GDPDEF, M2SL.  Fisher
    equation: MV = PY, so V = PY/M.
    """
    gdp_real = _to_monthly(_load_series("GDPC1"), aggregation="mean")
    deflator = _to_monthly(_load_series("GDPDEF"), aggregation="mean")
    m2 = _to_monthly(_load_series("M2SL"), aggregation="mean")
    common = gdp_real.index.intersection(deflator.index).intersection(m2.index)
    nominal = gdp_real.loc[common] * (deflator.loc[common] / 100.0)
    v = nominal / m2.loc[common]
    v.name = "M2V_derived"
    return v.dropna()


def load_extra_corpus(
    fred_ids: list[str],
    aggregation: dict[str, AggMethod] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load an ad-hoc panel of FRED IDs at monthly (ME) cadence.

    - Sub-monthly series aggregated per the `aggregation` dict (default
      "mean" for sub-monthly, "last" for monthly, time-interpolate for
      quarterly+).
    - Returns a complete-cases DataFrame (inner-joined on date).
    """
    aggregation = aggregation or {}
    columns: dict[str, pd.Series] = {}
    for fid in fred_ids:
        s = _load_series(fid)
        if start is not None:
            s = s.loc[start:]
        if end is not None:
            s = s.loc[:end]
        agg = aggregation.get(fid, "mean")
        columns[fid] = _to_monthly(s, aggregation=agg)
    df = pd.DataFrame(columns)
    return df.dropna()
