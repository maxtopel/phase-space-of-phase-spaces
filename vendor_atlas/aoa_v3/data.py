"""
aoa_v3.data — Law-corpus loaders for Phillips, Solow, Okun.

Single-dispatch `load_law_corpus(name)` returning a pandas DataFrame with a
monthly (ME) calendar index and one column per constituent series. Series
are pre-registered (plan.tex, pre-Step-05 commit to LESSONS.md) so the law
corpora are frozen before any barycenter/prediction run.

Pre-registered targets (2026-04-22, Step 8):
  Phillips (inflation law):  CPILFESL, UNRATE, CES0500000003, MICH, T5YIE
  Okun     (output law):     UNRATE, GDPC1, INDPRO, ICSA, TCU
  Solow    (growth law):     GDPC1, PAYEMS, OPHNFB, TCU, NCBDBIQ027S

NCBDBIQ027S (nonfinancial corporate business debt, quarterly since 1990) is
the Solow K-stock proxy. With it, Solow primary window is 1990-2019 (vs. the
1970 start for the other laws); we accept the shorter history as the cost of
having K represented at all. v3.1 may swap in a true BEA K-stock series if
available.

Data source: FRED CSVs at atlas/data/streams/economics/fred/fred_<ID>.csv.
No data movement (memory `feedback_no_data_moves`).

Resample policy (paper § 2.1.2 + atlas-research CLAUDE.md monthly-MI rule):
  - Daily / weekly (auto by median gap < 28 days): `aggregation` decides
    (default "mean"; ICSA uses "sum" per standard claims convention).
  - Monthly (28-34 days): last observation in the month.
  - Quarterly / annual (> 34 days): **time-linear interpolation** to month-end
    (not forward-fill — ffill injects a step function that corrupts monthly
    MI; R04 blocker in Step 8 panel).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd


LawName = Literal["phillips", "okun", "solow"]
AggMethod = Literal["mean", "sum", "last"]


import os as _os
ATLAS_DATA = Path(_os.environ["AOA_ATLAS_DATA"]) if "AOA_ATLAS_DATA" in _os.environ else (Path.home() / "Desktop" / "atlas" / "data" / "streams" if (Path.home() / "Desktop" / "atlas" / "data" / "streams").exists() else Path(__file__).resolve().parents[2] / "data")
# atlas layout nests fred under economics/; the repo ships data/fred directly
FRED_DIR = (ATLAS_DATA / "economics" / "fred") if (ATLAS_DATA / "economics" / "fred").exists() \
    else ATLAS_DATA / "fred"


# --- Pre-registered series per law (frozen 2026-04-22 -> LESSONS.md) ---------


@dataclass(frozen=True)
class LawSpec:
    name: str
    targets: tuple[str, ...]
    theoretical_dof_reference: int  # reference column only, NOT a gate
    # Per-series aggregation override for sub-monthly series. Default = "mean".
    # Quarterly / annual series use time-linear interpolation (fixed) —
    # this dict only maps sub-monthly series to a specific aggregation.
    aggregation: dict[str, AggMethod] = field(default_factory=dict)


LAW_SPECS: dict[LawName, LawSpec] = {
    "phillips": LawSpec(
        name="phillips",
        targets=(
            "CPILFESL",          # core CPI (inflation target)
            "UNRATE",            # unemployment
            "CES0500000003",     # avg hourly earnings, private (wage growth)
            "MICH",              # Michigan expected inflation (1y)
            "T5YIE",             # 5y breakeven inflation (daily since 2003)
        ),
        theoretical_dof_reference=3,
        aggregation={"T5YIE": "mean"},  # daily -> monthly mean
    ),
    "okun": LawSpec(
        name="okun",
        targets=(
            "UNRATE",            # unemployment
            "GDPC1",             # real GDP (quarterly; time-interpolated)
            "INDPRO",            # industrial production
            "ICSA",              # weekly initial claims (sum per month)
            "TCU",               # capacity utilization
        ),
        theoretical_dof_reference=4,
        aggregation={"ICSA": "sum"},  # weekly claims -> monthly sum (standard)
    ),
    "solow": LawSpec(
        name="solow",
        targets=(
            "GDPC1",             # real GDP (Y)
            "PAYEMS",            # total nonfarm payrolls (L)
            "OPHNFB",            # output per hour, nonfarm business (TFP proxy)
            "TCU",               # capacity utilization (K-utilization proxy)
            "NCBDBIQ027S",       # K proxy: nonfinancial corporate debt (Q, 1990+)
        ),
        theoretical_dof_reference=4,
        aggregation={},
    ),
}


# ------------------------------------------------------------- loader


@functools.lru_cache(maxsize=64)
def _load_series(fred_id: str) -> pd.Series:
    """Load a FRED CSV to a pd.Series indexed by date. LRU-cached (64 series)."""
    path = FRED_DIR / f"fred_{fred_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"FRED series not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if "value" not in df.columns:
        raise ValueError(f"{path} missing expected 'value' column")
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=df["date"],
        name=fred_id,
    )
    s = s.dropna().sort_index()
    return s


def _to_monthly(s: pd.Series, aggregation: AggMethod = "mean") -> pd.Series:
    """Resample a series to month-end (ME) cadence.

    Policy (dispatch by median inter-observation gap):
      - sub-monthly (daily/weekly):     aggregation in {mean, sum, last}
      - monthly  (28-34 day gap):       last per month (ME)
      - quarterly/annual (> 34 day):    time-linear interpolation to ME
        (not ffill — ffill produces step functions that corrupt monthly MI
        and bias derivatives to zero for 2/3 months per quarter, R04 Step 8)

    Caller supplies the sub-monthly `aggregation`. For sub-monthly series a
    terminal `.dropna()` is used so empty months (no reports received) don't
    appear; higher-level `load_law_corpus` then reconciles across series.
    """
    if len(s) < 2:
        return s
    median_gap = s.index.to_series().diff().median()
    month = pd.Timedelta(days=34)
    sub_monthly = pd.Timedelta(days=27)

    if median_gap < sub_monthly:
        # sub-monthly (daily/weekly) -> user-chosen aggregation
        if aggregation == "mean":
            return s.resample("ME").mean().dropna()
        if aggregation == "sum":
            return s.resample("ME").sum(min_count=1).dropna()
        if aggregation == "last":
            return s.resample("ME").last().dropna()
        raise ValueError(f"unknown aggregation {aggregation!r}")
    if median_gap <= month:
        # monthly -> last per ME
        return s.resample("ME").last().dropna()
    # quarterly / annual -> time-interpolate to monthly
    # resample to ME (NaN between observations), then interpolate linearly in time.
    resampled = s.resample("ME").last()
    return resampled.interpolate(method="time").dropna()


def load_law_corpus(
    name: LawName,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load a pre-registered law corpus.

    Returns
    -------
    pd.DataFrame
        Monthly (ME) index, one column per FRED ID. Rows with all-NaN are
        dropped; caller should subset via `.dropna()` / `.loc[start:end]` if
        they need a complete-cases view.
    """
    if name not in LAW_SPECS:
        raise KeyError(f"unknown law {name!r}; expected one of {list(LAW_SPECS)}")
    spec = LAW_SPECS[name]

    cols: dict[str, pd.Series] = {}
    for fid in spec.targets:
        s = _load_series(fid)
        agg = spec.aggregation.get(fid, "mean")
        cols[fid] = _to_monthly(s, aggregation=agg)

    df = pd.DataFrame(cols).sort_index()
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    return df.dropna(how="all")


def list_laws() -> list[str]:
    return list(LAW_SPECS.keys())


def describe_law(name: LawName) -> LawSpec:
    if name not in LAW_SPECS:
        raise KeyError(f"unknown law {name!r}")
    return LAW_SPECS[name]


def clear_series_cache() -> None:
    """Clear the `_load_series` LRU cache. Useful for test isolation."""
    _load_series.cache_clear()
