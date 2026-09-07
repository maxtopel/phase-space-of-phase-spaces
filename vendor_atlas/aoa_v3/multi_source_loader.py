"""aoa_v3.multi_source_loader — Unified loader across all macro data sources.

Atlas has ~400k time series files across 20+ sources (FRED, OECD, IMF,
BIS, ECB, BOE, WRDS, statcan, worldbank, PWT, ...) with a common
`date,value` CSV schema. This module provides:

  1. SourceCatalog — discover which series IDs each source has.
  2. load_multisource_series(source, series_id) → pd.Series (monthly).
  3. build_multisource_corpus(source_filters, domain_classifier, window)
     → {domain: [(source, series_id)]} expanded from the base FRED
     corpus to include international/cross-source coverage.

The goal: scale the Level-1 AoA barycenter from ~100 FRED series to
the 1k-10k range the paper envisions, without losing the unified
domain-cluster structure.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pandas as pd

from aoa_v3.data import _to_monthly

# Base directory — matches layout of atlas/data/streams/economics/{source}/*.csv
import os as _os
ATLAS_ECON = Path(_os.environ["AOA_ATLAS_DATA"]) if "AOA_ATLAS_DATA" in _os.environ else (Path.home() / "Desktop" / "atlas" / "data" / "streams" / "economics" if (Path.home() / "Desktop" / "atlas" / "data" / "streams" / "economics").exists() else Path(__file__).resolve().parents[2] / "data")


# --------------------------------------------------------- source discovery
SUPPORTED_SOURCES = {
    "fred":       {"prefix": "fred_",    "suffix": ".csv"},
    "oecd":       {"prefix": "oecd_",    "suffix": ".csv"},
    "imf":        {"prefix": "imf_",     "suffix": ".csv"},
    "bis":        {"prefix": "bis_",     "suffix": ".csv"},
    "ecb":        {"prefix": "ecb_",     "suffix": ".csv"},
    "boe":        {"prefix": "boe_",     "suffix": ".csv"},
    "pwt":        {"prefix": "pwt_",     "suffix": ".csv"},
    "worldbank":  {"prefix": "wb_",      "suffix": ".csv"},
    "statcan":    {"prefix": "statcan_", "suffix": ".csv"},
    "rba":        {"prefix": "rba_",     "suffix": ".csv"},
    "bcb":        {"prefix": "bcb_",     "suffix": ".csv"},
    "sf_fed":     {"prefix": "sf_fed_",  "suffix": ".csv"},
    "snb":        {"prefix": "snb_",     "suffix": ".csv"},
}


@functools.lru_cache(maxsize=None)
def list_series(source: str) -> list[str]:
    """Return the list of series IDs available in a source."""
    if source not in SUPPORTED_SOURCES:
        raise KeyError(f"unsupported source {source!r}")
    cfg = SUPPORTED_SOURCES[source]
    dir_ = ATLAS_ECON / source
    if not dir_.exists():
        return []
    out = []
    for p in dir_.glob(f"{cfg['prefix']}*{cfg['suffix']}"):
        sid = p.name[len(cfg["prefix"]):-len(cfg["suffix"])]
        out.append(sid)
    return sorted(out)


@functools.lru_cache(maxsize=4096)
def load_multisource_series(source: str, series_id: str,
                             agg: str = "mean") -> pd.Series:
    """Load a series from any supported source, return a monthly pd.Series."""
    if source not in SUPPORTED_SOURCES:
        raise KeyError(f"unsupported source {source!r}")
    cfg = SUPPORTED_SOURCES[source]
    path = ATLAS_ECON / source / f"{cfg['prefix']}{series_id}{cfg['suffix']}"
    if not path.exists():
        raise FileNotFoundError(f"{path}")
    df = pd.read_csv(path, parse_dates=False)
    if "value" not in df.columns or "date" not in df.columns:
        raise ValueError(f"{path}: expected 'date,value' schema, got {list(df.columns)}")

    # Date parsing — handle several source formats.
    dates = _parse_dates(df["date"])
    valid = dates.notna()
    df = df.loc[valid].copy()
    df["date"] = dates[valid]
    df = df.sort_values("date")

    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=pd.DatetimeIndex(df["date"]),
        name=f"{source}/{series_id}",
    ).dropna()
    return _to_monthly(s, aggregation=agg)


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse heterogeneous date formats (FRED YYYY-MM-DD, OECD YYYY-Qn,
    IMF YYYY annual). Returns NaT for unparseable rows.
    """
    raw = raw.astype(str).str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    # Handle OECD quarterly: 2010-Q1 → 2010-03-31 (quarter end)
    mask = out.isna() & raw.str.match(r"^\d{4}-Q[1-4]$")
    if mask.any():
        yq = raw[mask].str.split("-Q", expand=True)
        q = yq[1].astype(int)
        mm = (q * 3)
        out[mask] = pd.to_datetime(yq[0] + "-" + mm.astype(str) + "-01") + \
                    pd.offsets.MonthEnd(0)
    # Handle annual: YYYY → YYYY-12-31
    mask = out.isna() & raw.str.match(r"^\d{4}$")
    if mask.any():
        out[mask] = pd.to_datetime(raw[mask] + "-12-31")
    return out


# ---------------------------------------- multi-source corpus construction
DEFAULT_DOMAIN_PATTERNS: dict[str, list[str]] = {
    # Patterns against series-id text; first-match wins.
    "real_activity": [
        # GDP-like
        r"(?i)^(GDP|FYGDP|GDPPOT|gdpc|rgdp)",
        r"(?i)(ppp.*gdp|gdp.*ppp|B1GQ|real.*gdp|rgdpe|rgdpo)",
        # Production
        r"(?i)^IND(PRO|US)", r"(?i)IPMAN", r"(?i)PRODUCTION",
        # Capacity / orders / inventories
        r"(?i)^TCU", r"(?i)CAPACITY.*UTIL", r"(?i)ISRATIO",
        r"(?i)NEW.*ORDER", r"(?i)DGORDER", r"(?i)ACDGNO",
        # Housing
        r"(?i)HOUST", r"(?i)PERMIT", r"(?i)HSN1F",
        r"(?i)HPRICE|HOUSE.*PRICE|CSUSHPINSA|SPCS",
        # Retail
        r"(?i)RSXFS|RETAIL",
        # CFNAI-style indices
        r"(?i)CFNAI",
        # Productivity
        r"(?i)^OPHNF|productivity",
    ],
    "labor": [
        r"(?i)^UNRATE", r"(?i)NROU",
        r"(?i)PAYEMS|NON.*FARM.*PAYROLL",
        r"(?i)CIVPART|participation",
        r"(?i)^AWHAETP|AVG.*HOURS|HOURS.*EMPLOY",
        r"(?i)^CES0500|AVG.*HOURLY.*EARN|wage|earning",
        r"(?i)ICSA|CCSA|claim",
        r"(?i)JTSJOL|JTSQUR|JOLT",
        r"(?i)SAHM|RECESSION.*PROB",
        r"(?i)unemploy", r"(?i)employ",
    ],
    "inflation": [
        r"(?i)^CPI|^PCEPI|PCEPILFE|CORE.*CPI|CPIAUCSL|CPILFESL",
        r"(?i)TRIM.*CPI|MEDCPI|MEAN.*CPI",
        r"(?i)^PPI|producer.*price",
        r"(?i)CPALTT01",
        r"(?i)MICH|inflation.*expect",
        r"(?i)T5YIE|T10YIE|BREAKEVEN",
        r"(?i)OIL.*WTI|DCOIL|BRENT", r"(?i)DHHNGSP|gas",
    ],
    "monetary_rates": [
        r"(?i)^FEDFUNDS|^DFF|^DPRIME|^MPRIME|FED.*RATE",
        r"(?i)^DGS\d+|TREAS.*YIELD|^DFII\d+",
        r"(?i)^DTB\d+|T-?BILL",
        r"(?i)^DAAA|^DBAA|^AAA10Y|^BAA10Y|CORP.*BOND",
        r"(?i)^M[012]SL|MZMSL|money.*stock|BOGMBASE",
        r"(?i)T10Y2Y|T10Y3M|T10YFF|T5YFF|yield.*spread|YIELD_CURVE",
        r"(?i)TEDRATE|TED.*spread",
        r"(?i)^IR3TIB|^IRLTLT",
    ],
    "credit_financial": [
        r"(?i)BAMLC|BAMLH|HIGH.*YIELD|INVESTMENT.*GRADE",
        r"(?i)ANFCI|NFCI|STLFSI|FINANCIAL.*CONDITION",
        r"(?i)MORTGAGE",
        r"(?i)^BUSLOANS|^TOTCI|^TOTLL|^REALLN|LOAN",
        r"(?i)^CDSP$|^TDSP|^MDSP|delinqu|charge.*off",
        r"(?i)^DRALACBS|^DRCCLACBS|^DRCLACBS|DRCRELEXFACBS|DRSFRMACBS",
        r"(?i)^CORALACBS|^CORCACBS|^CORCCACBS",
    ],
    "sentiment": [
        r"(?i)UMCSENT|CONSUMER.*SENT|CSCICP|CONFIDENCE",
    ],
    "external": [
        r"(?i)^DEX[A-Z]{4}|EXCHANGE.*RATE|FX", r"(?i)^DTWEXAFE|^DTWEXBGS|^DTWEXEME|effective.*exchange",
        r"(?i)BOPGSTB|trade.*balance|current.*account",
        r"(?i)IEABC",
    ],
    "capital": [
        r"(?i)NCBDBIQ|BCNSDODNS|NETWORTH|balance.*sheet",
    ],
}


def classify_series(series_id: str,
                     patterns: dict[str, list[str]] | None = None) -> str | None:
    """Return the first domain whose pattern matches `series_id`, or None."""
    patterns = patterns or DEFAULT_DOMAIN_PATTERNS
    for domain, pats in patterns.items():
        for pat in pats:
            if re.search(pat, series_id):
                return domain
    return None


def build_multisource_corpus(
    sources: list[str] | None = None,
    window_start: str | pd.Timestamp | None = None,
    window_end: str | pd.Timestamp | None = None,
    exclude: set[tuple[str, str]] | None = None,
    min_members_per_cluster: int = 3,
    max_members_per_cluster: int | None = None,
    patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Return {domain: [(source, series_id), ...]} from multi-source corpus.

    Only series whose index covers the full [window_start, window_end] are
    included. Classification via `classify_series` regex rules.

    NOTE: This is discovery-only — it does NOT load series. Call
    load_multisource_series(src, sid) lazily at barycenter-build time.
    """
    sources = sources or list(SUPPORTED_SOURCES)
    exclude = exclude or set()
    patterns = patterns or DEFAULT_DOMAIN_PATTERNS
    start = pd.Timestamp(window_start) if window_start else None
    end = pd.Timestamp(window_end) if window_end else None

    out: dict[str, list[tuple[str, str]]] = {d: [] for d in patterns}
    for src in sources:
        for sid in list_series(src):
            key = (src, sid)
            if key in exclude:
                continue
            domain = classify_series(sid, patterns=patterns)
            if domain is None:
                continue
            # Coverage check — load lightly to peek at date range.
            try:
                s = load_multisource_series(src, sid, agg="mean")
                if len(s) < 24:
                    continue
                if start is not None and s.index[0] > start:
                    continue
                if end is not None and s.index[-1] < end:
                    continue
                out[domain].append((src, sid))
            except Exception:
                continue

    # Apply size limits.
    filtered: dict[str, list[tuple[str, str]]] = {}
    for dom, members in out.items():
        if len(members) < min_members_per_cluster:
            continue
        if max_members_per_cluster is not None and len(members) > max_members_per_cluster:
            members = members[:max_members_per_cluster]
        filtered[dom] = members
    return filtered
