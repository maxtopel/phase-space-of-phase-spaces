"""aoa_v3.corpus_clusters — Full-corpus domain clusters for Level-1 AoA.

Organizes the available FRED corpus (~160 series) into macroeconomically
coherent domain clusters, with per-series pre-embed transform rules
(log-return for exponentially growing series, raw for bounded/stationary).

This is the input to the Level-1 barycentric prediction pipeline:
each cluster defines one GW barycenter whose "typical dynamics" are
the predictive basis for targets.

The taxonomy is based on standard FRED category structure + macro
modeling convention. Per domain we include all series in the available
corpus (`aoa_v3.data.FRED_DIR`) that plausibly belong to the domain.
At run time, we filter by window-coverage to get the per-target cluster
membership.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from aoa_v3.data import _load_series


# ----------------------------------------------------------- domain taxonomy
DOMAIN_MEMBERS: dict[str, list[str]] = {
    "real_activity": [
        "GDPC1", "GDPDEF", "GDPPOT", "INDPRO", "IPMAN", "CMRMTSPL",
        "TCU", "ISRATIO", "CFNAI", "CFNAIMA3",
        "NEWORDER", "DGORDER", "ACDGNO",
        "HOUST", "PERMIT", "HSN1F", "CSUSHPINSA", "SPCS20RSA",
        "RSXFS", "EXHOSLUSM495S",
        "OPHNFB",                             # productivity (output/hour)
    ],
    "labor": [
        "UNRATE", "NROU", "PAYEMS", "CIVPART", "AWHAETP",
        "CES0500000003", "ICSA", "CCSA",
        "JTSJOL", "JTSQUR", "SAHMREALTIME", "RECPROUSM156N",
    ],
    "inflation": [
        "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE",
        "PCETRIM12M159SFRBDAL", "MEDCPIM158SFRBCLE", "TRMMEANCPIM158SFRBCLE",
        "PPIFIS", "CPALTT01USM657N",
        "MICH", "T5YIE", "T10YIE",
        "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP",
    ],
    "monetary_rates": [
        "FEDFUNDS", "DFF", "DPRIME", "MPRIME",
        "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
        "DTB3", "DTB6",
        "DFII5", "DFII10", "DFII20", "DFII30",
        "DAAA", "DBAA", "AAA10Y", "BAA10Y",
        "M1SL", "M2SL", "MZMSL", "BOGMBASE",
        "T10Y2Y", "T10Y3M", "T10YFF", "T5YFF", "TEDRATE",
    ],
    "credit_financial": [
        "BAMLC0A0CM", "BAMLC0A1CAAA", "BAMLC0A4CBBB",
        "BAMLH0A0HYM2", "BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC",
        "ANFCI", "NFCI", "STLFSI2",
        "MORTGAGE15US", "MORTGAGE30US",
        "BUSLOANS", "TOTCI", "TOTLL", "REALLN",
        "CDSP", "TDSP", "MDSP",
        "DRALACBS", "DRCCLACBS", "DRCLACBS",
        "CORALACBS", "CORCACBS", "CORCCACBS",
    ],
    "sentiment": [
        "UMCSENT", "CSCICP03USM665S", "CONSUMER",
    ],
    "external": [
        "DEXCAUS", "DEXJPUS", "DEXSZUS", "DEXUSAL", "DEXUSEU", "DEXUSUK",
        "DTWEXAFEGS", "DTWEXBGS", "DTWEXEMEGS",
        "BOPGSTB", "IEABC",
    ],
    "capital": [
        "NCBDBIQ027S", "BCNSDODNS",           # corporate debt / balance sheets
    ],
}


# ---------------------- per-series pre-embed transform (logret vs raw)
# Rule: exponentially growing series (levels, aggregates, indexes) use
# log-returns before Takens embedding so DMAP captures business-cycle
# dynamics rather than secular trend. Bounded series (rates, ratios,
# sentiment indexes, unemployment-level variables) use raw levels.
LOGRET_SERIES: set[str] = {
    # Real activity: GDP/production/housing/orders/retail all level-trending
    "GDPC1", "GDPDEF", "GDPPOT", "INDPRO", "IPMAN", "CMRMTSPL",
    "NEWORDER", "DGORDER", "ACDGNO",
    "HOUST", "PERMIT", "HSN1F", "CSUSHPINSA", "SPCS20RSA",
    "RSXFS", "EXHOSLUSM495S", "OPHNFB",
    # Labor levels (payrolls)
    "PAYEMS", "CES0500000003", "ICSA", "CCSA", "JTSJOL",
    # Prices (level indexes)
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIFIS",
    # Commodity prices
    "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP",
    # Money aggregates + velocity
    "M1SL", "M2SL", "MZMSL", "BOGMBASE", "M2V_DERIVED",
    # Loans (level)
    "BUSLOANS", "TOTCI", "TOTLL", "REALLN",
    # Capital
    "NCBDBIQ027S", "BCNSDODNS",
}


def preembed_for(series_id: str) -> str | None:
    """Return the appropriate pre-embed transform for a series, or None."""
    return "logret" if series_id in LOGRET_SERIES else None


# ----------------------------------------------------- cluster construction
def get_cluster_members(
    domain: str,
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    exclude: list[str] | None = None,
    min_members: int = 3,
) -> list[str]:
    """Return series IDs in the given domain that cover the full window,
    excluding any in `exclude` (typically the target series).

    If fewer than `min_members` qualify, returns an empty list —
    caller skips the cluster.
    """
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    exclude_set = set(exclude or [])
    members = []
    for fid in DOMAIN_MEMBERS.get(domain, []):
        if fid in exclude_set:
            continue
        try:
            s = _load_series(fid)
            if s.index[0] <= start and s.index[-1] >= end:
                members.append(fid)
        except Exception:
            continue
    if len(members) < min_members:
        return []
    return members


def get_all_clusters(
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    exclude: list[str] | None = None,
    min_members: int = 3,
) -> dict[str, list[str]]:
    """Return {domain: [members]} for every domain with at least
    `min_members` window-covering series (after exclusion)."""
    return {
        dom: members
        for dom in DOMAIN_MEMBERS
        for members in [get_cluster_members(
            dom, window_start, window_end, exclude=exclude, min_members=min_members,
        )]
        if members
    }
