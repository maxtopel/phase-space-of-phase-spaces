"""Data access: raw series from Atlas, manifests, and cached PSoPS artifacts.

Raw series live under config.DATA_ROOT as <source>/<key>.csv with columns
(date, value). Cached PSoPS artifacts (used to CONFIRM against, or to scale past
what is tractable to recompute here) live under the Atlas checkpoints/results.
"""
import json
import numpy as np

from . import config


# ---------------------------------------------------------------- raw series
def load_dated_series(path):
    """Load a `date,value` CSV from a full filesystem path.

    Returns (dates, values) as a tuple of ndarrays (dates are datetime
    objects, values float). Rows with non-parseable values are skipped.
    """
    from datetime import datetime
    import csv
    dates, vals = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                d = datetime.strptime(row[0][:10], "%Y-%m-%d")
                v = float(row[1])
            except (ValueError, TypeError):
                continue
            dates.append(d); vals.append(v)
    return np.array(dates), np.array(vals, float)


def load_series(key, root=None):
    """Load a single scalar series by manifest key (e.g. 'boe/boe_mill_...').

    Returns a float ndarray of values (date column dropped).
    """
    root = config.DATA_ROOT if root is None else root
    path = (root / key)
    if not str(path).endswith(".csv"):
        path = root / (key + ".csv")
    import csv
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) >= 2:
                try:
                    vals.append(float(row[1]))
                except ValueError:
                    pass
    return np.asarray(vals, float)


def load_manifest(name="corpus_manifest.json"):
    with open(config.DATA_ROOT / name) as f:
        return json.load(f)


# ---------------------------------------------------------------- cached artifacts
def load_m5_features():
    """The universal-PSoPS per-series coordinates + labels (241,882 x 7).

    Returns dict with keys: features, keys, sources, indicators, countries, n_points.
    """
    z = np.load(config.CHECKPOINTS / "family_discovery" / "m5_features_241k.npz",
                allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_result(filename, where="v3"):
    """Load a cached result JSON. where in {'v3','checkpoints','convergence'}."""
    base = {
        "v3": config.RESULTS_V3,
        "checkpoints": config.CHECKPOINTS,
        "convergence": config.CHECKPOINTS / "convergence_battery",
    }[where]
    with open(base / filename) as f:
        return json.load(f)


def robust_standardize(F, clip=8.0):
    """Median/MAD standardize columns + clip (used before UMAP / projections)."""
    F = np.asarray(F, float)
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-9
    return np.clip((F - med) / mad, -clip, clip)
