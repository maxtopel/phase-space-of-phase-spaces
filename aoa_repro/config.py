"""Central paths, seeds, and vendored-module access for PSoPS reproduction.

All figure/result scripts import from here so paths and the shared seed lives here (scripts may pin their own).
Paths point at the private atlas tree when present and fall back to the
vendored anchors under this repository: data/ otherwise.
"""
import os
import sys
from pathlib import Path

# --- repository / atlas locations -------------------------------------------
COMPUTATION  = Path(__file__).resolve().parents[1]           # this repo, whatever it is named
PROJECT_ROOT = COMPUTATION.parent                             # the paper directory (if cloned inside it)
# main.tex includes ../figures when this repo sits inside the paper tree; a
# standalone clone writes to ./figures instead
FIG_OUT      = (PROJECT_ROOT / "figures") if (PROJECT_ROOT / "main.tex").exists() \
               else (COMPUTATION / "figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

ATLAS_ROOT   = Path(os.environ.get("AOA_ATLAS_ROOT", str(Path.home() / "Desktop" / "atlas")))
DATA_ROOT    = Path(os.environ.get("AOA_DATA_ROOT", ATLAS_ROOT / "data" / "streams" / "economics"))
ATLAS_AOA    = Path(os.environ.get("AOA_V1_ROOT", ATLAS_ROOT / "research" / "core" / "PSoPS"))
ATLAS_V3     = Path(os.environ.get("AOA_V3_ROOT", ATLAS_ROOT / "research" / "core" / "AoA_v3.0"))
RESULTS_V3   = ATLAS_V3 / "results"
CHECKPOINTS  = ATLAS_AOA / "checkpoints"

# --- public-clone fallback ---------------------------------------------------
# When the private atlas tree is absent (any clone that is not the author's
# machine), fall back to the vendored anchors under this repository: data/. Every
# path is still individually overridable with the env vars above.
_ANCHORS = COMPUTATION / "data" / "atlas_anchors"
if not DATA_ROOT.exists() and (COMPUTATION / "data" / "fred").exists():
    DATA_ROOT = COMPUTATION / "data"
if not ATLAS_AOA.exists() and _ANCHORS.exists():
    ATLAS_AOA = _ANCHORS
if not RESULTS_V3.exists() and (_ANCHORS / "results").exists():
    RESULTS_V3 = _ANCHORS / "results"
if not CHECKPOINTS.exists() and (_ANCHORS / "checkpoints").exists():
    CHECKPOINTS = _ANCHORS / "checkpoints"

_VENDOR = Path(__file__).resolve().parent / "_vendor"
VENDOR_ATLAS = COMPUTATION / "vendor_atlas"

# --- reproducibility ---------------------------------------------------------
SEED = 17

# --- vendored atlas machinery (embed/diffuse/transport/...) ------------------
def use_vendor():
    """Put the vendored atlas modules on sys.path so `import embed` etc. work.

    vendor_atlas/ carries the full set (aoa_v3 package, landscape, transport,
    synthetic_gate, eps008 layer, plus the config/diffuse/embed siblings they
    import); _vendor/ keeps the original minimal trio for back-compat.
    """
    for q in (str(VENDOR_ATLAS), str(_VENDOR)):
        if q not in sys.path:
            sys.path.insert(0, q)
    p = str(_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)

def ensure_dirs():
    FIG_OUT.mkdir(exist_ok=True)
