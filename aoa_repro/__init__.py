"""aoa_repro -- reusable machinery to reproduce every figure and numerical
result in "The Geometry of the Macroeconomy".

Layout:
    config     paths, seeds, vendored-module access
    dataio     raw series + cached artifact loaders
    systems    reference dynamical systems + invariants
    pipeline   Takens embed / diffusion map / GW transport wrappers
    plotting   shared style, palettes, panel helpers
    _vendor/   verbatim copies of the Atlas machinery (embed/diffuse/transport/...)
"""
from . import config, dataio, systems, pipeline, plotting  # noqa: F401

__all__ = ["config", "dataio", "systems", "pipeline", "plotting"]
