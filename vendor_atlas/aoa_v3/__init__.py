"""
aoa_v3 — Attractor-of-Attractors pipeline for "The Geometry of the Macroeconomy".

Modules map 1:1 to paper sections:
  embed        § 2.1  Takens delay embedding
  dmap         § 2.2  Per-series diffusion maps
  gw           § 3.1–3.2  Entropic Gromov-Wasserstein transport
  barycenter   § 3.3  GW barycenter
  aoa          § 3.3 (end)  Hellinger-DMAP on transport plans
  sensitivity  § 4    Sensitivity tensor S_{i->j}^{k,l}
  predict      § Prediction  Levels 1 and 2
  data         —       Law-corpus loaders (Phillips, Solow, Okun)
  viz          —       Attractor rendering

Paper defaults (from v1 empirical optimization): epsilon = 0.008, n_s = 40.
"""

__version__ = "0.1.0"
