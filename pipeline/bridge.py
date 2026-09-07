#!/usr/bin/env python3
r"""
Thin bridge that imports the *actual V3 AoA pipeline* functions so the universal-AoA
scripts apply the published methodology rather than reimplementing it.

Everything here is an import from the atlas AoA v3 codebase:
  - aoa_v3.barycenter.member_transport_plans   (entropic GW plan, series -> barycenter)
  - landscape.estimate_d_eff                    (TWO-NN / Levina-Bickel / spectral d_eff)
  - eps008_complete_layer1.build_hellinger_dmap (Hellinger diffusion map on plans)
  - transport.{compute_multi_reference_plans, coupling_weighted_dmap,
               extract_coupling_matrix}         (multi-barycenter coupling)
The only thing we do *not* import is the slow entropic barycenter solver: for the
Frechet-mean scaling sweep we use POT's non-entropic `ot.gromov.gromov_barycenters`,
which is exactly what the V3 repo's `finite_size_scaling.py` uses for N-sweeps.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import sys
import importlib.util
import numpy as np
import ot

# vendored atlas modules ship with the repo (vendor_atlas/); the private tree
# is not needed at runtime
VENDOR = os.path.join(_ROOT, "vendor_atlas")
sys.path.insert(0, VENDOR)

from aoa_v3.barycenter import member_transport_plans, BarycenterConfig          # noqa: E402
from aoa_v3.gw import normalize_cost                                            # noqa: E402
from landscape import estimate_d_eff                                            # noqa: E402
from transport import (compute_multi_reference_plans, coupling_weighted_dmap,   # noqa: E402
                       extract_coupling_matrix)

# build_hellinger_dmap lives in a script module; load it by path
_spec = importlib.util.spec_from_file_location(
    "eps008_layer1", os.path.join(VENDOR, "eps008_complete_layer1.py"))
_eps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eps)
build_hellinger_dmap = _eps.build_hellinger_dmap        # (plans, n_components=20) -> coords, evals, gap, meanS

N_S = 40
EPS = 0.008


def gw_barycenter_nonentropic(costs, n_sup=N_S, max_iter=100, seed=0):
    """Non-entropic GW Frechet mean of a list of (already unit-max-normalized) cost
    matrices. POT `gromov_barycenters`, the V3 finite_size_scaling.py solver. Fast."""
    p = np.full(n_sup, 1.0 / n_sup)
    B = ot.gromov.gromov_barycenters(
        N=n_sup, Cs=costs, ps=[p] * len(costs), p=p,
        lambdas=np.ones(len(costs)) / len(costs), loss_fun="square_loss",
        max_iter=max_iter, tol=1e-5)
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 0.0)
    return B


def plans_to(costs, B, epsilon=EPS):
    """Entropic GW transport plans (series -> barycenter B) via the V3
    member_transport_plans. Returns a list of (n_s x n_s) plans."""
    cfg = BarycenterConfig(n_supports=N_S, epsilon=epsilon, n_workers=1)
    plans, _ = member_transport_plans(costs, B, cfg=cfg)
    return [np.asarray(T, np.float64) for T in plans]


def aoa_coords(plans, n_components=20):
    """V3 Hellinger-DMAP on a stack of transport plans -> AoA coordinates + spectrum."""
    P = np.array([T.ravel() for T in plans]) if isinstance(plans, list) else plans
    coords, evals, gap, meanS = build_hellinger_dmap(P, n_components=n_components)
    return coords, evals, gap


def d_eff(coords):
    """V3 landscape.estimate_d_eff; returns the dict (two_nn, levina_bickel, ...)."""
    return estimate_d_eff(coords)
