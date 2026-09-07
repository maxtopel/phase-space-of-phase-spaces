#!/usr/bin/env python3
r"""
Universal PSoPS experiment: SINGLE vs MULTI barycenter, with Frechet-mean scaling.

Applies the V3 PSoPS pipeline (via the v3 bridge -> aoa_v3.barycenter,
landscape.estimate_d_eff, eps008.build_hellinger_dmap, transport.*) to the
non-eurostat macro corpus (pipeline/corpus.py). The reference is always a GW BARYCENTER
(Frechet mean) -- never a medoid (a medoid gives near-orthogonal/degenerate plans).

Stages (run one or 'all'):
  bary      build the universal barycenter on the corpus + per-series plans -> memmap
  scaling   geometric stability vs N_Frechet (the # of series building the Frechet mean):
            d_eff, curvature, DMAP spectrum -> plateau
  single    universal d_eff, robust across sampling measures + estimators
  timescale per-timescale barycenters: timescale sorting + temporal network (MDS)
  concept   per-concept barycenters: coupling network (V3 compute_multi_reference_plans)
  auto      descriptor-bucketed barycenters: unsupervised network (does it recover concepts?)
  umap      real PSoPS UMAP (replaces the m5-feature mislabel)
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os
import sys
import json
import time
import numpy as np

import pipeline.bridge as V3
import pipeline.corpus as C

OUT = C.OUT
N_S = 40
EPS = 0.008
SEED = 7
N_BARY = 1500           # series used to build the universal Frechet mean (headline)
DENSE_MAX = 12000       # cap for a dense Hellinger-DMAP
EVAL_N = 1500           # fixed evaluation set for the scaling sweep


# --------------------------------------------------------------------- utilities
def _arr():
    return np.load(C.COSTS, mmap_mode="r")


def _costs_of(arr, idx):
    out, keep = [], []
    for i in idx:
        M = np.asarray(arr[i], np.float64)
        if np.isfinite(M).all() and M.max() > C.MIN_COSTMAX:
            out.append(V3.normalize_cost(M)); keep.append(i)
    return out, np.array(keep)


def curvature(coords, k=12, cap=1500, seed=SEED):
    """Manifold curvature rho = median(geodesic / Euclidean) on the PSoPS coords
    (the paper's geodesic/Euclidean ratio). rho=1 flat, rho>1 curved."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    X = coords[rng.choice(len(coords), min(cap, len(coords)), replace=False)] if len(coords) > cap else coords
    G = kneighbors_graph(X, min(k, len(X) - 1), mode="distance")
    geo = shortest_path(G, method="D", directed=False)
    euc = cdist(X, X)
    m = np.isfinite(geo) & (euc > 0)
    return float(np.median(geo[m] / euc[m]))


def _deff_row(coords):
    d = V3.d_eff(coords)
    return {"two_nn": float(d.get("two_nn", float("nan"))),
            "levina_bickel": float(d.get("levina_bickel", float("nan"))),
            "primary": float(d.get("primary", float("nan")))}


# --------------------------------------------------------------------- bary+plans
def stage_bary():
    """Universal barycenter on the corpus + per-series entropic GW plans -> memmap."""
    from concurrent.futures import ProcessPoolExecutor
    arr = _arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    idx = bal["idx"]
    rng = np.random.default_rng(SEED)
    land, _ = _costs_of(arr, rng.permutation(idx)[:N_BARY + 300])
    land = land[:N_BARY]
    B = V3.gw_barycenter_nonentropic(land)
    np.save(os.path.join(OUT, "barycenter.npy"), B)
    print(f"universal barycenter built on {len(land)} series", flush=True)

    n = len(idx)
    mmap = os.path.join(OUT, "plans.dat")
    mm = np.memmap(mmap, dtype=np.float32, mode="w+", shape=(n, N_S * N_S))
    done = np.zeros(n, bool)
    CH = 200
    chunks = [np.arange(s, min(s + CH, n)) for s in range(0, n, CH)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_plan_chunk, idx[ch].tolist(), B.ravel()): ch for ch in chunks}
        completed = 0
        for fut in futs:
            ch = futs[fut]
            out, ok = fut.result()
            mm[ch] = out; done[ch] = ok
            completed += len(ch)
            if completed % 2000 < CH:
                r = completed / max(time.time() - t0, 1e-9)
                print(f"  plans {completed}/{n} | {r:.0f}/s ETA {(n-completed)/max(r,1e-9)/60:.0f}m", flush=True)
    mm.flush()
    valid = (np.abs(mm).sum(1) > 0)
    np.save(os.path.join(OUT, "plans_valid.npy"), valid)
    print(f"DONE plans: {valid.sum()}/{n} valid", flush=True)


def _plan_chunk(idx_chunk, Bflat):
    """Worker: V3 member_transport_plans (via bridge) for a chunk of cache indices."""
    import pipeline.bridge as V3
    arr = np.load(C.COSTS, mmap_mode="r")
    B = Bflat.reshape(N_S, N_S)
    costs, keep_local = [], []
    for j, i in enumerate(idx_chunk):
        M = np.asarray(arr[i], np.float64)
        if np.isfinite(M).all() and M.max() > C.MIN_COSTMAX:
            costs.append(V3.normalize_cost(M)); keep_local.append(j)
    out = np.zeros((len(idx_chunk), N_S * N_S), np.float32)
    ok = np.zeros(len(idx_chunk), bool)
    if costs:
        plans = V3.plans_to(costs, B)
        for j, T in zip(keep_local, plans):
            T = np.asarray(T, np.float64)
            if np.isfinite(T).all() and T.sum() > 0:
                out[j] = (T / T.sum()).ravel().astype(np.float32)
                ok[j] = True
    return out, ok


def _plans_parallel(ex, idx, B, chunk=200):
    """Parallel V3 plans for an index array against barycenter B; returns (P, valid)."""
    chunks = [idx[s:s + chunk] for s in range(0, len(idx), chunk)]
    Bflat = B.ravel()
    parts = list(ex.map(_plan_chunk, [c.tolist() for c in chunks], [Bflat] * len(chunks)))
    P = np.concatenate([p[0] for p in parts])
    ok = np.concatenate([p[1] for p in parts])
    return P[ok], ok


def _load_plans():
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    n = len(bal["idx"])
    mm = np.memmap(os.path.join(OUT, "plans.dat"), dtype=np.float32, mode="r", shape=(n, N_S * N_S))
    valid = np.load(os.path.join(OUT, "plans_valid.npy"))
    return mm, valid, bal


# --------------------------------------------------------------------- scaling
def stage_scaling():
    """Geometric stability vs the number of series used to BUILD the Frechet mean.
    For each N_bary we build a fresh universal barycenter, project a FIXED evaluation
    set onto it, and record d_eff / curvature / spectrum. Plateau => the universal
    barycenter is a well-defined object (not an artifact of which landmarks)."""
    from concurrent.futures import ProcessPoolExecutor
    arr = _arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    idx = bal["idx"]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(idx)
    eval_idx = perm[:EVAL_N]
    pool = perm[EVAL_N:]                       # disjoint pool for building barycenters
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for Nb in [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, min(8000, len(pool))]:
            if Nb > len(pool):
                break
            land, _ = _costs_of(arr, pool[:Nb])
            B = V3.gw_barycenter_nonentropic(land[:Nb])
            P, _ = _plans_parallel(ex, eval_idx, B)
            coords, evals, gap = V3.aoa_coords(P, n_components=20)
            row = {"N_frechet": int(Nb), **_deff_row(coords),
                   "curvature": curvature(coords), "gap": float(gap),
                   "spectrum": [float(x) for x in evals[:5]]}
            rows.append(row)
            print(f"  N_frechet={Nb:5d}  d_eff(2NN)={row['two_nn']:.2f}  LB={row['levina_bickel']:.2f}  "
                  f"curv={row['curvature']:.2f}  gap={gap:.2f}", flush=True)
    json.dump({"eval_n": EVAL_N, "scaling": rows, "eps": EPS},
              open(os.path.join(OUT, "scaling.json"), "w"), indent=2)
    print(f"PLATEAU d_eff(2NN) @ largest N_frechet = {rows[-1]['two_nn']:.2f}", flush=True)


# --------------------------------------------------------------------- single
def _concept_capped(valid, bal, frac=0.15, seed=SEED):
    con = bal["concept"]; vpos = np.where(valid)[0]
    rng = np.random.default_rng(seed); cap = max(int(len(vpos) * frac), 50)
    keep = []
    g = {}
    for p in vpos:
        g.setdefault(con[p], []).append(p)
    for c, m in g.items():
        m = np.array(m)
        keep.extend((rng.choice(m, cap, replace=False) if len(m) > cap else m).tolist())
    return np.sort(np.array(keep))


def _density_equalized(mm, valid, k=10, seed=SEED):
    from scipy.spatial import cKDTree
    vpos = np.where(valid)[0]
    X = np.asarray(mm[vpos]) / np.sqrt(2.0)
    dk, _ = cKDTree(X).query(X, k=k + 1, workers=-1)
    rk = dk[:, k]; w = rk / (np.median(rk) + 1e-12)
    rng = np.random.default_rng(seed)
    return np.sort(vpos[rng.random(len(X)) < np.clip(w, 0.05, 1.0)])


def stage_single():
    """Universal d_eff under 3 sampling measures x 2 estimators (referee-robust)."""
    mm, valid, bal = _load_plans()
    rng = np.random.default_rng(SEED)
    measures = {"natural": np.where(valid)[0],
                "concept_capped": _concept_capped(valid, bal),
                "density_equalized": _density_equalized(mm, valid)}
    res = {}
    for name, pool in measures.items():
        Nuse = min(len(pool), DENSE_MAX)
        sub = rng.choice(pool, Nuse, replace=False) if len(pool) > Nuse else pool
        coords, evals, gap = V3.aoa_coords(np.asarray(mm[sub]), n_components=20)
        res[name] = {"n": int(len(sub)), **_deff_row(coords),
                     "curvature": curvature(coords), "gap": float(gap)}
        print(f"  [{name:18s}] n={len(sub):6d}  d_eff(2NN)={res[name]['two_nn']:.2f} "
              f"LB={res[name]['levina_bickel']:.2f} curv={res[name]['curvature']:.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "single.json"), "w"), indent=2)
    tns = [v["two_nn"] for v in res.values()]
    print(f"UNIVERSAL d_eff(2NN) range across measures: [{min(tns):.2f}, {max(tns):.2f}]", flush=True)


# --------------------------------------------------------------------- multi-barycenter
N_BARY_GROUP = 600       # cap on series per group when building a group barycenter
COUPLE_EVAL = 1200       # series to embed for a coupling network


def _group_barys(arr, groups):
    """One GW barycenter (Frechet mean) per group. groups: {label: [cache idx]}."""
    barys, labels = {}, []
    for lab in sorted(groups):
        costs, _ = _costs_of(arr, np.array(groups[lab])[:N_BARY_GROUP])
        if len(costs) >= 8:
            barys[lab] = V3.gw_barycenter_nonentropic(costs)
            labels.append(lab)
    return [barys[l] for l in labels], labels


def _coupling_network(arr, eval_idx, eval_primary, barys, labels):
    """V3 multi-reference coupling -> K x K matrix -> MDS network coordinates."""
    from transport import compute_multi_reference_plans, coupling_weighted_dmap, extract_coupling_matrix
    from sklearn.manifold import MDS
    costs, keep = _costs_of(arr, eval_idx)
    prim = eval_primary[np.isin(eval_idx, keep)]
    K = len(barys)
    mp, sig = compute_multi_reference_plans(costs, barys, prim.astype(int),
                                            epsilon=EPS, probe_epsilon=0.1, entropy_gate=0.95)
    coords, evals, *_ = coupling_weighted_dmap(mp, K, N_S, sig, n_components=min(30, K * 2),
                                               n_landmarks=min(800, len(costs)))
    Cpl = extract_coupling_matrix(coords, mp, K, N_S, evals, prim.astype(int))
    Cpl = np.asarray(Cpl, float)
    D = 1.0 - (Cpl + Cpl.T) / 2.0
    np.fill_diagonal(D, 0.0); D = np.clip(D, 0, None)
    net = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              normalized_stress="auto").fit_transform(D)
    return Cpl.tolist(), net.tolist(), labels


def stage_timescale():
    """Per-timescale barycenters: (1) do attractors SORT by timescale on the PSoPS?
    (2) temporal network of the monthly/quarterly/annual barycenters."""
    arr = _arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    af_idx, af_ts = bal["allfreq_idx"], bal["allfreq_timescale"]
    rng = np.random.default_rng(SEED)
    groups = {t: af_idx[af_ts == t].tolist() for t in C.MACRO_FREQ if (af_ts == t).sum() >= 50}
    barys, labels = _group_barys(arr, groups)
    print(f"timescale barycenters: {labels}", flush=True)
    # sample a balanced eval set across timescales
    ev, prim = [], []
    for k, t in enumerate(labels):
        m = rng.choice(groups[t], min(COUPLE_EVAL // len(labels), len(groups[t])), replace=False)
        ev.extend(m.tolist()); prim.extend([k] * len(m))
    ev = np.array(ev); prim = np.array(prim)
    Cpl, net, labs = _coupling_network(arr, ev, prim, barys, labels)
    # timescale sorting: silhouette of timescale labels on the universal PSoPS
    from concurrent.futures import ProcessPoolExecutor
    Buni = np.load(os.path.join(OUT, "barycenter.npy"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        P, ok = _plans_parallel(ex, ev, Buni)
    coords, _, _ = V3.aoa_coords(P, n_components=20)
    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(coords, prim[ok])) if len(set(prim[ok])) > 1 else None
    print(f"  timescale sorting silhouette on universal PSoPS: {sil}", flush=True)
    np.savez(os.path.join(OUT, "timescale_network.npz"), coupling=np.array(Cpl),
             net=np.array(net), labels=np.array(labs), aoa_coords=coords,
             aoa_timescale=prim[ok], silhouette=sil)
    json.dump({"labels": labs, "silhouette": sil, "coupling": Cpl},
              open(os.path.join(OUT, "timescale_network.json"), "w"), indent=2)
    print(f"wrote timescale_network (temporal subpanel)", flush=True)


def stage_concept():
    """Per-concept barycenters -> V3 coupling network (the conceptual relational map)."""
    arr = _arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    idx, con = bal["idx"], bal["concept"]
    rng = np.random.default_rng(SEED)
    groups = {c: idx[con == c].tolist() for c in sorted(set(con.tolist()))
              if (con == c).sum() >= 60}
    barys, labels = _group_barys(arr, groups)
    print(f"concept barycenters ({len(labels)}): {labels}", flush=True)
    lab2k = {l: k for k, l in enumerate(labels)}
    ev, prim = [], []
    for c in labels:
        m = rng.choice(groups[c], min(COUPLE_EVAL // len(labels), len(groups[c])), replace=False)
        ev.extend(m.tolist()); prim.extend([lab2k[c]] * len(m))
    ev = np.array(ev); prim = np.array(prim)
    Cpl, net, labs = _coupling_network(arr, ev, prim, barys, labels)
    np.savez(os.path.join(OUT, "concept_network.npz"), coupling=np.array(Cpl),
             net=np.array(net), labels=np.array(labs))
    json.dump({"labels": labs, "coupling": Cpl},
              open(os.path.join(OUT, "concept_network.json"), "w"), indent=2)
    print("wrote concept_network", flush=True)


def _attractor_descriptor(C40):
    """V1-style attractor-shape descriptor: top classical-MDS eigenvalues of the
    cost matrix (a GW-free similarity descriptor for unsupervised bucketing)."""
    n = C40.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    with np.errstate(all="ignore"):
        Bd = -0.5 * J @ (C40 ** 2) @ J
    ev = np.sort(np.linalg.eigvalsh(Bd))[::-1][:16]
    return np.sign(ev) * np.log1p(np.abs(ev))


def stage_auto(K=10):
    """UNSUPERVISED multi-barycenter: descriptor-cluster attractors into K similarity
    buckets, build a barycenter per bucket, relate buckets via the coupling network.
    Test whether the auto-buckets recover concept-like structure (ARI vs indicator)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    arr = _arr()
    bal = np.load(os.path.join(OUT, "corpus.npz"))
    idx, con = bal["idx"], bal["concept"]
    rng = np.random.default_rng(SEED)
    samp = rng.choice(idx, min(COUPLE_EVAL, len(idx)), replace=False)
    costs, keep = _costs_of(arr, samp)
    D = np.array([_attractor_descriptor(c * 1.0) for c in costs])  # costs already normalized
    Z = (D - D.mean(0)) / (D.std(0) + 1e-9)
    buckets = KMeans(K, n_init=10, random_state=SEED).fit_predict(Z)
    ari = float(adjusted_rand_score(con[np.isin(idx, keep)], buckets))
    print(f"  auto-buckets vs concept ARI = {ari:.3f}", flush=True)
    groups = {int(b): keep[buckets == b].tolist() for b in range(K) if (buckets == b).sum() >= 8}
    barys, labels = _group_barys(arr, groups)
    prim = np.array([labels.index(b) if b in labels else 0 for b in buckets])
    Cpl, net, labs = _coupling_network(arr, keep, prim, barys, labels)
    np.savez(os.path.join(OUT, "auto_network.npz"), coupling=np.array(Cpl),
             net=np.array(net), labels=np.array(labs), buckets=buckets, ari=ari)
    json.dump({"K": K, "ari_vs_concept": ari, "labels": [str(l) for l in labs]},
              open(os.path.join(OUT, "auto_network.json"), "w"), indent=2)
    print("wrote auto_network", flush=True)


def stage_umap():
    """Real universal-PSoPS UMAP (replaces the m5-feature mislabel)."""
    import umap
    mm, valid, bal = _load_plans()
    vidx = _density_equalized(mm, valid)
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(vidx, min(40000, len(vidx)), replace=False)) if len(vidx) > 40000 else vidx
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=SEED).fit_transform(np.asarray(mm[sub]))
    np.savez(os.path.join(OUT, "aoa_umap.npz"), embedding=emb,
             concept=bal["concept"][sub], source=bal["source"][sub], timescale=bal["timescale"][sub])
    print(f"wrote aoa_umap.npz ({len(sub)} pts)", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("bary", "all"):
        stage_bary()
    if stage in ("scaling", "all"):
        stage_scaling()
    if stage in ("single", "all"):
        stage_single()
    if stage in ("timescale", "all", "multi"):
        stage_timescale()
    if stage in ("concept", "all", "multi"):
        stage_concept()
    if stage in ("auto", "all", "multi"):
        stage_auto()
    if stage in ("umap", "all"):
        stage_umap()
    print(f"[{stage}] done in {(time.time()-t0)/60:.1f} min", flush=True)
