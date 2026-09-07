"""Complete Layer 1/L2/L3 results at eps=0.008 on balanced corpus.

Computes ALL submanifold d_eff values, within-country proximity p-values,
and coupling matrix rho values using the frozen 18K barycenters and the
57K stage4 distance matrices, recomputed at eps=0.008.

Layer 1 (d_eff): GDP, Okun, Solow, Phillips, Real, Rates, Inflation,
                 Credit, Full
Layer 2 (proximity): Phillips, Okun, Fisher, PPP, Monetary
Layer 3 (coupling rho): Solow+Monetary, Real+Nominal, Solow+Real

Run: python3 -u scripts/eps008_complete_layer1.py
"""
import sys
import os
import pickle
import time
import json
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPANDED_DIR = os.path.join(ROOT, 'checkpoints', 'expanded')
OUT_DIR = os.path.join(ROOT, 'checkpoints', 'convergence_battery')
os.makedirs(OUT_DIR, exist_ok=True)

EPSILON = 0.008
N_SAMPLE = 1500  # Smaller to keep balanced (238 unique pairs + capped backfill)
N_COMPONENTS = 20
SEED = 42

# ── Submanifold definitions ───────────────────────────────────────

INDICATORS = {
    'GDP': {'gdp'},
    'Okun': {'gdp', 'unemployment'},
    'Solow': {'gdp', 'capital', 'labor', 'tfp'},
    'Phillips': {'inflation', 'unemployment'},
    'Real': {'gdp', 'unemployment', 'labor'},
    'Rates': {'interest_rate'},
    'Inflation': {'inflation'},
    'Credit': {'credit', 'gdp'},
    'Full': None,
}

# Proximity tests: (law_name, tag_a, tag_b)
PROXIMITY_TESTS = [
    ('Phillips', 'inflation', 'unemployment'),
    ('Okun', 'gdp', 'unemployment'),
    ('Fisher', 'interest_rate', 'inflation'),
    ('PPP', 'exchange_rate', 'inflation'),
    ('Monetary', 'interest_rate', 'gdp'),
]

# Coupling pairs: (name, domain_A_indicators, domain_B_indicators)
COUPLING_PAIRS = [
    ('Solow+Monetary', {'gdp', 'capital', 'labor', 'tfp'},
                       {'interest_rate', 'credit', 'gdp', 'inflation'}),
    ('Real+Nominal', {'gdp', 'unemployment', 'labor'},
                     {'inflation', 'interest_rate'}),
    ('Solow+Real', {'gdp', 'capital', 'labor', 'tfp'},
                   {'gdp', 'unemployment', 'labor'}),
]


# ── Helpers ───────────────────────────────────────────────────────

def load_stage4():
    """Load stage4 expanded state (57K plans + dist matrices + metadata)."""
    path = os.path.join(EXPANDED_DIR, 'stage4.pkl')
    print(f"Loading stage4 from {path} ...", flush=True)
    with open(path, 'rb') as f:
        s4 = pickle.load(f)
    print(f"  {len(s4['plans'])} series, n_sup={s4['n_sup']}", flush=True)
    return s4


def load_frozen_barycenters():
    """Load frozen 18K barycenters."""
    path = os.path.join(ROOT, 'frozen_barycenters_18k.pkl')
    print(f"Loading frozen barycenters from {path} ...", flush=True)
    with open(path, 'rb') as f:
        fb = pickle.load(f)
    print(f"  n_sup={fb['n_sup']}, epsilon={fb['epsilon']}", flush=True)
    print(f"  Timescales: {sorted(fb['timescale_barycenters'].keys())}", flush=True)
    return fb


def build_balanced_subsample(metadata, dm_eff_ts, n_target, rng):
    """Build a balanced subsample: 1 per (country, indicator) pair.

    If fewer than n_target unique (country, indicator) pairs exist,
    use stratified capped sampling to reach n_target.
    """
    from scripts.stage_runner import COUNTRY_CODES, SOURCE_COUNTRY

    by_ci = defaultdict(list)
    for j, m in enumerate(metadata):
        ind = m.get('indicator', 'other')
        if ind == 'other':
            continue
        source = m.get('source', '')
        country = SOURCE_COUNTRY.get(source, m.get('country'))
        if not country:
            kl = m.get('key', '').lower()
            for code, name in COUNTRY_CODES.items():
                if f'_{code}' in kl or f'_{code}_' in kl or kl.endswith(f'_{code}'):
                    country = name
                    break
        if country:
            by_ci[(country, ind)].append(j)

    ci_keys = sorted(by_ci.keys())
    print(f"  Unique (country, indicator) pairs: {len(ci_keys)}", flush=True)

    if len(ci_keys) >= n_target:
        # Pure balanced: 1 per pair, random selection
        rng.shuffle(ci_keys)
        selected = []
        for ci in ci_keys[:n_target]:
            selected.append(rng.choice(by_ci[ci]))
        return sorted(selected)

    # Stratified fallback: cycle through pairs until we hit n_target
    selected = set()
    # First pass: 1 per pair
    for ci in ci_keys:
        selected.add(rng.choice(by_ci[ci]))

    # Fill remaining: cap each INDICATOR at 200 total to prevent interest_rate domination
    remaining = n_target - len(selected)
    if remaining > 0:
        by_ind = defaultdict(list)
        for ci in ci_keys:
            ind = ci[1]
            available = [j for j in by_ci[ci] if j not in selected]
            by_ind[ind].extend(available)
        fill_pool = []
        ind_cap = max(50, remaining // len(by_ind))  # equal share per indicator
        for ind, indices in by_ind.items():
            rng.shuffle(indices)
            fill_pool.extend(indices[:ind_cap])
        rng.shuffle(fill_pool)
        for j in fill_pool[:remaining]:
            selected.add(j)

    return sorted(selected)


def compute_plans_at_eps(dist_matrices, dm_eff_ts, barycenters, n_sup,
                         epsilon, indices):
    """Recompute transport plans at the given epsilon for selected indices."""
    from transport import compute_transport_plan

    plans = []
    valid_indices = []
    n_total = len(indices)

    for count, j in enumerate(indices):
        eff_ts = dm_eff_ts[j]
        if eff_ts not in barycenters:
            continue
        D = dist_matrices[j]
        C_bary = barycenters[eff_ts]
        try:
            T = compute_transport_plan(D, C_bary, epsilon=epsilon)
            plans.append(T)
            valid_indices.append(j)
        except Exception:
            # Uniform fallback
            T_uniform = np.ones(n_sup**2, dtype=np.float32) / n_sup**2
            plans.append(T_uniform)
            valid_indices.append(j)

        if (count + 1) % 500 == 0:
            print(f"    {count+1}/{n_total} plans ...", flush=True)

    return np.array(plans), valid_indices


def build_hellinger_dmap(plans, n_components=20):
    """Build Hellinger DMAP from transport plans.

    Returns coords, eigenvalues, spectral gap, mean plan signal strength.
    """
    n_sup = int(np.sqrt(plans.shape[1]))
    sqrt_p = np.sqrt(np.maximum(plans, 1e-15))
    H_dist = cdist(sqrt_p, sqrt_p)

    bw = np.median(H_dist[H_dist > 0])
    if bw < 1e-10:
        return None, None, 0.0, 0.0

    K_mat = np.exp(-H_dist**2 / (2 * bw**2))
    np.fill_diagonal(K_mat, 0)
    d_a = K_mat.sum(axis=1)
    d_inv = 1.0 / np.maximum(d_a, 1e-10)
    K_norm = K_mat * np.outer(d_inv, d_inv)
    rs = K_norm.sum(axis=1)
    P = K_norm / np.maximum(rs[:, None], 1e-10)

    evals, evecs = np.linalg.eigh(P)
    idx = np.argsort(evals)[::-1]
    n_keep = min(n_components, len(evals) - 1)
    evals_s = evals[idx][1:n_keep + 1]
    evecs_s = evecs[:, idx][:, 1:n_keep + 1]
    coords = evecs_s * evals_s

    gap = float(evals_s[0] / evals_s[1]) if len(evals_s) > 1 and evals_s[1] > 0 else 0.0

    # Mean plan signal strength
    H_max = 2 * np.log(n_sup)
    mean_S = float(np.mean([
        1.0 - (-np.sum(np.maximum(T, 1e-15) * np.log(np.maximum(T, 1e-15)))) / H_max
        for T in plans
    ]))

    return coords, evals_s, gap, mean_S


def compute_d_eff_all(coords, metadata, valid_indices, indicators):
    """Compute d_eff for all submanifold definitions."""
    from landscape import estimate_d_eff

    results = {}
    for name, inds in indicators.items():
        if inds is None:
            mask = list(range(len(valid_indices)))
        else:
            mask = [i for i, j in enumerate(valid_indices)
                    if j < len(metadata) and metadata[j].get('indicator', 'other') in inds]
        if len(mask) < 20:
            results[name] = {'d_eff': None, 'n': len(mask), 'methods': {}}
            continue
        d = estimate_d_eff(coords[mask])
        results[name] = {
            'd_eff': d.get('two_nn'),
            'n': len(mask),
            'methods': {k: float(v) for k, v in d.items() if v is not None and np.isfinite(v)},
        }
    return results


def run_proximity_tests(coords, metadata, valid_indices, tests, n_perm=200):
    """Run within-country proximity tests for macro law validation."""
    from validate import within_country_proximity_test
    from scripts.stage_runner import COUNTRY_CODES, SOURCE_COUNTRY, TAG_ALIASES

    # Build tagged metadata list aligned with coords
    meta_list = []
    for i, j in enumerate(valid_indices):
        if j >= len(metadata):
            meta_list.append({'tags': [], 'country': None})
            continue
        m = metadata[j]
        ind = m.get('indicator', 'other')
        tags = list(TAG_ALIASES.get(ind, [ind]))
        source = m.get('source', '')
        country = SOURCE_COUNTRY.get(source, m.get('country'))
        if not country:
            kl = m.get('key', '').lower()
            for code, name in COUNTRY_CODES.items():
                if f'_{code}' in kl or f'_{code}_' in kl or kl.endswith(f'_{code}'):
                    country = name
                    break
        meta_list.append({**m, 'tags': tags, 'country': country})

    results = {}
    for law, ta, tb in tests:
        try:
            stat, pval = within_country_proximity_test(
                coords, meta_list, ta, tb, n_permutations=n_perm)
            results[law] = {
                'stat': float(stat) if np.isfinite(stat) else None,
                'p': float(pval) if np.isfinite(pval) else None,
            }
            status = 'PASS' if pval < 0.05 else 'FAIL' if np.isfinite(pval) else 'N/A'
            print(f"    {law:15s}: stat={stat:.3f}, p={pval:.4f} -> {status}", flush=True)
        except Exception as e:
            results[law] = {'stat': None, 'p': None, 'error': str(e)}
            print(f"    {law:15s}: ERROR ({e})", flush=True)
    return results


def compute_coupling_rho(d_eff_results, coupling_pairs, coords, metadata,
                         valid_indices):
    """Compute coupling rho = d_eff(A union B) / (d_eff(A) + d_eff(B)).

    Values near 0.5 = independent submanifolds (additive).
    Values < 0.5 = shared dimensions (coupling).
    Values > 0.5 = interactions increase dimension.
    """
    from landscape import estimate_d_eff

    results = {}
    for name, inds_a, inds_b in coupling_pairs:
        # d_eff(A)
        mask_a = [i for i, j in enumerate(valid_indices)
                  if j < len(metadata) and metadata[j].get('indicator', 'other') in inds_a]
        # d_eff(B)
        mask_b = [i for i, j in enumerate(valid_indices)
                  if j < len(metadata) and metadata[j].get('indicator', 'other') in inds_b]
        # d_eff(A union B)
        mask_union = sorted(set(mask_a) | set(mask_b))

        d_a, d_b, d_union = None, None, None
        if len(mask_a) >= 20:
            d_a = estimate_d_eff(coords[mask_a]).get('two_nn')
        if len(mask_b) >= 20:
            d_b = estimate_d_eff(coords[mask_b]).get('two_nn')
        if len(mask_union) >= 20:
            d_union = estimate_d_eff(coords[mask_union]).get('two_nn')

        rho = None
        if d_a is not None and d_b is not None and d_union is not None:
            denom = d_a + d_b
            if denom > 0:
                rho = d_union / denom

        results[name] = {
            'd_eff_A': float(d_a) if d_a is not None else None,
            'd_eff_B': float(d_b) if d_b is not None else None,
            'd_eff_union': float(d_union) if d_union is not None else None,
            'rho': float(rho) if rho is not None else None,
            'n_A': len(mask_a),
            'n_B': len(mask_b),
            'n_union': len(mask_union),
            'indicators_A': sorted(inds_a),
            'indicators_B': sorted(inds_b),
        }

        rho_str = f"{rho:.3f}" if rho is not None else "N/A"
        interpretation = ""
        if rho is not None:
            if rho < 0.45:
                interpretation = "(shared dims -> coupled)"
            elif rho < 0.55:
                interpretation = "(independent)"
            else:
                interpretation = "(interaction effects)"

        d_a_s = f"{d_a:.1f}" if d_a else "N/A"
        d_b_s = f"{d_b:.1f}" if d_b else "N/A"
        d_u_s = f"{d_union:.1f}" if d_union else "N/A"
        print(f"    {name:20s}: rho={rho_str} "
              f"[d_A={d_a_s}, d_B={d_b_s}, d_AB={d_u_s}] "
              f"(n={len(mask_a)}/{len(mask_b)}/{len(mask_union)}) "
              f"{interpretation}", flush=True)

    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    rng = np.random.RandomState(SEED)

    print("=" * 70, flush=True)
    print("EPS=0.008 COMPLETE LAYER 1/L2/L3 RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  epsilon = {EPSILON}", flush=True)
    print(f"  N_SAMPLE = {N_SAMPLE}", flush=True)
    print(f"  seed = {SEED}", flush=True)

    # ── Load data ─────────────────────────────────────────────────
    s4 = load_stage4()
    fb = load_frozen_barycenters()

    metadata = s4['dm_metadata']
    dist_matrices = s4['dist_matrices']
    dm_eff_ts = s4['dm_eff_ts']
    n_sup = s4['n_sup']
    barycenters = fb['timescale_barycenters']

    assert n_sup == fb['n_sup'], (
        f"n_sup mismatch: stage4={n_sup}, frozen={fb['n_sup']}"
    )

    n_total = len(metadata)
    print(f"\n  Total series: {n_total}", flush=True)

    # Indicator distribution
    ind_counts = Counter(m.get('indicator', 'other') for m in metadata)
    print("  Indicator distribution:", flush=True)
    for ind, cnt in ind_counts.most_common():
        print(f"    {ind:20s}: {cnt}", flush=True)

    # ── Build balanced subsample ──────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"BUILDING BALANCED SUBSAMPLE (N={N_SAMPLE})", flush=True)

    subsample_idx = build_balanced_subsample(metadata, dm_eff_ts, N_SAMPLE, rng)
    print(f"  Selected {len(subsample_idx)} series", flush=True)

    # Verify balance
    sub_inds = Counter(metadata[j].get('indicator', 'other') for j in subsample_idx)
    print("  Subsample indicator distribution:", flush=True)
    for ind, cnt in sub_inds.most_common():
        print(f"    {ind:20s}: {cnt}", flush=True)

    sub_ts = Counter(dm_eff_ts[j] for j in subsample_idx)
    print("  Subsample timescale distribution:", flush=True)
    for ts, cnt in sub_ts.most_common():
        print(f"    {ts:20s}: {cnt}", flush=True)

    # ── Recompute transport plans at eps=0.008 ────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"RECOMPUTING TRANSPORT PLANS AT eps={EPSILON}", flush=True)
    t0 = time.time()

    plans, valid_indices = compute_plans_at_eps(
        dist_matrices, dm_eff_ts, barycenters, n_sup, EPSILON, subsample_idx)

    print(f"  {len(valid_indices)} valid plans ({time.time()-t0:.0f}s)", flush=True)

    # ── Build Hellinger DMAP ──────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("BUILDING HELLINGER DMAP", flush=True)
    t0 = time.time()

    coords, evals, gap, mean_S = build_hellinger_dmap(plans, N_COMPONENTS)

    if coords is None:
        print("  ERROR: DMAP failed (zero bandwidth). Aborting.", flush=True)
        return

    print(f"  coords shape: {coords.shape}", flush=True)
    print(f"  spectral gap: {gap:.3f}", flush=True)
    print(f"  mean signal S: {mean_S:.4f}", flush=True)
    print(f"  top eigenvalues: {evals[:5]}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── Layer 1: d_eff for ALL submanifolds ───────────────────────
    print(f"\n{'='*70}", flush=True)
    print("LAYER 1: SUBMANIFOLD d_eff", flush=True)
    t0 = time.time()

    d_eff_results = compute_d_eff_all(coords, metadata, valid_indices, INDICATORS)

    print(f"\n  {'Submanifold':>15s} {'d_eff':>8s} {'n':>6s} {'D2':>8s}", flush=True)
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*8}", flush=True)
    for name in ['GDP', 'Okun', 'Solow', 'Phillips', 'Real', 'Rates',
                 'Inflation', 'Credit', 'Full']:
        r = d_eff_results[name]
        d = r['d_eff']
        d_str = f"{d:.2f}" if d is not None else "N/A"
        d2 = r['methods'].get('D2')
        d2_str = f"{d2:.2f}" if d2 is not None else "N/A"
        print(f"  {name:>15s} {d_str:>8s} {r['n']:>6d} {d2_str:>8s}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── Layer 2: Within-country proximity tests ───────────────────
    print(f"\n{'='*70}", flush=True)
    print("LAYER 2: WITHIN-COUNTRY PROXIMITY TESTS", flush=True)
    t0 = time.time()

    proximity_results = run_proximity_tests(
        coords, metadata, valid_indices, PROXIMITY_TESTS, n_perm=200)

    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── Layer 3: Coupling matrix rho ──────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("LAYER 3: COUPLING MATRIX RHO", flush=True)
    t0 = time.time()

    coupling_results = compute_coupling_rho(
        d_eff_results, COUPLING_PAIRS, coords, metadata, valid_indices)

    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n  Configuration:", flush=True)
    print(f"    epsilon:    {EPSILON}", flush=True)
    print(f"    N_SAMPLE:   {N_SAMPLE}", flush=True)
    print(f"    n_sup:      {n_sup}", flush=True)
    print(f"    DMAP gap:   {gap:.3f}", flush=True)
    print(f"    mean S:     {mean_S:.4f}", flush=True)

    print(f"\n  Layer 1 (d_eff):", flush=True)
    for name in INDICATORS:
        r = d_eff_results[name]
        d = r['d_eff']
        status = ""
        if d is not None:
            if name in ('GDP', 'Rates', 'Inflation'):
                status = "OK" if 1 <= d <= 4 else "HIGH" if d > 4 else "LOW"
            elif name in ('Okun', 'Phillips', 'Credit'):
                status = "OK" if 1.5 <= d <= 5 else "HIGH" if d > 5 else "LOW"
            elif name == 'Solow':
                status = "OK" if 2 <= d <= 5 else "HIGH" if d > 5 else "LOW"
            elif name == 'Real':
                status = "OK" if 2 <= d <= 6 else "HIGH" if d > 6 else "LOW"
            elif name == 'Full':
                status = "OK" if 3 <= d <= 12 else "HIGH" if d > 12 else "LOW"
        d_str = f"{d:.2f}" if d is not None else "N/A"
        print(f"    {name:>12s}: d_eff = {d_str:>6s} (n={r['n']:>4d}) {status}", flush=True)

    print(f"\n  Layer 2 (proximity p-values):", flush=True)
    for law in ['Phillips', 'Okun', 'Fisher', 'PPP', 'Monetary']:
        r = proximity_results.get(law, {})
        p = r.get('p')
        s = r.get('stat')
        if p is not None and np.isfinite(p):
            status = 'PASS' if p < 0.05 else 'MARGINAL' if p < 0.10 else 'FAIL'
            print(f"    {law:>12s}: p={p:.4f}, ratio={s:.3f} -> {status}", flush=True)
        else:
            print(f"    {law:>12s}: N/A", flush=True)

    print(f"\n  Layer 3 (coupling rho):", flush=True)
    for name, _, _ in COUPLING_PAIRS:
        r = coupling_results.get(name, {})
        rho = r.get('rho')
        if rho is not None:
            interp = "coupled" if rho < 0.45 else "independent" if rho < 0.55 else "interaction"
            print(f"    {name:>20s}: rho={rho:.3f} ({interp})", flush=True)
        else:
            print(f"    {name:>20s}: N/A", flush=True)

    # ── Save results ──────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f'eps008_complete_results_{timestamp}.json')

    all_results = {
        'config': {
            'epsilon': EPSILON,
            'n_sample': N_SAMPLE,
            'n_sup': n_sup,
            'seed': SEED,
            'n_components': N_COMPONENTS,
            'n_valid_plans': len(valid_indices),
            'spectral_gap': gap,
            'mean_S': mean_S,
            'top_eigenvalues': evals[:10].tolist() if evals is not None else None,
        },
        'layer1_d_eff': {
            name: {
                'd_eff': float(r['d_eff']) if r['d_eff'] is not None else None,
                'n': r['n'],
                'methods': r['methods'],
            }
            for name, r in d_eff_results.items()
        },
        'layer2_proximity': proximity_results,
        'layer3_coupling': coupling_results,
        'subsample_stats': {
            'indicator_counts': dict(sub_inds.most_common()),
            'timescale_counts': dict(sub_ts.most_common()),
            'total_selected': len(subsample_idx),
        },
    }

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETE ({elapsed/60:.1f} min)", flush=True)
    print(f"Results saved to: {out_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
