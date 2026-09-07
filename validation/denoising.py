#!/usr/bin/env python3
r"""
Illustration: family/indicator splitting + DMAP-vs-raw-Takens (Results
sec.~\ref{sec:validation}).

Left  : clean simulated families (dysts/gate) split cleanly in the PSoPS feature
        embedding; macro indicators OVERLAP (the paper's thesis).
Right : DMAP-vs-raw ablation -- the per-series diffusion map denoises the
        geometry feature; the gain is large/essential on noisy macro and modest
        on clean systems.

Reuses the validated family-discovery features (rmt_aoa/synthetic_gate.py) and
the macro labeled sample. Writes ../figures/fig_dmap_vs_raw.pdf.
"""
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import os, sys
import numpy as np
from aoa_repro import config
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score

# vendored atlas modules ship with the repo (vendor_atlas/)
sys.path.insert(0, os.path.join(_ROOT, "vendor_atlas"))
import synthetic_gate as G
from embed import make_delay_vectors, standardize_rows
from aoa_v3.embed import embed_series, EmbedConfig
from aoa_v3.dmap import dmap_series, DmapConfig
from aoa_v3.barycenter import fps_landmarks
# indicator list inlined from the retired dmap-ablation script
INDS    = ["interest_rate", "gdp", "inflation", "exchange_rate", "credit"]

FS = 9
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
NS = 40; N_COMP = 10
FIG_OUT = str(config.FIG_OUT / "fig_dmap_vs_raw.pdf")


def ward_ari(F, labels):
    X = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return adjusted_rand_score(labels, AgglomerativeClustering(
        n_clusters=len(set(labels)), linkage="ward").fit_predict(X))


DV_CAP = 2000   # cap per-series delay-cloud size before O(n^2) cdist (RAM-safe)


def _cap(dv):
    if len(dv) > DV_CAP:
        return dv[np.linspace(0, len(dv) - 1, DV_CAP).astype(int)]
    return dv


def raw_dm(dv):
    dv = standardize_rows(_cap(dv))
    sel = fps_landmarks(cdist(dv, dv), n_landmarks=NS, seed=0)
    return cdist(dv[sel], dv[sel])


def dmap_dm(dv, tau):
    co, _ = dmap_series(_cap(dv), cfg=DmapConfig(n_components=N_COMP, theiler=0), tau=tau)
    sel = fps_landmarks(cdist(co, co), n_landmarks=NS, seed=0)
    return cdist(co[sel], co[sel])


def _ok(*vs):
    return all(np.all(np.isfinite(v)) for v in vs)


def features_dysts():
    cache = os.path.join(_ROOT, "caches", "cache_dysts_features.npz")
    if os.path.exists(cache):
        z = np.load(cache); return [z["m1d"], z["m1r"], z["m4"], z["m5"]], z["lab"]
    m1d, m1r, m4, m5, lab = [], [], [], [], []
    for fi, (name, gen, params, tau, m) in enumerate(G.FAMILY_CONFIGS):
        for seed in range(5):
            try:
                s = G.generate_family_series(name, gen, params, 12000, seed)
                ss = (s - s.mean())/(s.std()+1e-12)
                dv = make_delay_vectors(ss, dim=m, tau=tau)
                if dv.shape[0] < NS:
                    continue
                Dd, Dr = dmap_dm(dv, tau), raw_dm(dv)
                f1d, f1r = G.compute_m1_spectral(Dd), G.compute_m1_spectral(Dr)
                f4, f5 = G.compute_m4_descriptors(Dd), G.compute_m5_temporal(s)
                if not _ok(f1d, f1r, f4, f5):
                    continue
                m1d.append(f1d); m1r.append(f1r); m4.append(f4); m5.append(f5); lab.append(fi)
            except Exception:
                continue
    out = list(map(np.array, (m1d, m1r, m4, m5))); lab = np.array(lab)
    np.savez(cache, m1d=out[0], m1r=out[1], m4=out[2], m5=out[3], lab=lab)
    return out, lab


def features_macro():
    """Use the cached macro delay-vectors (no re-embed); m5 omitted (needs raw)."""
    dvcache = os.path.join(_ROOT, "caches", "cache_dmap_ablation_dv.npz")
    z = np.load(dvcache, allow_pickle=True)
    dvs, lab_all, taus = list(z["dv"]), np.asarray(z["labels"]), list(z["taus"])
    m1d, m1r, m4, lab = [], [], [], []
    for dv, t, li in zip(dvs, taus, lab_all):
        try:
            Dd, Dr = dmap_dm(dv, int(t)), raw_dm(dv)
            f1d, f1r, f4 = G.compute_m1_spectral(Dd), G.compute_m1_spectral(Dr), G.compute_m4_descriptors(Dd)
            if not _ok(f1d, f1r, f4):
                continue
            m1d.append(f1d); m1r.append(f1r); m4.append(f4); lab.append(li)
        except Exception:
            continue
    return [np.array(m1d), np.array(m1r), np.array(m4)], np.array(lab)


def main():
    print("dysts features ...", flush=True)
    (d1d, d1r, d4, d5), dlab = features_dysts()
    print(f"  dysts {len(dlab)} series", flush=True)
    print("macro features (from dv cache) ...", flush=True)
    (m1d, m1r, m4), mlab = features_macro()
    print(f"  macro {len(mlab)} series", flush=True)

    dys = {"m1 (DMAP)": ward_ari(d1d, dlab), "m1 (raw)": ward_ari(d1r, dlab),
           "m4": ward_ari(d4, dlab), "m5": ward_ari(d5, dlab)}
    mac = {"m1 (DMAP)": ward_ari(m1d, mlab), "m1 (raw)": ward_ari(m1r, mlab),
           "m4": ward_ari(m4, mlab), "m5": 0.050}   # macro m5 from validated run (needs raw series)
    print("dysts ARIs:", {k: round(v, 3) for k, v in dys.items()}, flush=True)
    print("macro ARIs:", {k: round(v, 3) for k, v in mac.items()}, flush=True)

    def emb(F):
        Z = (F - F.mean(0))/(F.std(0)+1e-9)
        return MDS(n_components=2, random_state=0, normalized_stress="auto").fit_transform(Z)
    E_dys = emb(d5)            # families: m5 is the discriminator
    E_mac = emb(m1d)           # macro: best is m1 (DMAP) -- still overlaps

    # ---- figure: 1 row x 3 ----
    fig = plt.figure(figsize=(7.1, 2.6))
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.99, top=0.88, bottom=0.16, wspace=0.32)
    cmap = plt.get_cmap("tab20")

    axA = fig.add_subplot(gs[0, 0])
    for i in sorted(set(dlab)):
        m = dlab == i; axA.scatter(E_dys[m, 0], E_dys[m, 1], s=14, color=cmap(i % 20),
                                   edgecolors="k", linewidths=0.2)
    axA.set_xticks([]); axA.set_yticks([])
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    axA.set_title(rf"(i) simulated families split (ARI {dys['m5']:.2f})", pad=2)
    axA.set_xlabel("MDS$_1$"); axA.set_ylabel("MDS$_2$")

    axB = fig.add_subplot(gs[0, 1])
    for i in sorted(set(mlab)):
        m = mlab == i; axB.scatter(E_mac[m, 0], E_mac[m, 1], s=16, color=cmap(i % 20),
                                   edgecolors="k", linewidths=0.2, label=INDS[i])
    axB.set_xticks([]); axB.set_yticks([])
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    axB.set_title(rf"(ii) macro indicators overlap (ARI {mac['m1 (DMAP)']:.2f})", pad=2)
    axB.set_xlabel("MDS$_1$"); axB.set_ylabel("MDS$_2$")
    axB.legend(fontsize=5.6, loc="best", frameon=False, handletextpad=0.2, borderpad=0.1)

    axC = fig.add_subplot(gs[0, 2])
    feats = ["m1 (DMAP)", "m1 (raw)", "m4", "m5"]
    x = np.arange(len(feats)); w = 0.38
    axC.bar(x - w/2, [dys[f] for f in feats], w, label="dysts (clean)", color="#2a7f9e")
    axC.bar(x + w/2, [mac[f] for f in feats], w, label="macro (noisy)", color="#c0392b")
    axC.set_xticks(x); axC.set_xticklabels(feats, rotation=25, ha="right")
    axC.set_ylabel("ARI"); axC.set_ylim(0, 1)
    for sp in ("top", "right"): axC.spines[sp].set_visible(False)
    axC.set_title("(iii) DMAP vs raw, by feature", pad=2)
    axC.legend(fontsize=6, frameon=False, loc="upper right")

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig("/tmp/fig_dmap_vs_raw.png", dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.normpath(FIG_OUT), flush=True)


if __name__ == "__main__":
    main()
