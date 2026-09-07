"""
GW Attractor Landscape — Pipeline Configuration.

Status: ACTIVE

All configurable parameters for the full pipeline:
classification, embedding, descriptors, DMAP, GW alignment,
convergence, and validation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassifyConfig:
    """Series classification parameters."""
    # ADF test
    adf_max_lags: Optional[int] = None  # None = auto (12*(nobs/100)^{1/4})
    adf_regression: str = "ct"  # constant + trend

    # KPSS test
    kpss_regression: str = "ct"
    kpss_nlags: str = "auto"

    # HEGY seasonal unit roots
    hegy_enabled: bool = True
    hegy_significance: float = 0.05

    # GPH fractional d (for inconclusive ADF/KPSS)
    gph_bandwidth_exp: float = 0.65  # m = n^0.65

    # Bai-Perron structural breaks
    break_model: str = "l2"  # Pelt cost model
    break_penalty: str = "bic"  # BIC penalty
    break_min_size: int = 24  # minimum segment (2 years monthly)

    # Fractional integration threshold
    frac_d_threshold: float = 0.5  # d >= 0.5 → treat as I(1)


@dataclass
class EmbedConfig:
    """Takens embedding parameters."""
    # AMI for tau selection
    ami_max_lag: int = 50
    ami_bins: int = 64

    # FNN for dimension selection
    fnn_max_dim: int = 20
    fnn_rtol: float = 15.0
    fnn_atol: float = 2.0
    fnn_noise_floor: float = 0.01  # adaptive threshold proportional to (eps/sigma_s)^2

    # Dimension bounds
    dim_min: int = 3
    dim_max: int = 16
    dim_default: int = 8  # fallback if FNN inconclusive

    # Tau bounds
    tau_min: int = 1
    tau_max: int = 20
    tau_default: int = 3

    # Global normalization
    normalize: bool = True  # z-score rows of delay matrix

    # Offset rule: dim * tau (NOT (dim-1)*tau)
    # enforced in code, not configurable


@dataclass
class DescriptorConfig:
    """16D descriptor computation parameters."""
    # Permutation entropy
    pe_orders: tuple = (3, 4, 5)
    pe_delay: int = 1

    # Lyapunov exponent (Rosenstein)
    lyap_min_tsep: int = -1  # -1 = auto (set to tau); 0 = no separation (not recommended)
    lyap_max_iter: int = 500

    # Correlation dimension (Grassberger-Procaccia)
    d2_max_points: int = 2000
    d2_n_radii: int = 20  # log-spaced radii for scaling region

    # RQA
    rqa_threshold_percentile: float = 10.0  # % of max distance
    rqa_min_diag: int = 2
    rqa_theiler: int = 1

    # Mutual information (KSG)
    mi_k: int = 5  # k-nearest neighbors for KSG estimator

    # Persistent homology
    ph_max_points: int = 150  # subsample for ripser
    ph_max_dim: int = 1  # H0 + H1

    # Beta1 significance (FT surrogates)
    beta1_n_surrogates: int = 39  # rank-based p < 0.025

    # Spectral entropy
    hspec_nperseg: int = 256

    # Curvature (geodesic/Euclidean diameter ratio)
    curvature_k_neighbors: int = 8  # for kNN graph

    # Cross-domain invariant subset (provably transform-invariant only)
    invariant_indices: tuple = (2, 3, 4, 13, 14)  # PE3, PE4, PE5, PH_total, PH_max


@dataclass
class DmapConfig:
    """Per-series diffusion map parameters."""
    # Kernel
    n_components: int = 10  # diffusion coordinates to keep
    alpha: float = 1.0  # Laplace-Beltrami normalization
    k_adaptive: int = 10  # k-th neighbor for adaptive bandwidth

    # Theiler window
    theiler_window: int = 0  # 0 = auto (set to tau)

    # Support size
    n_support: int = 200  # target support points per series

    # Thresholds for FPS/Nystrom
    nystrom_threshold: int = 2000  # use dense DMAP below this


@dataclass
class AlignConfig:
    """GW alignment parameters."""
    # Landmarks
    n_landmarks: int = 300  # FPS landmarks for barycenter

    # GW barycenter
    epsilon_candidates: tuple = (0.01, 0.05, 0.1, 0.5)
    epsilon_default: float = 0.05
    max_iter: int = 100
    tol: float = 1e-7

    # MDS embedding of barycenter
    mds_max_dim: int = 20
    mds_stress_threshold: float = 0.15

    # Projection (legacy centroid approach — superseded by TransportDmapConfig)
    n_support: int = 200  # must match DmapConfig.n_support
    projection_epsilon: float = 0.001

    # Parallelism
    n_workers: int = 1


@dataclass
class TransportDmapConfig:
    """Transport-plan DMAP parameters (the AoA coordinate system).

    Architecture: each series → per-series DMAP → n_support-point skeleton
    → GW transport plan to barycenter → Hellinger DMAP on transport plans.

    The centroid projection (T @ bary_MDS → mean) is identically zero
    for uniform-marginal transport plans with centered MDS. Transport-plan
    DMAP discovers the intrinsic nonlinear geometry of the Birkhoff polytope
    where the plans live.
    """
    # Skeleton support size for per-series DMAP
    # Sensitivity sweep (100 series): phase transition at n~35 (gap 1.37→2.39).
    # Bootstrap stability (10 reps × 80 series):
    #   n=30: gap=1.02±0.01 (no structure)
    #   n=35: gap=2.05±0.76 (high variance, min=1.02)
    #   n=40: gap=2.12±0.28 (stable, min=1.66)
    # n=40 is optimal: strong gap (2.12) with low variance across subsamples.
    n_support: int = 40

    # Entropic regularization for GW transport to barycenter
    # Small epsilon keeps T in interior of Birkhoff polytope (needed for Hellinger)
    # while preserving structural correspondence
    gw_epsilon: float = 0.01

    # Nystrom-DMAP on transport plans
    n_landmarks_dmap: int = 800  # FPS landmarks in Hellinger space
    n_components: int = 20  # diffusion coordinates to keep
    dmap_alpha: float = 1.0  # Laplace-Beltrami density normalization
    dmap_k_bandwidth: int = 10  # k-th NN for adaptive bandwidth (auto-tuned by persistence)

    # Per-series DMAP components (before GW)
    # At n_support=40, using only 10 discards 75% of resolvable structure.
    # R3 review: increase to n_support-1 to preserve all geometric info.
    # None = auto (n_support - 1).
    n_dmap_components: int = None  # auto: n_support - 1

    # Domain discovery
    min_domain_size: int = 30  # minimum series for per-domain DMAP
    # Bandwidth persistence: k-th NN range for sweep
    # Set to None for automatic range based on corpus size
    k_range: tuple = None  # e.g., (3, 5, 7, 10, 15, 20, 30, 50)
    # Eigengap stability check: 50 DMAP calls. Set False for fast runs.
    run_stability_check: bool = True


@dataclass
class MultiReferenceConfig:
    """Multi-reference transport plan parameters for cross-domain coupling.

    Each series gets K transport plans (one per domain barycenter).
    Entropy gating skips near-uniform foreign plans. The coupling
    signal emerges from the GW geometry — no supervised losses.
    """
    probe_epsilon: float = 0.1         # larger ε for fast foreign-plan probe
    entropy_gate: float = 0.95         # H/H_max above this → skip (near-uniform)
    normalize_blocks: bool = True      # per-block mean/std normalization
    min_block_weight: float = 0.01     # floor to prevent zero-weight blocks
    k_bandwidth_product: int = 30      # larger k for high-D product space
    n_components_product: int = 30     # more components to capture coupling modes
    n_eigenvectors_coupling: int = 20  # eigenvectors to analyze for coupling matrix


@dataclass
class ConvergeConfig:
    """Convergence experiment parameters."""
    # Replications
    n_reps_small: int = 30   # domains < 10K
    n_reps_large: int = 20   # domains 10K-100K
    n_reps_universal: int = 10  # 281K+

    # Strategies
    strategies: tuple = ("random", "stratified", "fps_diversity")

    # Convergence thresholds
    procrustes_threshold: float = 0.05
    eigenvalue_ratio_threshold: float = 0.02
    probe_rank_threshold: float = 0.98

    # Running-mean stability
    stability_window: int = 10
    stability_pct: float = 0.05  # < 5% of overall std
    consecutive_stable: int = 3

    # Bootstrap
    n_bootstrap: int = 500
    subsample_fraction: float = 0.6  # 60% subsample


@dataclass
class ValidateConfig:
    """Macroeconomic validation parameters."""
    # Permutation null
    n_permutations: int = 1000
    alpha: float = 0.01

    # Cross-validation
    cv_splits: int = 2  # 50/50
    cv_corr_threshold: float = 0.5

    # Time stability
    break_years: tuple = (1990, 2009, 2020)
    cov_threshold: float = 0.5

    # Quantitative calibration
    spearman_threshold: float = 0.3

    # Multiple testing
    fdr_q: float = 0.05  # Benjamini-Hochberg

    # Falsification
    n_falsification_surrogates: int = 100  # for each F-control

    # HP filter lambda (Ravn-Uhlig 2002 scaling rule: 1600 * n^4)
    hp_lambda_quarterly: float = 1600.0
    hp_lambda_monthly: float = 129600.0  # 1600 * 3^4
    hp_lambda_annual: float = 6.25  # 1600 * (1/4)^4


@dataclass
class PipelineConfig:
    """Master configuration for the full GW pipeline."""
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    dmap: DmapConfig = field(default_factory=DmapConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    transport_dmap: TransportDmapConfig = field(default_factory=TransportDmapConfig)
    multi_reference: MultiReferenceConfig = field(default_factory=MultiReferenceConfig)
    converge: ConvergeConfig = field(default_factory=ConvergeConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)

    # Parallelism
    n_workers: int = 10
    chunk_size: int = 500  # series per chunk for multiprocessing

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10000  # save every N series

    # Logging
    verbose: bool = True

    def to_dict(self) -> dict:
        """Serialize config to dict for reproducibility logging."""
        import dataclasses
        return dataclasses.asdict(self)
