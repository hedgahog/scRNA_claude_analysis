#!/usr/bin/env python3
"""
Dimensionality reduction and clustering of an ALREADY preprocessed PBMC dataset.

Input state (not redone here):
  - Cells are QC-filtered; QC metrics are in .obs.
  - .X is log-normalized AND scaled (it contains negative values).
  - .raw holds log-normalized, unscaled expression for the same genes.
  - Highly variable genes are flagged in .var['highly_variable'].

Normalization, log1p, scaling, filtering and QC-metric calculation are
deliberately skipped:
  - Re-normalizing or re-logging scaled data (negative values) would be
    mathematically invalid.
  - Re-scaling would distort the existing z-scores.
  - Re-filtering would change the cell/gene set that was already curated.
  - Raw counts are not available, so those steps could not be redone correctly.

Steps: validate -> QC report -> PCA -> neighbors -> UMAP -> Leiden -> save.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INPUT_PATH = Path("./data/pbmc_sample.h5ad")
OUTPUT_PATH = Path("processed_dataset.h5ad")

N_PCS_COMPUTE = 50          # PCs to compute (capped automatically if data is small)
N_PCS_NEIGHBORS = 30        # PCs used for the neighborhood graph
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5     # default Leiden resolution
LEIDEN_KEY = "leiden"
RANDOM_STATE = 0

REQUIRED_OBS_COLS = ["n_genes", "percent_mito", "n_counts"]
PRESERVED_OBS_COLS = ["louvain", "bulk_labels"]   # must remain untouched
HVG_COL = "highly_variable"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sc.settings.verbosity = 2


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def validate_input(adata: sc.AnnData) -> None:
    """Defensive checks that the data is in the expected preprocessed state."""
    # .X must be scaled (negative values present)
    x_min = adata.X.min()  # works for dense arrays and scipy sparse matrices
    if not x_min < 0:
        raise ValueError(
            f".X does not appear to be scaled (min value = {x_min:.3f}, no negative values). "
            "This script expects log-normalized AND scaled data in .X."
        )

    # .raw must exist so unscaled expression stays available for DE/markers
    if adata.raw is None:
        raise ValueError(
            ".raw is missing. It must hold the log-normalized, unscaled expression "
            "for downstream differential expression / marker gene analysis."
        )

    # QC columns
    missing_obs = [c for c in REQUIRED_OBS_COLS if c not in adata.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing expected .obs QC columns: {missing_obs}")

    # Highly variable gene flag
    if HVG_COL not in adata.var.columns:
        raise KeyError(f"Missing .var['{HVG_COL}'] column.")
    n_hvg = int(adata.var[HVG_COL].sum())
    if n_hvg == 0:
        raise ValueError(f".var['{HVG_COL}'] contains no highly variable genes.")

    # Columns that must be preserved
    missing_keep = [c for c in PRESERVED_OBS_COLS if c not in adata.obs.columns]
    if missing_keep:
        log.warning("Columns expected to be preserved are absent from .obs: %s", missing_keep)

    log.info(
        "Validation passed: X range [%.3f, %.3f], .raw present (%d genes), %d HVGs.",
        x_min, adata.X.max(), adata.raw.n_vars, n_hvg,
    )


def report_qc(adata: sc.AnnData) -> None:
    """Print summary statistics of per-cell QC metrics (report only, no filtering)."""
    summary = adata.obs[REQUIRED_OBS_COLS].describe().T
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\n=== QC summary (no filtering applied) ===")
        print(summary)
        print()


def run_pca(adata: sc.AnnData, n_comps: int) -> None:
    """
    PCA on the scaled .X restricted to highly variable genes.
    Uses the mask_var API (scanpy >= 1.10) and falls back to use_highly_variable
    for older versions. Neither changes .X, .raw or the gene set of the object.
    """
    try:
        sc.pp.pca(adata, n_comps=n_comps, mask_var=HVG_COL,
                  svd_solver="arpack", random_state=RANDOM_STATE)
    except TypeError:
        sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=True,
                  svd_solver="arpack", random_state=RANDOM_STATE)


def run_leiden(adata: sc.AnnData) -> None:
    """
    Leiden clustering on the neighborhood graph.
    Prefers the igraph backend (the future scanpy default) and falls back to
    leidenalg for older scanpy versions.
    """
    try:
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY,
                     random_state=RANDOM_STATE, flavor="igraph",
                     n_iterations=2, directed=False)
    except (TypeError, ValueError, ImportError) as err:
        log.warning("igraph-flavor Leiden unavailable (%s); falling back to leidenalg.", err)
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY,
                     random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1. Load input -------------------------------------------------------- #
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")
    adata = sc.read_h5ad(INPUT_PATH)
    log.info("Loaded %s: %d cells x %d genes", INPUT_PATH, adata.n_obs, adata.n_vars)

    # 2. Validate ---------------------------------------------------------- #
    validate_input(adata)

    # Snapshots used to prove that existing annotations/.raw are untouched
    preserved_before = {c: adata.obs[c].copy() for c in PRESERVED_OBS_COLS if c in adata.obs}
    raw_shape_before = adata.raw.shape

    # 3. QC review (report only) ------------------------------------------- #
    report_qc(adata)

    # 4. Dimensionality reduction ------------------------------------------ #
    # Recompute PCA on the scaled .X with HVGs only. Existing X_pca / X_umap /
    # neighbors entries are overwritten with freshly computed, consistent ones.
    n_hvg = int(adata.var[HVG_COL].sum())
    n_comps = min(N_PCS_COMPUTE, n_hvg - 1, adata.n_obs - 1)
    n_pcs_nb = min(N_PCS_NEIGHBORS, n_comps)
    if n_pcs_nb < N_PCS_NEIGHBORS:
        log.warning("Using %d PCs for neighbors instead of %d (limited by data size).",
                    n_pcs_nb, N_PCS_NEIGHBORS)

    log.info("Running PCA (%d components) on %d highly variable genes", n_comps, n_hvg)
    run_pca(adata, n_comps)

    log.info("Computing neighborhood graph (n_neighbors=%d, n_pcs=%d)", N_NEIGHBORS, n_pcs_nb)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs_nb,
                    use_rep="X_pca", random_state=RANDOM_STATE)

    log.info("Computing UMAP embedding")
    sc.tl.umap(adata, random_state=RANDOM_STATE)

    # 5. Clustering -------------------------------------------------------- #
    log.info("Running Leiden clustering (resolution=%.2f) -> obs['%s']",
             LEIDEN_RESOLUTION, LEIDEN_KEY)
    run_leiden(adata)

    cluster_counts = adata.obs[LEIDEN_KEY].value_counts().sort_index(
        key=lambda idx: idx.astype(int))
    print("\n=== Leiden cluster sizes ===")
    print(cluster_counts.to_string())
    print()

    # Sanity checks: existing annotations and .raw must be intact
    for col, before in preserved_before.items():
        if not adata.obs[col].equals(before):
            raise RuntimeError(f"obs['{col}'] was unexpectedly modified.")
    if adata.raw is None or adata.raw.shape != raw_shape_before:
        raise RuntimeError(".raw was unexpectedly modified or removed.")
    log.info("Verified: 'louvain', 'bulk_labels' and .raw are unchanged.")

    # 6. Save -------------------------------------------------------------- #
    # .raw (log-normalized, unscaled) is written with the object so downstream
    # DE/marker analysis, e.g. sc.tl.rank_genes_groups(adata, 'leiden', use_raw=True),
    # can use the unscaled values.
    adata.write_h5ad(OUTPUT_PATH, compression="gzip")
    log.info("Saved processed AnnData to %s", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # clear, user-facing failure message
        log.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)