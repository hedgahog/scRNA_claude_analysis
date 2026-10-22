from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse

RUNS_DIR = Path(__file__).resolve().parent / "runs"
FILE_NAME = "final_adata.h5ad"
N_ROWS = 5

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n=== {title} ===")


# Run folders are named YYYYMMDD_HHMM_<model>, so name order is time order.
# Skip runs that stopped before saving the file.
candidates = sorted(RUNS_DIR.glob(f"*/{FILE_NAME}"), key=lambda p: p.parent.name)
if not candidates:
    raise SystemExit(f"No {FILE_NAME} found under {RUNS_DIR}")
path = candidates[-1]

adata = ad.read_h5ad(path)
print(f"File: {path}")

section("Summary")
print(adata)

section(f"obs (first {N_ROWS} cells)")
print(adata.obs.head(N_ROWS))

# Label-like columns are more readable as counts than as rows
for col in adata.obs.select_dtypes(include=["category", "object"]).columns:
    section(f"obs['{col}'] counts")
    print(adata.obs[col].value_counts().to_string())

section(f"var (first {N_ROWS} genes)")
print(adata.var.head(N_ROWS))

section(f"X (first {N_ROWS} cells x {N_ROWS} genes)")
X = adata.X[:N_ROWS, :N_ROWS]
X = X.toarray() if sparse.issparse(X) else X
print(pd.DataFrame(X, index=adata.obs_names[:N_ROWS], columns=adata.var_names[:N_ROWS]).round(3))

section("obsm")
for key, value in adata.obsm.items():
    print(f"{key}: shape {value.shape}")

section("layers")
# Newer anndata lists .X itself as a None key; show only real layers
print([k for k in adata.layers.keys() if k is not None] or "none")

section("uns keys")
print(list(adata.uns.keys()))
