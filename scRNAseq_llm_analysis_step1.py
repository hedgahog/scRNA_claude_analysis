# NAME:  scRNAseq_llm_analysis_step1.py

# IMPORTS
import os
import scanpy as sc
from anthropic import Anthropic

# Initialize the Anthropic client using the environment variable 'ANTHROPIC_API_KEY'
client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
file_path = "./data/pbmc_sample.h5ad"   # Change to point to the desired .h5ad file

# ANALYSIS FUNS
def analyze_scrna_with_claude(file_path: str):
    """
    Loads a .h5ad file, extracts metadata, and uses Claude Sonnet 5
    to interpret cell clusters and recommend cell type annotations.
    """
    print(f"--- Loading dataset: {file_path} ---")
    # 1. Load the scRNA-seq object
    adata = sc.read_h5ad(file_path)

    # 2. Extract key metrics for contextualizing the LLM
    num_cells = adata.n_obs
    num_genes = adata.n_vars
    obs_columns = list(adata.obs.columns)
    var_columns = list(adata.var.columns)

    # Safely look for cluster columns or highly variable gene lists
    clusters_found = [col for col in obs_columns if 'cluster' in col or 'leiden' in col or 'louvain' in col]

    cluster_context = ""
    if clusters_found:
        primary_cluster_col = clusters_found[0]
        cluster_counts = adata.obs[primary_cluster_col].value_counts().to_dict()
        cluster_context = f"Detected clusters ({primary_cluster_col}): {cluster_counts}"
    else:
        cluster_context = "No pre-computed structural clusters found in metadata."

    # 3. Construct a clear analysis prompt for Claude
    prompt = f"""
    Act as an expert bioinformatician and Python developer specializing in single-cell RNA sequencing (scRNA-seq) analysis. 

    Execute a complete, production-ready Python script using the Scanpy library to perform a standard quality control (QC), filtering, normalization, dimensionality reduction, and clustering workflow on a single-cell dataset.
    
    Here are the requirements for the script:
    1. Input: Read an existing input file at path "./data/pbmc_sample.h5ad".
    2. Quality Control & Filtering:
       - Calculate QC metrics (mitochondrial, ribosomal, and hemoglobin genes if applicable).
       - Filter out low-quality cells based on standard thresholds (e.g., min_genes=200, min_cells=3).
       - Filter cells based on mitochondrial content (e.g., keeping cells with < 5% or 10% mitochondrial counts).
    3. Normalization & Log Transformation:
       - Normalize total counts per cell to 10,000.
       - Log-transform the data (log1p).
    4. Feature Selection & Scale:
       - Identify Highly Variable Genes (HVGs).
       - Scale the data to unit variance and zero mean, clipping max values to 10.
    5. Dimensionality Reduction:
       - Run Principal Component Analysis (PCA).
       - Compute the neighborhood graph (using an appropriate number of PCs, like 30).
       - Run UMAP embedding.
    6. Clustering:
       - Cluster the cells using the Leiden algorithm (set a default resolution of 0.5).
    7. Output: Save the final, fully processed AnnData object to a new file named "processed_dataset.h5ad".
    
    Please include:
    - Clear comments explaining each step.
    - Proper handling of the raw data layer (`adata.raw`) before scaling so downstream differential expression/marker gene analysis can use unscaled data.
    - Defensive checks (e.g., verifying if mitochondrial genes exist in the dataset using 'MT-' or 'mt-' prefixes).
    
    Provide:
    1) The clean Python code block.
    2) Save "processed_dataset.h5ad" into the subdirectory folder named '/outputs'.
    """

    print("--- Sending dataset context to Claude Sonnet 5 ---")

    # 4. Make the API Call to Claude Sonnet 5
    response = client.messages.create(
        model="claude-sonnet-5-5",  # Official API identifier for Sonnet 5
        max_tokens=4005,
        messages=[
            {
                "role": "user",
                "content": (prompt)
            }
        ]
    )

    # 5. Extract and print Claude's response
    # Find the first block that contains text
    text_block = next((block for block in response.content if block.type == "text"), None)

    analysis_strategy = "No text response generated"    # Init default response
    if text_block:
        analysis_strategy = text_block.text

    # Extract code block and save to a local file
    # This regex looks for text wrapped inside ```python ... ``` markdown blocks
    import re
    code_block_match = re.search(r"```python\s*(.*?)\s*```", analysis_strategy, re.DOTALL)

    if code_block_match:
        generated_code = code_block_match.group(1)
    else:
        # Fallback if Claude omitted markdown blocks and just provided raw code
        generated_code = analysis_strategy

    output_filename = "cs223_llm_step1_scanpy_analysis.py"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(generated_code)

    print(f"🎉 Successfully created and wrote code to: {output_filename}")

    #analysis_strategy = response.content[0].text
    print("\n--- Claude's Recommended Analysis & Code Blueprint ---\n")
    print(analysis_strategy)


if __name__ == "__main__":
    # Ensure your ANTHROPIC_API_KEY is either exported in your terminal:
    # export ANTHROPIC_API_KEY="your-api-key-here"
    # -OR-
    # You have specified dirctly specified the client = Anthropic(api_key= ...) statement above.

    file_path = "./data/pbmc_sample.h5ad"

    if os.path.exists(file_path):
        analyze_scrna_with_claude(file_path)
    else:
        print(f"File '{file_path}' not found. Please provide a valid path to your single-cell data.")

print(">>>>>>>>>> DONE <<<<<<<<<<")
