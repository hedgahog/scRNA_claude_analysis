# NAME:  scRNAseq_llm_analysis_step1.py

# IMPORTS
import os
import subprocess
import sys
import scanpy as sc
from anthropic import Anthropic
from anthropic.types.beta import BetaMessageParam, BetaToolParam, BetaToolResultBlockParam

# Initialize the Anthropic client using the environment variable 'ANTHROPIC_API_KEY'
client = Anthropic(api_key="your api key here")  # reads ANTHROPIC_API_KEY from the environment
file_path = "./data/pbmc_sample.h5ad"   # Change to point to the desired .h5ad file

MODEL = "claude-sonnet-5-5"             # Official API identifier for Sonnet 5.5
OUTPUT_DIR = "./output"                 # All AnnData outputs are written here
OUTPUT_H5AD_PATH = os.path.join(OUTPUT_DIR, "processed_dataset.h5ad")
SCRIPTS_DIR = os.path.join(OUTPUT_DIR, "agent_scripts")   # Copy of every script Claude runs
MAX_AGENT_TURNS = 25                    # Safety cap on the tool-use loop


# TOOL FUNS
def execute_python_code(code: str, step: int) -> tuple[str, bool]:
    """
    Tool allowing Claude to execute Python code locally via a subprocess.
    The code is saved to ./output/agent_scripts/step_XX.py (so the run is
    reproducible) and executed with the same interpreter running this script.
    Returns (combined stdout/stderr, is_error).
    """
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    script_path = os.path.join(SCRIPTS_DIR, f"step_{step:02d}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.getcwd()
        )
        output = f"EXIT CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "Execution failed: script timed out after 600 seconds.", True
    except Exception as e:
        return f"Execution failed: {str(e)}", True


tools: list[BetaToolParam] = [
    {
        "name": "execute_python",
        "description": (
            "Executes a complete, self-contained Python script in a fresh local Python process "
            "(scanpy, anndata, numpy and pandas are installed) and returns its exit code, stdout and stderr. "
            "Variables do NOT persist between calls, so every script must load whatever it needs from disk. "
            "Use this to inspect .h5ad files, run the scanpy pipeline, and save/verify outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The full Python script to run."
                }
            },
            "required": ["code"]
        }
    }
]


# ANALYSIS FUNS
def analyze_scrna_with_claude(file_path: str):
    """
    Loads a .h5ad file, extracts metadata, and has Claude Sonnet 5.5 perform the
    scanpy QC -> dimensionality reduction -> clustering workflow itself by running
    Python code locally through the execute_python tool. The processed AnnData
    object is saved to ./output.
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Construct a clear analysis prompt for Claude
    prompt = f"""
    Act as an expert bioinformatician specializing in single-cell RNA sequencing (scRNA-seq) analysis.

    Your task is to PERFORM a standard Scanpy quality control (QC), filtering, normalization,
    dimensionality reduction, and clustering workflow on a single-cell dataset. Do not just write
    the code for me: run every step yourself by writing Python scripts and executing them with
    your `execute_python` tool, then read the output and continue.

    Important facts about the `execute_python` tool:
    - Each call runs in a NEW Python process from the project root, so nothing in memory carries over
      between calls. Every script must read its input from disk (and write anything later steps need).
    - Scripts should print the information you need to check their results (shapes, value ranges,
      column names, cluster sizes, etc.).
    - If a script fails, read the error, fix the script, and run it again.

    Dataset context (already extracted):
    - Input file: "{file_path}"
    - Shape: {num_cells} cells x {num_genes} genes
    - .obs columns: {obs_columns}
    - .var columns: {var_columns}
    - {cluster_context}

    Perform the following steps:
    1. Inspect: Load "{file_path}" with `sc.read_h5ad` and inspect it before changing anything:
       the range of .X (are there negative values, i.e. is it already scaled?), whether .raw exists,
       the .obs QC columns, whether .var['highly_variable'] exists, and any existing .obsm/.uns entries.
    2. Quality Control & Filtering:
       - Calculate QC metrics (mitochondrial, ribosomal, and hemoglobin genes if applicable). First verify
         that mitochondrial genes exist using 'MT-' or 'mt-' prefixes.
       - Filter out low-quality cells based on standard thresholds (e.g., min_genes=200, min_cells=3)
         and mitochondrial content (e.g., keeping cells with < 5% or 10% mitochondrial counts).
       - If the inspection shows the data is ALREADY QC-filtered and preprocessed (QC metrics already in .obs,
         .X scaled with negative values, no raw counts available), do NOT recalculate or re-filter. Instead,
         print a summary of the existing QC metrics (e.g. n_genes, percent_mito, n_counts) as a report only.
    3. Normalization & Log Transformation:
       - Normalize total counts per cell to 10,000 and log-transform the data (log1p).
       - Skip this if .X is already log-normalized/scaled: re-normalizing or re-logging scaled data is invalid.
    4. Feature Selection & Scale:
       - Identify Highly Variable Genes (HVGs).
       - Store the log-normalized, unscaled data in `adata.raw` BEFORE scaling so downstream differential
         expression/marker gene analysis can use unscaled data.
       - Scale the data to unit variance and zero mean, clipping max values to 10.
       - Skip any of these that were already done (existing .var['highly_variable'], existing .raw, scaled .X);
         never overwrite an existing .raw.
    5. Dimensionality Reduction:
       - Run Principal Component Analysis (PCA) on the highly variable genes (up to 50 components,
         capped by the data size), random_state=0.
       - Compute the neighborhood graph (n_neighbors=15, using an appropriate number of PCs, like 30).
       - Run UMAP embedding (random_state=0).
    6. Clustering:
       - Cluster the cells using the Leiden algorithm with a resolution of 0.5, storing the labels in
         .obs['leiden'] (random_state=0). Print the size of each cluster.
       - Existing annotation columns such as 'louvain' and 'bulk_labels' must remain unchanged; check this.
    7. Output: Save the final, fully processed AnnData object (the `adata` variable) to
       "{OUTPUT_H5AD_PATH}" inside the "{OUTPUT_DIR}" folder, creating the folder if needed
       (`os.makedirs("{OUTPUT_DIR}", exist_ok=True)`). Write it with `adata.write_h5ad(..., compression="gzip")`
       so .raw is saved along with it. Any intermediate .h5ad files you need between tool calls must
       also go in "{OUTPUT_DIR}".
    8. Verify: In a separate tool call, reload "{OUTPUT_H5AD_PATH}" and confirm it contains .raw,
       .obsm['X_pca'], .obsm['X_umap'], .obs['leiden'], and the neighbors graph.

    When you are finished, reply with a short summary of what you did, which steps you skipped and why,
    the number of Leiden clusters with their sizes, and the path of the saved file.
    """

    print(f"--- Starting {MODEL} agent workflow ---")

    # 4. Run the agent loop: Claude asks to run code, we run it locally and send back the output
    conversation_history: list[BetaMessageParam] = [{"role": "user", "content": prompt}]
    step = 0
    response = None

    for _ in range(MAX_AGENT_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=tools,
            messages=conversation_history,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default"     # Retry on a fallback model if the request is declined
        )
        # Append the full content (text + tool_use blocks) to keep the history valid
        conversation_history.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        # Run every tool call in this turn and return all results in a single user message
        tool_results: list[BetaToolResultBlockParam] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "execute_python":
                step += 1
                code_to_run = str(block.input["code"])
                print(f"\n--- [Claude Execution Request: step {step}] ---\n{code_to_run}\n")

                tool_output, is_error = execute_python_code(code_to_run, step)
                print(f"--- [Execution Result: step {step}] ---\n{tool_output}\n")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                    "is_error": is_error
                })

        conversation_history.append({"role": "user", "content": tool_results})
    else:
        print(f"⚠️\n"
               f"Stopped after {MAX_AGENT_TURNS} turns without Claude finishing.")

    if response is None:    # Only possible if MAX_AGENT_TURNS is 0
        return

    if response.stop_reason == "refusal":
        print("⚠️ Claude declined the request.")
    elif response.stop_reason == "max_tokens":
        print("⚠️ Claude's response was cut off (max_tokens reached).")

    # 5. Extract and print Claude's final summary
    analysis_summary = "".join(block.text for block in response.content if block.type == "text")
    print("\n--- Claude's Analysis Summary ---\n")
    print(analysis_summary or "No text response generated")

    if os.path.exists(OUTPUT_H5AD_PATH):
        print(f"🎉 Processed AnnData saved to: {OUTPUT_H5AD_PATH}")
    else:
        print(f"⚠️ Expected output file was not found: {OUTPUT_H5AD_PATH}")
    print(f"Scripts executed by Claude are in: {SCRIPTS_DIR}")


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
