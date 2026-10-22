import os
import re
import sys
from datetime import datetime
from pathlib import Path

MODEL = "claude-sonnet-5-5"  # switch to claude-sonnet-5-5 for the Sonnet run

# One folder per run so results and logs never overwrite each other
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / ".data"
RUN_DIR = PROJECT_DIR / "runs" / f"{datetime.now():%Y%m%d_%H%M}_{MODEL.removeprefix('claude-')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirror console output into run.log."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        self._log.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_log_file = open(RUN_DIR / "run.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"Run folder: {RUN_DIR}")

# Agent-generated plotting code runs in this process; the default Tk backend
# crashes at exit when figures are left open, so render off-screen instead.
os.environ.setdefault("MPLBACKEND", "Agg")

# Newer Claude models reject a trailing assistant message (prefill), but Biomni
# appends each <observation> as an AIMessage. Keep the observation and add a
# human turn after it; dropping it makes the model regenerate the same code forever.
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.outputs import ChatGeneration, ChatResult

original_generate = ChatAnthropic._generate
MAX_EMPTY_RESPONSES = 3  # consecutive empty replies before ending the run cleanly
_empty_streak = [0]


def patched_generate(self, messages, *args, **kwargs):
    if isinstance(messages, list) and messages and isinstance(messages[-1], AIMessage):
        # Copy so the LangGraph state is not mutated
        messages = list(messages) + [HumanMessage(content="Please proceed.")]
    result = original_generate(self, messages, *args, **kwargs)
    msg = result.generations[0].message
    if msg.text:
        _empty_streak[0] = 0
        return result

    # Empty text makes Biomni's parser retry. Its "give up after 2 parse errors"
    # check looks for the retry text in an AIMessage although it is added as a
    # HumanMessage, so it never fires and the agent loops until the step cap.
    _empty_streak[0] += 1
    blocks = [b.get("type") for b in msg.content] if isinstance(msg.content, list) else msg.content
    print(f"EMPTY RESPONSE #{_empty_streak[0]}: stop_reason={msg.response_metadata.get('stop_reason')} "
          f"blocks={blocks} usage={msg.usage_metadata}")
    if _empty_streak[0] >= MAX_EMPTY_RESPONSES:
        stop = "<solution>Stopped: the model returned empty responses repeatedly.</solution>"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=stop))])
    return result


ChatAnthropic._generate = patched_generate

# Claude 5 models return 400 "`temperature` is deprecated for this model", but
# Biomni always passes one (falls back to 0.7 when unset). Drop it from the
# request for those models only, so Haiku 4.5 keeps its configured temperature.
original_get_request_payload = ChatAnthropic._get_request_payload


def patched_get_request_payload(self, *args, **kwargs):
    payload = original_get_request_payload(self, *args, **kwargs)
    if re.match(r"claude-(sonnet|opus|fable)-5", str(payload.get("model", ""))):
        payload.pop("temperature", None)
        # langchain-anthropic moves it into extra_body for these models
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            extra_body.pop("temperature", None)
            if not extra_body:
                payload.pop("extra_body")
    return payload


ChatAnthropic._get_request_payload = patched_get_request_payload
# ---------------------------------
# Load the API key from .env into this process only, so it never lives in code
# and Claude Code itself keeps using its own login instead of this key.
from dotenv import load_dotenv
load_dotenv()
if os.getenv("ANTHROPIC_API_KEY", "your-api-key-here") == "your-api-key-here":
    raise RuntimeError("Set ANTHROPIC_API_KEY in .env before running.")

# Now safely import the Biomnni agent
from biomni.config import default_config
from biomni.agent import A1
from langgraph.errors import GraphRecursionError
# Set defaults for running Biomni
default_config.timeout_seconds = 60  # Kill execution if it hangs too long
default_config.temperature = 0.1
# BiomniConfig has no max_iterations field; the real cap is the LangGraph
# recursion_limit that go() hardcodes to 500, so we override it below.
MAX_GRAPH_STEPS = 40  # generate+execute count as 2 steps, so ~20 iterations

# 1. Initialize the Biomni A1 agent
# path must match where the dataset lives (.data); an empty expected_data_lake_files
# skips the multi-GB Biomni data lake download, which this analysis does not need.
agent = A1(path=str(DATA_DIR), llm=MODEL, expected_data_lake_files=[])


class _CappedApp:
    """Wrap the compiled graph so go() runs with a small recursion_limit."""

    def __init__(self, app):
        self._app = app

    def stream(self, inputs, *args, **kwargs):
        config = dict(kwargs.pop("config", None) or {})
        config["recursion_limit"] = MAX_GRAPH_STEPS
        return self._app.stream(inputs, *args, config=config, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


agent.app = _CappedApp(agent.app)

# 2. Attach your custom MCP configuration file for tool enhancement (optional)
mcp_config = PROJECT_DIR / "mcp_config.yaml"
if mcp_config.exists():
    agent.add_mcp(config_path=str(mcp_config))

# 3. Define your single-cell bioinformatics tasks as a single cohesive prompt
sc_rna_task = (
    f"Analyze this single-cell RNA-seq dataset located at {(DATA_DIR / 'pbmc_sample.h5ad').as_posix()}:\n"
    "1. Run quality control (QC) filtering out low-quality cells (high mitochondrial ratio or low counts).\n"
    "2. Perform normalization, log-transformation, and find highly variable genes.\n"
    "3. Compute PCA and cluster the cell populations using a UMAP embedding.\n"
    "4. Annotate the cell clusters using canonical marker genes.\n"
    "5. Identify differentially expressed genes (DEGs) between the Control and Treated conditions.\n"
    f"Treat the input as read-only. Save every output (plots, tables, h5ad) in {RUN_DIR.as_posix()}/ "
    "and do not write anywhere else."
)

# 4. Pass the string into the agent execution loop
print("Starting single-cell RNA-seq analysis via Biomni LLM...")
# Agent code runs in this process and saves plots with relative paths, so
# working from the run folder keeps them (and the report) out of the project root.
os.chdir(RUN_DIR)
try:
    result = agent.go(sc_rna_task)
except GraphRecursionError:
    # Stopped by the step cap instead of looping; keep what the agent produced so far
    print(f"Stopped after {MAX_GRAPH_STEPS} graph steps without a final <solution>.")
    result = agent.log

# Save every AnnData the agent left in its REPL, even if it never wrote one itself
# (e.g. the step cap hit). Agent code runs in Biomni's persistent namespace.
from anndata import AnnData
from biomni.tool.support_tools import _persistent_namespace

saved_ids = set()
for name, obj in list(_persistent_namespace.items()):
    # The same object is often bound to several names; save it once
    if isinstance(obj, AnnData) and id(obj) not in saved_ids:
        saved_ids.add(id(obj))
        out_path = RUN_DIR / f"final_{name}.h5ad"
        try:
            obj.write_h5ad(out_path)
            print(f"Saved AnnData '{name}' {obj.shape} to {out_path}")
        except Exception as e:
            print(f"Could not save AnnData '{name}': {e}")
if not saved_ids:
    print("No AnnData object found in the agent's namespace.")

# 5. Output the textual insights and interpretation generated by the agent
print("\n--- Analysis Report Summary ---")
print(result)

# 6. Save the full execution history and generated plots/code to a PDF report
report_pdf = "scrna_seq_analysis_report.pdf"
agent.save_conversation_history(report_pdf)
if os.path.exists(report_pdf):
    print(f"Analysis complete. Detailed report saved to {report_pdf}")
else:
    # Biomni's PDF export uses signal.SIGALRM, which does not exist on Windows,
    # so fall back to the markdown it builds internally.
    report_md = "scrna_seq_analysis_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(agent._generate_markdown_content(include_images=False))
    print(f"PDF export failed; report saved as {report_md} instead.")
print()
