# NAME:  cs223_in_class_exercise_2_solution.py

import os
import subprocess
import sys

from anthropic import Anthropic

# Ensure your API key is set up
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def execute_python_code(code: str) -> str:
    """Tool allowing Claude to execute code and manipulate AnnData locally."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=300
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Execution failed: {str(e)}"


tools = [
    {
        "name": "execute_python",
        "description": "Executes arbitrary Python code in the local environment. Use this to read files, run scanpy/anndata pipelines, and save outputs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code snippet to run sequentially."
                }
            },
            "required": ["code"]
        }
    }
]

input_h5ad_path = "./data/input_dataset.h5ad"
output_h5ad_path = "./output/processed_dataset.h5ad"

prompt = f"""You are an expert single-cell bioinformatics agent executing tasks natively via python tools.
Your task is to orchestrate a complete scanpy-based single-cell analysis pipeline on a local `.h5ad` file.

You must run this entire workflow by writing and executing Python scripts through your `execute_python` tool.

Input File Location: `{input_h5ad_path}`
Output Target Location: `{output_h5ad_path}`

Execute the following sequential operations:
1. **Read File**: Load the `.h5ad` file from the specified path using `scanpy.read_h5ad`.
2. **Quality Control & Filtering**: 
   - Calculate QC metrics (mitochondrial gene percentages, ribo counts).
   - Apply filtering thresholds (e.g., minimum cells per gene, minimum/maximum counts or genes per cell).
3. **Normalization & Log Transformation**: Perform total-count normalization to 10,000 reads per cell and target-log transform (`sc.pp.log1p`).
4. **Feature Selection & Scaling**: Identify highly variable genes (HVGs) and scale data to unit variance, clipping extreme values.
5. **Dimension Reduction**: Compute Principal Component Analysis (PCA) and establish the neighborhood graph (`sc.pp.neighbors`).
6. **Clustering**: Cluster cells using the Leiden algorithm (`sc.tl.leiden`).
7. **Write Dataset**: Save the final modified AnnData object into the designated output folder path.

Begin by reading the file and inspecting its shape, then safely advance through the pipeline step-by-step. Verify your output file is successfully written before concluding.
"""

print("Initializing agent workflow with claude-sonnet-5...")

# 1. Initialize a persistent conversation history list
conversation_history = [{"role": "user", "content": prompt}]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=4096,
    tools=tools,
    messages=conversation_history
)

# 2. Correctly append the model's first structural response
conversation_history.append({"role": "assistant", "content": response.content})

while response.stop_reason == "tool_use":
    tool_use = response.content[-1]
    if tool_use.type == "tool_use" and tool_use.name == "execute_python":
        code_to_run = tool_use.input["code"]
        print(f"\n--- [Claude Execution Request] ---\n{code_to_run}\n")

        tool_output = execute_python_code(code_to_run)

        # 3. Append the execution results into the running history array
        conversation_history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": tool_output
                }
            ]
        })

        # 4. Request the next step from Claude with the cumulative history context preserved
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4096,
            tools=tools,
            messages=conversation_history
        )

        # 5. Track the assistant's new response state
        conversation_history.append({"role": "assistant", "content": response.content})

print("\n--- [Pipeline Finished] ---")
# Safely pull the text block out of the final message payload
final_text = "".join([block.text for block in response.content if hasattr(block, 'text')])
print(final_text)
