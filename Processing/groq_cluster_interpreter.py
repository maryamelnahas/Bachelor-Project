import os
import json
import time
import pandas as pd
from groq import Groq
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

CSV_PATH = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\clusters_sample_for_groq.csv"
OUTPUT_DIR = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\groq_cluster_analysis"

MODEL_NAME = "llama-3.3-70b-versatile"

# Optional truncation
MAX_CODE_CHARS = 2500
MAX_SUBTREE_CHARS = 1200
MAX_DESCRIPTION_CHARS = 1500

# =========================
# INITIALIZE
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = Groq(
    api_key=""
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_PATH)

# =========================
# HELPER FUNCTIONS
# =========================

def truncate(text, limit):
    if pd.isna(text):
        return ""
    text = str(text)
    return text[:limit]


def build_cluster_text(cluster_df):
    """
    Convert cluster samples into structured prompt text.
    """

    entries = []

    for idx, row in cluster_df.iterrows():

        entry = f"""
SAMPLE

Problem Description:
{truncate(row['Problem Description'], MAX_DESCRIPTION_CHARS)}

Submission Status:
{row['Status']}

Attention Weight:
{row['Attention Weight']}

Raw Code Snippet:
{truncate(row['Raw Code Snippet'], MAX_CODE_CHARS)}

Flagged Subtree:
{truncate(row['Subtree'], MAX_SUBTREE_CHARS)}
"""

        entries.append(entry)

    return "\n\n".join(entries)


def call_llm(system_prompt, user_prompt, max_retries=5):
    """
    Calls the Groq API with error handling and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"\n[Warning] API call failed (Attempt {attempt + 1} of {max_retries}): {e}")
            if attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16 seconds
                sleep_time = 2 ** (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"[Error] Max retries reached for prompt. Skipping.")
                return "ERROR: API call failed after multiple attempts."


# =========================
# SYSTEM PROMPTS
# =========================

INDIVIDUAL_ANALYSIS_SYSTEM = """
You are analyzing suspicious Java AST subtrees extracted from incorrect student programming submissions.

The subtrees were identified using attention scores from a neural network model trained for programming error detection.

Your task is NOT to assume the subtree is definitely the cause of failure.

For EACH sample:
1. Identify what the subtree represents in the code.
2. Explain what part of the code it corresponds to.
3. Assess whether it plausibly contributed to the failure.
4. Explain why or why not.
5. Mention your confidence level (High, Medium, Low).

Be cautious and avoid hallucinating unsupported claims.
"""

CLUSTER_ANALYSIS_SYSTEM = """
You are analyzing a cluster of suspicious AST subtrees extracted from student code.

Your goal is to explain why these subtrees may have been grouped together.

Focus on:
1. Structural similarities.
2. Semantic similarities.
3. Shared programming patterns.
4. Shared logical behaviors.
5. Whether the grouping appears meaningful.

Avoid overclaiming.
"""

MISCONCEPTION_ANALYSIS_SYSTEM = """
You are analyzing clustered programming error patterns.

Your task is to infer whether the cluster may reflect shared misconceptions or conceptual difficulties.

Focus on:
1. Possible misconceptions.
2. Why novice programmers may make these mistakes.
3. Whether the misconception hypothesis is strong or weak.
4. Confidence level.

Do NOT claim certainty.
"""

# =========================
# PROCESS CLUSTERS
# =========================

cluster_labels = sorted(df["Cluster Label"].unique())

for cluster_id in tqdm(cluster_labels):

    cluster_df = df[df["Cluster Label"] == cluster_id]

    cluster_text = build_cluster_text(cluster_df)

    print(f"\nProcessing Cluster {cluster_id}")

    # ====================================
    # STEP 1 — INDIVIDUAL ANALYSIS
    # ====================================

    individual_prompt = f"""
Analyze the following cluster samples individually.

{cluster_text}
"""

    individual_response = call_llm(
        INDIVIDUAL_ANALYSIS_SYSTEM,
        individual_prompt
    )

    time.sleep(2)

    # ====================================
    # STEP 2 — CLUSTER ANALYSIS
    # ====================================

    cluster_prompt = f"""
Below are samples belonging to the same cluster.

SAMPLES:
{cluster_text}

PREVIOUS INDIVIDUAL ANALYSIS:
{individual_response}

Analyze the cluster as a whole.
"""

    cluster_response = call_llm(
        CLUSTER_ANALYSIS_SYSTEM,
        cluster_prompt
    )

    time.sleep(2)

    # ====================================
    # STEP 3 — MISCONCEPTION ANALYSIS
    # ====================================

    misconception_prompt = f"""
Below are clustered subtree samples and prior analyses.

SAMPLES:
{cluster_text}

INDIVIDUAL ANALYSIS:
{individual_response}

CLUSTER ANALYSIS:
{cluster_response}

Infer whether this cluster may reflect shared misconceptions or conceptual difficulties.
"""

    misconception_response = call_llm(
        MISCONCEPTION_ANALYSIS_SYSTEM,
        misconception_prompt
    )

    # ====================================
    # SAVE RESULTS
    # ====================================

    output = {
        "cluster_id": int(cluster_id),
        "individual_analysis": individual_response,
        "cluster_analysis": cluster_response,
        "misconception_analysis": misconception_response
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        f"cluster_{cluster_id}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved: {output_path}")

    time.sleep(2)

print("\nAll clusters processed.")