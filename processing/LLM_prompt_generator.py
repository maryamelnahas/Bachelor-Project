import pandas as pd

print("Initializing LLM Prompt Generator with Taxonomy Integration...")

# --- 1. File Paths ---
mapping_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\error_clusters_mapped.csv"
taxonomy_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Misconceptions Categorization.csv"
original_dataset_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Programming Mistakes Dataset - Java P0 Solutions.csv"

# --- 2. Load the Data ---
df_mapped = pd.read_csv(mapping_path)
df_original = pd.read_csv(original_dataset_path)
df_taxonomy = pd.read_csv(taxonomy_path)

# Dictionary to look up a student's code by their File Name
code_column_name = 'Code Snippet' 
code_lookup = dict(zip(df_original['File Name'], df_original[code_column_name]))

# --- 3. Format the Taxonomy for the LLM ---
print("Formatting the 163 literature misconceptions...")
taxonomy_text = ""
# Grouping the specific misconceptions (Description) under their broad categories (Topic)
grouped_taxonomy = df_taxonomy.groupby('Topic')

for topic, group in grouped_taxonomy:
    taxonomy_text += f"\n### Category: {topic} ###\n"
    for _, row in group.iterrows():
        # Including the 'No.' as a strict ID helps the LLM reference it precisely
        taxonomy_text += f" - [ID: {row['No.']}] {row['Description']}\n"

# --- 4. Isolate Cluster 0 ---
TARGET_CLUSTER = 0
cluster_data = df_mapped[df_mapped['Cluster_Label'] == TARGET_CLUSTER]
print(f"Cluster {TARGET_CLUSTER} contains {len(cluster_data)} student submissions.")

sample_size = min(3, len(cluster_data))
sampled_errors = cluster_data.sample(n=sample_size, random_state=42)

# --- 5. Build the Dynamic Prompt ---
prompt = f"""Act as an expert university Computer Science teaching assistant. 

My students were assigned a Java programming task. 
[Write a program which prints multiplication tables in the following format:

1x1=1
1x2=2
.
.
9x8=72
9x9=81
Input
No input.

Output
1x1=1
1x2=2
.
.
9x8=72
9x9=81]

I have an AI model that grouped several failing student submissions into a single cluster based on structural similarities in their Abstract Syntax Trees. The model flagged the specific recursive subtree that it believes is the root cause of the error.

Your task is to analyze these {sample_size} submissions from the same cluster, focus on the flagged subtrees, and map their shared cognitive flaw to the established taxonomy of programming misconceptions provided below.

=========================================
THE LITERATURE TAXONOMY:
{taxonomy_text}
=========================================

INSTRUCTIONS FOR YOUR OUTPUT:
1. Broad Category: (Identify which of the broad 'Topics' this error belongs to).
2. Specific Misconception: (Select the exact 'Description' and its [ID] from the taxonomy list. If it absolutely does not fit any of them, output "NOVEL_DISCOVERY" and invent a highly accurate pedagogical name).
3. Cognitive Flaw: (A 2-3 sentence explanation of what the students in this cluster are fundamentally misunderstanding).
4. Pedagogical Fix: (How you would explain this concept to the students to correct the misconception).

Here is the source code data for Cluster {TARGET_CLUSTER}:
--------------------------------------------------
"""

# Inject the student code samples
for index, row in sampled_errors.iterrows():
    fname = row['File Name']
    flagged_subtree = row['Subtree_ID']
    student_code = code_lookup.get(fname, "// Code not found in original dataset")
    
    prompt += f"--- Submission: {fname} ---\n"
    prompt += f"Model Flagged Subtree ID: {flagged_subtree}\n"
    prompt += f"Source Code:\n{student_code}\n"
    prompt += "--------------------------------------------------\n"

# --- 6. Output the Result ---
output_file = f"llm_prompt_cluster_{TARGET_CLUSTER}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(prompt)

print(f"\nSuccess! Prompt generated and saved to '{output_file}'.")
print("Next steps: Open the text file, replace the placeholder with your P0 problem description, and paste it into the LLM.")