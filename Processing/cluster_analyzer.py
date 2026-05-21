import os
import json
import pandas as pd

print("Initializing Analytics Pipeline...")

# --- 1. Define Paths ---
base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
clusters_csv_path = os.path.join(base_dir, "error_clusters_mapped.csv")
global_vocab_path = os.path.join(base_dir, "global_subtree_vocabulary.json")
output_csv_path = os.path.join(base_dir, "error_clusters_mapped_with_ast.csv")

# --- 2. Load and Invert Vocabulary ---
with open(global_vocab_path, 'r', encoding='utf-8') as f:
    global_vocab = json.load(f)

# Invert dictionary: {Integer_ID: "AST String"}
id_to_ast = {int(v): k for k, v in global_vocab.items()}

# --- 3. Update Dataset ---
# Read the clustered dataset
df = pd.read_csv(clusters_csv_path)

# Add the new column by mapping the IDs to the AST strings
df['AST_Structure'] = df['Global_Subtree_ID'].map(id_to_ast).fillna("<UNKNOWN_AST>")

# Rearrange columns so AST_Structure is easily visible alongside the Raw Code
cols = df.columns.tolist()
# Assuming the default columns are: Submission_ID, Global_Subtree_ID, Raw_Code_Snippet, Attention_Weight, Cluster_Label
new_order = ['Submission_ID', 'Global_Subtree_ID', 'AST_Structure', 'Raw_Code_Snippet', 'Attention_Weight', 'Cluster_Label']
# Ensure columns exist to avoid KeyErrors, then reorder
df = df[[c for c in new_order if c in cols] + [c for c in cols if c not in new_order]]

# Save the updated dataset
df.to_csv(output_csv_path, index=False)
print(f"Successfully added 'AST_Structure' column. File saved to:\n{output_csv_path}\n")

# --- 4. Generate Analytics ---
print("================ Dataset Analytics ================")

# Total Rows
total_entries = len(df)
print(f"Total Entries (Isolated Error Subtrees): {total_entries}")

# Distinct Submissions
distinct_submissions = df['Submission_ID'].nunique()
print(f"Distinct Submissions (Unique Students):  {distinct_submissions}")

# Distinct Clusters
# -1 is typically used by HDBSCAN to denote isolated noise, so it is counted separately
valid_clusters_mask = df['Cluster_Label'] != -1
distinct_clusters = df[valid_clusters_mask]['Cluster_Label'].nunique()
noise_entries = len(df[~valid_clusters_mask])

print(f"Distinct Valid Clusters Identified:      {distinct_clusters}")
print(f"Entries Filtered as Random Noise (-1):   {noise_entries}")
print("===================================================\n")

# --- 5. Extract Top 20 Largest Clusters ---
print("--- Top 20 Largest Misconception Clusters ---")
print(f"{'Cluster ID':<15} {'Frequency (Subtree Count)':<10}")
print("-" * 45)

# Calculate frequencies excluding noise
top_20_clusters = df[valid_clusters_mask]['Cluster_Label'].value_counts().head(20)

for cluster_id, count in top_20_clusters.items():
    print(f"Cluster {cluster_id:<11} {count}")