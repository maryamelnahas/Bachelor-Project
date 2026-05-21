import os
import json
import pandas as pd

print("Initializing Cluster Sampling Script...")

# --- 1. Define Paths ---
base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
input_csv_path = os.path.join(base_dir, "error_clusters_mapped.csv")
global_vocab_path = os.path.join(base_dir, "global_subtree_vocabulary.json")
output_csv_path = os.path.join(base_dir, "cluster_samples_manual_review.csv")

# --- 2. Load the Vocabulary and Dataset ---
print("Loading vocabulary and dataset...")
with open(global_vocab_path, 'r', encoding='utf-8') as f:
    global_vocab = json.load(f)

# Invert dictionary: {Integer_ID: "AST String"}
id_to_ast = {int(v): k for k, v in global_vocab.items()}

df = pd.read_csv(input_csv_path)

# Sanitize column headers to remove any hidden leading/trailing spaces
df.columns = df.columns.str.strip()

# --- 3. Map the Subtree Column ---
df['Subtree'] = df['Global_Subtree_ID'].map(id_to_ast).fillna("<UNKNOWN_AST>")

# --- 4. Filter and Sample ---
print("Sampling 5 representative subtrees from each valid cluster...")

# Exclude noise (-1) as manual review targets cohesive misconceptions
valid_clusters_df = df[df['Cluster_Label'] != -1]

# Universal sampling approach to prevent Pandas index-dropping quirks
sampled_list = []
for cluster_id, group in valid_clusters_df.groupby('Cluster_Label'):
    sampled_group = group.sample(n=min(len(group), 5), random_state=42)
    sampled_list.append(sampled_group)

if sampled_list:
    sampled_df = pd.concat(sampled_list, ignore_index=True)
else:
    print("Error: No valid clusters found to sample.")
    exit()

# --- 5. Format and Sort Output ---
# Define the requested column arrangement
columns_to_keep = [
    'Global_Subtree_ID', 
    'Submission_ID', 
    'Raw_Code_Snippet', 
    'Subtree', 
    'Attention_Weight', 
    'Cluster_Label'
]

# Enforce column selection
final_columns = [col for col in columns_to_keep if col in sampled_df.columns]
sampled_df = sampled_df[final_columns]

# Sort sequentially by Cluster_Label, then by Attention_Weight (highest first)
sampled_df = sampled_df.sort_values(by=['Cluster_Label', 'Attention_Weight'], ascending=[True, False])

# --- 6. Save Results ---
sampled_df.to_csv(output_csv_path, index=False)

print(f"\nSampling Complete.")
print(f"Extracted {len(sampled_df)} total subtrees representing {sampled_df['Cluster_Label'].nunique()} distinct clusters.")
print(f"File saved ready for manual review at:\n{output_csv_path}")