import pandas as pd

# --- 1. File Paths ---
cluster_csv_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\error_clusters_mapped.csv"
original_submissions_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Programming Mistakes Dataset - Java P0 Solutions.csv"
subtree_csv_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\extracted_subtrees.csv" 
output_csv_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\manual_review_guide.csv"

# --- 2. Load the Datasets ---
df_clusters = pd.read_csv(cluster_csv_path)
df_original = pd.read_csv(original_submissions_path)
df_subtrees = pd.read_csv(subtree_csv_path)

code_column = 'Code Snippet'
subtree_text_column = 'Subtree_Expression' 

# --- 3. Merge the Data ---
# Link the Cluster ID to the raw Java code based on the shared 'File Name'
merged_df = pd.merge(df_clusters, df_original[['File Name', code_column]], on='File Name', how='left')

# Link the resulting dataframe to the subtree text based on both 'File Name' and 'Subtree_ID'
merged_df = pd.merge(merged_df, df_subtrees[['File Name', 'Subtree_ID', subtree_text_column]], on=['File Name', 'Subtree_ID'], how='left')

# --- 4. Filter, Sample, and Sort ---
# Exclude the -1 noise cluster
filtered_df = merged_df[merged_df['Cluster_Label'] != -1]

# Shuffle to ensure random representation
shuffled_df = filtered_df.sample(frac=1, random_state=42)

# Extract a maximum of 5 representative samples per cluster
sampled_df = shuffled_df.groupby('Cluster_Label').head(5)

# Sort sequentially by Cluster_Label for organized manual review
sampled_df = sampled_df.sort_values(by=['Cluster_Label'])

# --- 5. Format and Export ---
final_columns = ['Cluster_Label', 'File Name', 'Subtree_ID', subtree_text_column, code_column]
final_df = sampled_df[final_columns]

final_df.to_csv(output_csv_path, index=False)

print(f"Sampling complete. CSV successfully saved to: {output_csv_path}")