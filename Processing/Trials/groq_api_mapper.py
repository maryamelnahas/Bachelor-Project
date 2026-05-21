import pandas as pd
from groq import Groq
import time
import json

print("Initializing Open-Ended Groq API Pipeline...")

# --- 1. Configure the API ---
GROQ_API_KEY = "xx"
client = Groq(api_key=GROQ_API_KEY)

MODEL_ID = 'llama-3.3-70b-versatile'

# --- 2. File Paths ---
mapping_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\error_clusters_mapped.csv"
original_dataset_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Programming Mistakes Dataset - Java P0 Solutions.csv"
output_csv_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\automated_cluster_labels.csv"

# --- 3. Load Data ---
df_mapped = pd.read_csv(mapping_path)
df_original = pd.read_csv(original_dataset_path)
code_lookup = dict(zip(df_original['File Name'], df_original['Code Snippet']))

# --- 4. The Main API Loop ---
clusters = df_mapped['Cluster_Label'].unique()
clusters = [c for c in clusters if c != -1] # Remove noise
clusters.sort()

print(f"Discovered {len(clusters)} valid clusters to process. Beginning API calls...\n")
results_list = []

for cluster_id in clusters:
    print(f"Processing Cluster {cluster_id}...")
    cluster_data = df_mapped[df_mapped['Cluster_Label'] == cluster_id]
    
    # Sample up to 10 snippets to give the LLM enough context
    sample_size = min(10, len(cluster_data)) 
    sampled_errors = cluster_data.sample(n=sample_size, random_state=42)

    # The New Open-Ended Prompt
    prompt = f"""Act as an expert Computer Science professor evaluating novice Java code.
Problem Description: [Write a program which prints multiplication tables in the following format:

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

Analyze the following {sample_size} failing submissions. They were mathematically grouped together because their Abstract Syntax Trees share a highly specific structural/logical error. 

Your task is to independently identify what that shared cognitive programming flaw is. Keep it extremely concise.

Provide your response STRICTLY as a raw JSON object with no markdown formatting. Use these exact two keys:
"Label": (A concise, 2-4 word pedagogical name for this specific misconception)
"Reasoning": (A short, 1-2 sentence explanation of why this label applies based on the code provided)

STUDENT CODE SAMPLES:
"""
    for _, row in sampled_errors.iterrows():
        fname = row['File Name']
        prompt += f"\n--- Subtree ID: {row['Subtree_ID']} ---\n{code_lookup.get(fname, '')}\n"

    # Send to Groq
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_ID,
            response_format={"type": "json_object"}, # Forces perfect JSON output
            temperature=0.2 # Low temperature keeps the labels professional and consistent
        )
        
        response_text = response.choices[0].message.content.strip()
        result_dict = json.loads(response_text)
        
        # Add the Cluster ID to match your requested 3 columns
        result_dict['Cluster ID'] = cluster_id
        results_list.append(result_dict)
        print(f"  -> Success: Labeled as '{result_dict.get('Label')}'")
        
    except Exception as e:
        print(f"  -> API Error on Cluster {cluster_id}: {e}")
        results_list.append({
            'Cluster ID': cluster_id, 'Label': 'ERROR', 'Reasoning': str(e)
        })

    # Small pause to respect API rate limits
    time.sleep(2) 

# --- 5. Save the Final Results ---
print("\nAPI processing complete. Saving to CSV...")
df_final = pd.DataFrame(results_list)

# Reorder columns exactly as you requested
cols = ['Cluster ID', 'Label', 'Reasoning']
df_final = df_final[cols]
df_final.to_csv(output_csv_path, index=False)

print(f"Pipeline complete! Final clean labels saved to: {output_csv_path}")