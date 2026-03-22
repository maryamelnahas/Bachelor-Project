import pandas as pd
from google import genai
from google.genai import types
import time
import json

print("Initializing Automated Gemini API Pipeline...")

# --- 1. Configure the API ---
GOOGLE_API_KEY = "x"
client = genai.Client(api_key=GOOGLE_API_KEY)

# We use the new Gemini 3 Flash model which replaced the 1.5 versions
MODEL_ID = 'gemini-3-flash-preview'

# --- 2. File Paths ---
mapping_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\error_clusters_mapped.csv"
taxonomy_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Misconceptions Categorization.csv"
original_dataset_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\Programming Mistakes Dataset - Java P0 Solutions.csv"
output_csv_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\automated_misconceptions.csv"

# --- 3. Load Data & Format Taxonomy ---
df_mapped = pd.read_csv(mapping_path)
df_original = pd.read_csv(original_dataset_path)
df_taxonomy = pd.read_csv(taxonomy_path)

code_lookup = dict(zip(df_original['File Name'], df_original['Code Snippet']))

taxonomy_text = ""
for topic, group in df_taxonomy.groupby('Topic'):
    taxonomy_text += f"\n### {topic} ###\n"
    for _, row in group.iterrows():
        taxonomy_text += f" - [ID: {row['No.']}] {row['Description']}\n"

# --- 4. The Main API Loop ---
clusters = df_mapped['Cluster_Label'].unique()
# Remove noise (-1) from the clusters to analyze
clusters = [c for c in clusters if c != -1]
clusters.sort()

print(f"Discovered {len(clusters)} valid clusters to process. Beginning API calls...\n")

results_list = []

for cluster_id in clusters:
    print(f"Processing Cluster {cluster_id}...")
    cluster_data = df_mapped[df_mapped['Cluster_Label'] == cluster_id]
    
    # --- SAMPLING LOGIC ---
    sample_size = min(10, len(cluster_data)) 
    sampled_errors = cluster_data.sample(n=sample_size, random_state=42)

    prompt = f"""Act as an expert university Computer Science teaching assistant evaluating Java.
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

Analyze these {sample_size} failing submissions. They share a structural syntax/logic error in their AST.
Map the root cognitive flaw to the taxonomy below.

TAXONOMY:
{taxonomy_text}

Provide your response STRICTLY as a raw JSON object with no markdown formatting or backticks. Use these exact keys:
"Broad_Category": (The topic name from the taxonomy)
"Specific_Misconception_ID": (The exact ID number, or "NOVEL" if none fit)
"Cognitive_Flaw": (2-3 sentences explaining the fundamental misunderstanding)
"Pedagogical_Fix": (How to explain the correction to a student)

STUDENT CODE SAMPLES:
"""
    for _, row in sampled_errors.iterrows():
        fname = row['File Name']
        prompt += f"\n--- Subtree ID: {row['Subtree_ID']} ---\n{code_lookup.get(fname, '')}\n"

    # Send to Gemini
    try:
        # The new SDK uses client.models.generate_content and handles structured output via config
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        response_text = response.text.strip()
        
        # Clean up any accidental markdown backticks the LLM might include
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
            
        # Parse the JSON string into a Python dictionary
        result_dict = json.loads(response_text)
        result_dict['Cluster_ID'] = cluster_id
        results_list.append(result_dict)
        print(f"  -> Success: Mapped to ID {result_dict.get('Specific_Misconception_ID')}")
        
    except Exception as e:
        print(f"  -> API Error on Cluster {cluster_id}: {e}")
        # If the LLM output wasn't perfect JSON, we record a failure so the script doesn't crash
        results_list.append({
            'Cluster_ID': cluster_id, 'Broad_Category': 'ERROR', 
            'Specific_Misconception_ID': 'ERROR', 'Cognitive_Flaw': str(e), 'Pedagogical_Fix': ''
        })

    # Rate Limiting: Pause for 5 seconds between calls to respect the free tier limits
    time.sleep(5) 

# --- 5. Save the Final Results ---
print("\nAPI processing complete. Saving to CSV...")
df_final = pd.DataFrame(results_list)
# Reorder columns to make it readable
cols = ['Cluster_ID', 'Broad_Category', 'Specific_Misconception_ID', 'Cognitive_Flaw', 'Pedagogical_Fix']
df_final = df_final[cols]
df_final.to_csv(output_csv_path, index=False)
print(f"Pipeline complete! Final mappings saved to: {output_csv_path}")