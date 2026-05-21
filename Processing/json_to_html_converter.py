import os
import json
import glob

# Define directories based on prior configuration
INPUT_DIR = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\groq_cluster_analysis"
OUTPUT_PATH = os.path.join(INPUT_DIR, "Compiled_Analysis.html")

html_content = ["<html><body style='font-family: Arial, sans-serif;'>"]
html_content.append("<h1>Cluster Analysis Report</h1>")

# Retrieve and process JSON files
json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    cluster_id = data.get('cluster_id', 'Unknown')
    html_content.append(f"<h2>Cluster {cluster_id}</h2>")
    
    # Replace newline characters with HTML break tags for readability
    html_content.append("<h3>Individual Analysis</h3>")
    html_content.append(f"<p>{data.get('individual_analysis', '').replace(chr(10), '<br>')}</p>")
    
    html_content.append("<h3>Cluster Analysis</h3>")
    html_content.append(f"<p>{data.get('cluster_analysis', '').replace(chr(10), '<br>')}</p>")
    
    html_content.append("<h3>Misconception Analysis</h3>")
    html_content.append(f"<p>{data.get('misconception_analysis', '').replace(chr(10), '<br>')}</p>")
    
    html_content.append("<hr>")

html_content.append("</body></html>")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(html_content))

print(f"Compiled HTML saved to: {OUTPUT_PATH}")