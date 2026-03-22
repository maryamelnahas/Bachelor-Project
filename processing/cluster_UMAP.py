import torch
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json

# Import your model and data loader
from SANN_model import ModifiedSANN
from data_loader import SubtreeDatasetWithLabels

print("Initializing Extraction, UMAP, and DBSCAN Pipeline...")

# --- 1. Setup and Load Data ---
vocab_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\node_vocabulary.json"
tokenized_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\extracted_subtrees_tokenized.csv"
metadata_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\P0_metadata.csv" 

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

dataset = SubtreeDatasetWithLabels(tokenized_path, metadata_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
model = ModifiedSANN(vocab_size=len(vocab), num_unique_subtrees=50000)

print("Loading trained weights...")
weight_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\sann_model_weights.pth"
model.load_state_dict(torch.load(weight_path, weights_only=True))

# --- 2. Extracting High-Attention Error Vectors & Metadata ---
print("Scanning submissions for high-attention logical errors...")
error_vectors = []
error_metadata = [] # We must track the origin of each vector for the LLM step

model.eval() 
with torch.no_grad():
    for node_seq, sub_id, label, fname in dataloader:
        if label.item() == 1.0: # Skip correct submissions
            continue
            
        prediction, attention_weights, subtree_vectors = model(node_seq, sub_id)
        
        weights_flat = attention_weights.squeeze().numpy()
        vectors_flat = subtree_vectors.squeeze().numpy()
        sub_id_flat = sub_id.squeeze().numpy()
        
        if weights_flat.ndim == 0:
            weights_flat = np.expand_dims(weights_flat, axis=0)
            vectors_flat = np.expand_dims(vectors_flat, axis=0)
            sub_id_flat = np.expand_dims(sub_id_flat, axis=0)

        for i, weight in enumerate(weights_flat):
            if weight > 0.5:
                error_vectors.append(vectors_flat[i])
                # Save the exact file and subtree ID so we can look up the code later
                error_metadata.append({
                    'File Name': fname[0], 
                    'Subtree_ID': f"subtree_{sub_id_flat[i]}"
                })

print(f"Extraction complete! Isolated {len(error_vectors)} errors.")

# --- 3. UMAP Dimensionality Reduction ---
if len(error_vectors) > 5:
    print("Compressing 64D error vectors to 2D using UMAP...")
    error_matrix = np.array(error_vectors)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(error_matrix)
    
    # --- 4. DBSCAN Clustering ---
    print("Grouping errors into misconception clusters using DBSCAN...")
    # eps defines how close points must be to group together. 
    # min_samples is how many points are needed to officially form a cluster.
    dbscan = DBSCAN(eps=0.5, min_samples=15)
    cluster_labels = dbscan.fit_predict(embedding_2d)
    
    # Count the results
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    print(f"DBSCAN found {num_clusters} distinct misconception clusters.")
    print(f"DBSCAN filtered out {num_noise} random errors as noise.")

    # --- 5. Save the LLM Mapping CSV ---
    df_results = pd.DataFrame(error_metadata)
    df_results['Cluster_Label'] = cluster_labels
    mapping_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\error_clusters_mapped.csv"
    df_results.to_csv(mapping_path, index=False)
    print(f"Saved cluster mapping for the LLM to: {mapping_path}")

    # --- 6. Plotting the Results ---
    print("Generating color-coded scatter plot...")
    plt.figure(figsize=(10, 8))
    
    # Plot Noise (-1) in light grey
    noise_mask = cluster_labels == -1
    plt.scatter(embedding_2d[noise_mask, 0], embedding_2d[noise_mask, 1], 
                s=10, c='lightgrey', alpha=0.5, label='Noise (Isolated Errors)')
    
    # Plot valid clusters in distinct colors
    cluster_mask = cluster_labels != -1
    scatter = plt.scatter(embedding_2d[cluster_mask, 0], embedding_2d[cluster_mask, 1], 
                s=15, c=cluster_labels[cluster_mask], cmap='tab10', alpha=0.8)
    
    plt.title('DBSCAN Clusters of Student Misconceptions', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(scatter, label='Misconception Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('dbscan_misconception_clusters.png', dpi=300)
    print("Plot saved as 'dbscan_misconception_clusters.png'.")
    plt.show()

else:
    print("Not enough high-attention vectors found.")