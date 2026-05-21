import os
import json
import torch
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import model, dataset, and collate function
from SANN_model import ModifiedSANN, SubtreeDatasetWithLabels, pad_collate

print("Initializing Extraction, UMAP, and HDBSCAN Pipeline...")

# --- 1. Setup and Load Data ---
base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
vocab_path = os.path.join(base_dir, "node_vocabulary.json")
global_vocab_path = os.path.join(base_dir, "global_subtree_vocabulary.json")
tensor_dir = os.path.join(base_dir, "submission_tensors")
metadata_path = os.path.join(base_dir, "submissions_metadata_labels_cleaned.csv") 
weight_path = os.path.join(base_dir, "sann_model_weights.pth")

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
    
with open(global_vocab_path, 'r', encoding='utf-8') as f:
    global_vocab = json.load(f)

dataset = SubtreeDatasetWithLabels(tensor_dir, metadata_path)
# Increased batch size to 32 for significantly faster extraction
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pad_collate) 

submission_to_code = dict(zip(dataset.metadata['Submission ID'].astype(str), dataset.metadata['Code Snippet']))

model = ModifiedSANN(vocab_size=len(vocab), num_unique_subtrees=len(global_vocab) + 1)

print("Loading trained weights...")
model.load_state_dict(torch.load(weight_path, weights_only=True))
model.eval()

# --- 2. Intercept Intermediate Subtree Vectors ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.fc_combine.register_forward_hook(get_activation('fc_combine'))

# --- 3. Extracting High-Attention Error Vectors & Metadata ---
print("Scanning submissions for high-attention logical errors...")
error_vectors = []
error_metadata = [] 

with torch.no_grad():
    for node_seq, sub_id, label, fnames in dataloader:
        # Skip batch entirely if all are correct (Label 1)
        if torch.all(label == 1.0):
            continue
            
        prediction, attention_weights, _ = model(node_seq, sub_id)
        subtree_vectors = torch.relu(activation['fc_combine'])
        
        weights_array = attention_weights.cpu().numpy()
        vectors_array = subtree_vectors.cpu().numpy()
        ids_array = sub_id.cpu().numpy()
        labels_array = label.cpu().numpy()

        for b in range(len(fnames)):
            if labels_array[b] == 1.0:
                continue
                
            submission_id_str = str(fnames[b])
            
            for i, weight in enumerate(weights_array[b]):
                if weight > 0.5 and ids_array[b, i] != 0:
                    global_id_val = int(ids_array[b, i])
                    raw_code = submission_to_code.get(submission_id_str, "<CODE NOT FOUND>")
                    
                    error_vectors.append(vectors_array[b, i])
                    error_metadata.append({
                        'Submission_ID': submission_id_str, 
                        'Global_Subtree_ID': global_id_val,
                        'Raw_Code_Snippet': raw_code,
                        'Attention_Weight': float(weight)
                    })

print(f"Extraction complete! Isolated {len(error_vectors)} targeted errors.")

# --- 4. UMAP Dimensionality Reduction ---
if len(error_vectors) > 5:
    print("Compressing 64D error vectors to 2D using UMAP...")
    error_matrix = np.array(error_vectors)
    
    # Removed random_state=42 and added init='pca' to enable multi-threading and bypass spectral bottlenecks
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.15, n_components=2, init='pca')
    embedding_2d = reducer.fit_transform(error_matrix)
    
    # --- 4b. UMAP Performance Evaluation (Memory-Efficient Sub-sampling) ---
    print("\n--- UMAP Performance Metrics (Sub-sampled) ---")
    
    # Isolate a statistically significant random sample to prevent O(N^2) RAM failure
    sample_size = min(len(error_matrix), 5000)
    np.random.seed(42)
    sample_indices = np.random.choice(len(error_matrix), sample_size, replace=False)
    
    high_dim_sample = error_matrix[sample_indices]
    low_dim_sample = embedding_2d[sample_indices]
    
    k_neighbors = 15
    
    # 1. Trustworthiness
    tw_score = trustworthiness(high_dim_sample, low_dim_sample, n_neighbors=k_neighbors)
    print(f"Trustworthiness:        {tw_score:.4f}")
    
    # 2. Continuity
    cont_score = trustworthiness(low_dim_sample, high_dim_sample, n_neighbors=k_neighbors)
    print(f"Continuity:             {cont_score:.4f}")
    
    # 3. Neighborhood Overlap
    nn_high = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(high_dim_sample)
    nn_low = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(low_dim_sample)
    
    _, indices_high = nn_high.kneighbors(high_dim_sample)
    _, indices_low = nn_low.kneighbors(low_dim_sample)
    
    overlaps = []
    for i in range(sample_size):
        set_high = set(indices_high[i]) - {i}
        set_low = set(indices_low[i]) - {i}
        overlap = len(set_high.intersection(set_low)) / k_neighbors
        overlaps.append(overlap)
    
    mean_overlap = np.mean(overlaps)
    print(f"Neighborhood Overlap:   {mean_overlap:.4f}")
    
    # 4. Spearman Correlation of Pairwise Distances
    # Reduced slightly further as pdist is highly memory intensive even at 5000
    corr_sample_size = min(len(high_dim_sample), 2500)
    dist_high = pdist(high_dim_sample[:corr_sample_size])
    dist_low = pdist(low_dim_sample[:corr_sample_size])
    
    spearman_corr, _ = spearmanr(dist_high, dist_low)
    print(f"Spearman Correlation:   {spearman_corr:.4f}")

    # --- 5. HDBSCAN Clustering ---
    print("\nGrouping errors into misconception clusters using HDBSCAN...")
    hdbscan_model = HDBSCAN(min_cluster_size=1000, min_samples=50, cluster_selection_epsilon=0.2)
    cluster_labels = hdbscan_model.fit_predict(embedding_2d)
    
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    print(f"HDBSCAN found {num_clusters} broad misconception clusters.")
    print(f"HDBSCAN filtered out {num_noise} random errors as noise.")

    # --- 6. Internal Metrics Assessment ---
    print("\n--- Clustering Performance Metrics ---")
    cluster_mask = cluster_labels != -1
    if num_clusters > 1:
        valid_embeddings = embedding_2d[cluster_mask]
        valid_labels = cluster_labels[cluster_mask]
        
        sil_score = silhouette_score(valid_embeddings, valid_labels)
        db_score = davies_bouldin_score(valid_embeddings, valid_labels)
        
        print(f"Silhouette Coefficient: {sil_score:.4f}")
        print(f"Davies-Bouldin Index:   {db_score:.4f}")
    else:
        print("Not enough distinct clusters formed to calculate internal metrics.")

    # --- 7. Save the LLM Mapping CSV ---
    df_results = pd.DataFrame(error_metadata)
    df_results['Cluster_Label'] = cluster_labels
    mapping_path = os.path.join(base_dir, "error_clusters_mapped.csv")
    df_results.to_csv(mapping_path, index=False)
    print(f"\nSaved cluster mapping for the LLM to: {mapping_path}")

    # --- 8. Plotting the Results ---
    print("Generating color-coded scatter plot...")
    plt.figure(figsize=(10, 8))
    
    noise_mask = cluster_labels == -1
    plt.scatter(embedding_2d[noise_mask, 0], embedding_2d[noise_mask, 1], 
                s=10, c='lightgrey', alpha=0.5, label='Noise (Isolated Errors)')
    
    scatter = plt.scatter(embedding_2d[cluster_mask, 0], embedding_2d[cluster_mask, 1], 
                s=15, c=cluster_labels[cluster_mask], cmap='tab10', alpha=0.8)
    
    plt.title('HDBSCAN Clusters of Student Misconceptions', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(scatter, label='Misconception Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('hdbscan_misconception_clusters.png', dpi=300)
    print("Plot saved as 'hdbscan_misconception_clusters.png'.")
    plt.show()

else:
    print("Not enough high-attention vectors found.")