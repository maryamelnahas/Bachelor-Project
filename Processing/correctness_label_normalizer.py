import os
import torch
import pandas as pd
import hashlib
from tqdm import tqdm

def clean_and_report_lean(tensor_dir, metadata_path, cleaned_meta_output, report_output):
    print("Loading metadata...")
    df_meta = pd.read_csv(metadata_path)
    df_meta['Submission ID'] = df_meta['Submission ID'].astype(str)
    
    valid_ids = [f.split('.')[0] for f in os.listdir(tensor_dir) if f.endswith('.pt')]
    df_meta = df_meta[df_meta['Submission ID'].isin(valid_ids)]
    
    hash_to_ids = {}
    
    print("Pass 1: Grouping identical submissions using native Tensors...")
    for sub_id in tqdm(df_meta['Submission ID'], desc="Hashing Tensors"):
        tensor_file = os.path.join(tensor_dir, f"{sub_id}.pt")
        tensor = torch.load(tensor_file, weights_only=True)
        
        # We hash the tensor. This is mathematically identical to comparing the whole tree.
        tensor_bytes = tensor.cpu().numpy().tobytes()
        t_hash = hashlib.md5(tensor_bytes).hexdigest()
        
        if t_hash not in hash_to_ids:
            hash_to_ids[t_hash] = []
        hash_to_ids[t_hash].append(sub_id)

    print("\nPass 2: Identifying conflicts and generating report...")
    conflicting_ids = set()
    hash_to_ast_id = {}
    ast_id_counter = 1
    
    for t_hash, ids in hash_to_ids.items():
        if len(ids) > 1:
            group_labels = df_meta[df_meta['Submission ID'].isin(ids)]['Accuracy'].unique()
            if len(group_labels) > 1:
                hash_to_ast_id[t_hash] = ast_id_counter
                ast_id_counter += 1
                
                for sub_id in ids:
                    conflicting_ids.add(sub_id)
                    # Correct the label directly in the dataframe
                    df_meta.loc[df_meta['Submission ID'] == sub_id, 'Accuracy'] = 1.0

    print(f"Total conflicting AST structures identified: {len(hash_to_ast_id)}")
    
    # Save the cleaned metadata for the SANN Model
    df_meta.to_csv(cleaned_meta_output, index=False)
    print(f"\nCleaned metadata saved to: {cleaned_meta_output}")
    
    if not conflicting_ids:
        print("No conflicts found. Exiting.")
        return

    # Generate the ultra-lean report
    df_report = df_meta[df_meta['Submission ID'].isin(conflicting_ids)].copy()
    
    id_to_ast_id = {}
    for t_hash, ast_id in hash_to_ast_id.items():
        for sub_id in hash_to_ids[t_hash]:
            id_to_ast_id[sub_id] = ast_id
            
    df_report['AST ID'] = df_report['Submission ID'].map(id_to_ast_id)
    
    available_cols = df_report.columns.tolist()
    code_col = 'Code Snippet' if 'Code Snippet' in available_cols else 'Code Snippet' if 'Code Snippet' in available_cols else None
    
    final_cols = ['AST ID', 'Submission ID', 'Accuracy']
    if code_col: final_cols.append(code_col)
    
    df_report = df_report[final_cols].sort_values(by=['AST ID', 'Submission ID'])
    df_report.to_csv(report_output, index=False)
    print(f"Lean diagnostic report saved to: {report_output}")

if __name__ == '__main__':
    base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
    
    tensor_directory = os.path.join(base_dir, "submission_tensors")
    original_metadata = os.path.join(base_dir, "submissions_with_metadata_updated.csv") 
    
    cleaned_metadata = os.path.join(base_dir, "submissions_metadata_labels_cleaned.csv") 
    lean_report = os.path.join(base_dir, "lean_conflict_report.csv") 
    
    clean_and_report_lean(
        tensor_dir=tensor_directory, 
        metadata_path=original_metadata, 
        cleaned_meta_output=cleaned_metadata, 
        report_output=lean_report
    )