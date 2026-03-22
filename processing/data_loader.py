import pandas as pd
import torch
from torch.utils.data import Dataset
import ast

class SubtreeDatasetWithLabels(Dataset):
    def __init__(self, tokenized_csv_path, metadata_csv_path):
        print("Loading tokenized structural dataset...")
        self.subtrees_df = pd.read_csv(tokenized_csv_path)
        self.subtrees_df['Padded_Sequence'] = self.subtrees_df['Padded_Sequence'].apply(ast.literal_eval)
        
        print("Loading metadata and reconstructing File Names...")
        self.metadata_df = pd.read_csv(metadata_csv_path)
        
        # 1. Reconstruct the File Name safely
        clean_ext = self.metadata_df['filename_ext'].astype(str).str.replace('.', '', regex=False)
        self.metadata_df['Constructed_FileName'] = self.metadata_df['submission_id'].astype(str) + '.' + clean_ext
        
        # 2. Map the binary labels based on the 'status' column
        self.label_mapping = {}
        for index, row in self.metadata_df.iterrows():
            fname = row['Constructed_FileName']
            
            # Map "Accepted" to 1.0 (Correct), and everything else to 0.0 (Incorrect)
            if row['status'] == 'Accepted':
                self.label_mapping[fname] = 1.0
            else:
                self.label_mapping[fname] = 0.0
        
        print("Grouping subtrees by student submission...")
        self.grouped_data = self.subtrees_df.groupby('File Name')
        
        # 3. Only keep files that exist in BOTH the subtrees and our newly created label mapping
        self.file_names = [f for f in self.grouped_data.groups.keys() if f in self.label_mapping]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        submission_data = self.grouped_data.get_group(file_name)
        
        # 1. Node Sequence Tensor 
        sequences = submission_data['Padded_Sequence'].tolist()
        node_sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        
        # 2. Subtree IDs Tensor
        subtree_ids_strings = submission_data['Subtree_ID'].tolist()
        subtree_ids = [int(s.split('_')[1]) for s in subtree_ids_strings]
        subtree_ids_tensor = torch.tensor(subtree_ids, dtype=torch.long)
        
        # 3. Correctness Label Tensor
        label = float(self.label_mapping[file_name])
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return node_sequences_tensor, subtree_ids_tensor, label_tensor, file_name