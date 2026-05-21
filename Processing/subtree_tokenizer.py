import pandas as pd
import re
import json
import torch
import os
import gc

def build_vocab_and_tensorize(csv_path, output_dir, vocab_output_path, subtree_vocab_path, max_seq_length=35, max_subtrees=200):
    print("Pass 1: Constructing vocabularies...")
    node_vocabulary = {"<PAD>": 0, "<UNK>": 1}
    subtree_vocabulary = {"<PAD>": 0} # Global ID map for structural embeddings
    
    current_node_index = 2
    current_subtree_index = 1
    
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        for s_exp in chunk['Subtree Expression']:
            if pd.isna(s_exp):
                continue
            
            # Map the entire subtree structure to a Global ID
            if s_exp not in subtree_vocabulary:
                subtree_vocabulary[s_exp] = current_subtree_index
                current_subtree_index += 1
            
            clean_text = re.sub(r'[()]', ' ', str(s_exp))
            for token in clean_text.split():
                if token not in node_vocabulary:
                    node_vocabulary[token] = current_node_index
                    current_node_index += 1
        gc.collect()

    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        json.dump(node_vocabulary, f, indent=4)
        
    with open(subtree_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(subtree_vocabulary, f, indent=4)
        
    print(f"Total unique node tokens: {len(node_vocabulary)}")
    print(f"Total unique global subtrees: {len(subtree_vocabulary)}")
    
    print(f"Pass 2: Tensorizing sequences and global IDs to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    current_submission_id = None
    current_sequences = []
    current_global_ids = []
    rows_processed = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        for _, row in chunk.iterrows():
            sub_id = row['Submission ID']
            s_exp = row['Subtree Expression']
            
            if pd.isna(s_exp):
                continue
            
            if sub_id != current_submission_id:
                if current_submission_id is not None and current_sequences:
                    # Save both the sequences and the corresponding Global IDs as a dictionary
                    seqs_to_save = current_sequences[:max_subtrees]
                    ids_to_save = current_global_ids[:max_subtrees]
                    
                    tensor_dict = {
                        "node_sequences": torch.tensor(seqs_to_save, dtype=torch.long),
                        "subtree_ids": torch.tensor(ids_to_save, dtype=torch.long)
                    }
                    torch.save(tensor_dict, os.path.join(output_dir, f"{current_submission_id}.pt"))
                
                current_submission_id = sub_id
                current_sequences = []
                current_global_ids = []
            
            # 1. Retrieve Global Subtree ID
            global_id = subtree_vocabulary[s_exp]
            current_global_ids.append(global_id)
            
            # 2. Tokenize Node Sequence
            clean_text = re.sub(r'[()]', ' ', str(s_exp))
            tokens = clean_text.split()
            sequence = [node_vocabulary.get(token, node_vocabulary["<UNK>"]) for token in tokens]
            
            if len(sequence) > max_seq_length:
                padded_sequence = sequence[:max_seq_length]
            else:
                padded_sequence = sequence + [node_vocabulary["<PAD>"]] * (max_seq_length - len(sequence))
                
            current_sequences.append(padded_sequence)
            rows_processed += 1
            
        print(f"Processed {rows_processed} rows...")
        gc.collect()

    if current_submission_id is not None and current_sequences:
        seqs_to_save = current_sequences[:max_subtrees]
        ids_to_save = current_global_ids[:max_subtrees]
        tensor_dict = {
            "node_sequences": torch.tensor(seqs_to_save, dtype=torch.long),
            "subtree_ids": torch.tensor(ids_to_save, dtype=torch.long)
        }
        torch.save(tensor_dict, os.path.join(output_dir, f"{current_submission_id}.pt"))

    print("Tensorization complete.")

if __name__ == '__main__':
    base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
    input_path = os.path.join(base_dir, "all_extracted_subtrees.csv")
    output_tensor_dir = os.path.join(base_dir, "submission_tensors")
    vocab_path = os.path.join(base_dir, "node_vocabulary.json")
    subtree_vocab_path = os.path.join(base_dir, "global_subtree_vocabulary.json")
    
    build_vocab_and_tensorize(input_path, output_tensor_dir, vocab_path, subtree_vocab_path)