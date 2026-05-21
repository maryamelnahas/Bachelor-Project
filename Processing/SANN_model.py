import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Simplified Dataset Class ---
class SubtreeDatasetWithLabels(Dataset):
    def __init__(self, tensor_dir, metadata_csv):
        self.tensor_dir = tensor_dir
        self.metadata = pd.read_csv(metadata_csv)
        
        valid_ids = [f.split('.')[0] for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        self.metadata['Submission ID'] = self.metadata['Submission ID'].astype(str)
        self.metadata = self.metadata[self.metadata['Submission ID'].isin(valid_ids)]
        self.metadata.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sub_id = row['Submission ID']
        label = float(row['Accuracy']) 
        
        tensor_path = os.path.join(self.tensor_dir, f"{sub_id}.pt")
        
        # Modified to unpack the dictionary inputs
        tensor_dict = torch.load(tensor_path, weights_only=True) 
        node_sequences = tensor_dict["node_sequences"]
        subtree_ids = tensor_dict["subtree_ids"]
        
        return node_sequences, subtree_ids, torch.tensor(label, dtype=torch.float32), sub_id

# --- 2. Custom Collate Function ---
def pad_collate(batch):
    node_seqs, sub_ids, labels, fnames = zip(*batch)
    
    max_subtrees = max(seq.size(0) for seq in node_seqs)
    seq_len = 35 
    batch_size = len(batch)
    
    padded_node_seqs = torch.zeros((batch_size, max_subtrees, seq_len), dtype=torch.long)
    padded_sub_ids = torch.zeros((batch_size, max_subtrees), dtype=torch.long)
    
    for i, (n_seq, s_id) in enumerate(zip(node_seqs, sub_ids)):
        sub_count = n_seq.size(0)
        padded_node_seqs[i, :sub_count, :] = n_seq
        padded_sub_ids[i, :sub_count] = s_id

    return padded_node_seqs, padded_sub_ids, torch.stack(labels), list(fnames)

# --- 3. Model Architecture ---
class ModifiedSANN(nn.Module):
    def __init__(self, vocab_size, num_unique_subtrees, embedding_dim=128, hidden_dim=64):
        super(ModifiedSANN, self).__init__()
        
        self.node_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.subtree_embedding = nn.Embedding(num_unique_subtrees, hidden_dim, padding_idx=0)
        
        self.node_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        
        self.fc_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        nn.init.xavier_uniform_(self.context_vector)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_sequences, subtree_ids):
        batch_size, num_subtrees, seq_len = node_sequences.size()
        
        mask = (subtree_ids != 0).float().unsqueeze(2) 
        
        flat_nodes = node_sequences.view(batch_size * num_subtrees, seq_len)
        embedded_nodes = self.node_embedding(flat_nodes)
        
        lengths = (flat_nodes != 0).sum(dim=1).clamp(min=1)
        packed_nodes = nn.utils.rnn.pack_padded_sequence(embedded_nodes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        _, (hidden, _) = self.node_lstm(packed_nodes)
        node_repr = hidden[-1].view(batch_size, num_subtrees, -1) 
        
        subtree_repr = self.subtree_embedding(subtree_ids) 
        
        combined_repr = torch.cat((node_repr, subtree_repr), dim=2)
        combined_repr = self.dropout(combined_repr) 
        subtree_vectors = torch.relu(self.fc_combine(combined_repr))
        
        attention_scores = torch.matmul(subtree_vectors, self.context_vector).squeeze(2) 
        attention_weights = torch.sigmoid(attention_scores) * mask.squeeze(2)
        
        weighted_subtrees = subtree_vectors * attention_weights.unsqueeze(2)
        
        sum_weights = attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        source_code_vector = torch.sum(weighted_subtrees, dim=1) / sum_weights 
        
        prediction = self.classifier(source_code_vector)
        
        return prediction, attention_weights, source_code_vector

def evaluate_model(model, dataloader, device):
    model.eval()
    all_true_labels = []
    all_predicted_labels = []
    
    with torch.no_grad():
        for node_seq, sub_id, label, fname in dataloader:
            node_seq, sub_id, label = node_seq.to(device), sub_id.to(device), label.to(device)
            
            prediction, _, _ = model(node_seq, sub_id)
            
            probabilities = torch.sigmoid(prediction.squeeze()) 
            predicted_classes = (probabilities >= 0.5).float()
            
            if predicted_classes.dim() == 0:
                all_predicted_labels.append(predicted_classes.item())
                all_true_labels.append(label.item())
            else:
                all_predicted_labels.extend(predicted_classes.tolist())
                all_true_labels.extend(label.squeeze().tolist())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0, pos_label=0)
    recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0, pos_label=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0, pos_label=0)
    
    return accuracy, precision, recall, f1

def extract_and_save_features(model, dataloader, device, vector_output_path, weights_output_path):
    model.eval()
    extracted_vectors = []
    extracted_weights = []
    submission_ids = []

    with torch.no_grad():
        for node_seq, sub_id, label, fname in tqdm(dataloader, desc="Extracting Features"):
            node_seq, sub_id, label = node_seq.to(device), sub_id.to(device), label.to(device)
            
            _, attention_weights, source_code_vector = model(node_seq, sub_id)
            
            for i in range(len(label)):
                if label[i] == 0.0:
                    vector = source_code_vector[i].cpu().numpy()
                    extracted_vectors.append(vector)
                    
                    valid_length = (sub_id[i] != 0).sum().item()
                    weights = attention_weights[i, :valid_length].cpu().flatten().tolist()
                    
                    extracted_weights.append(json.dumps(weights))
                    submission_ids.append(fname[i])

    df_vectors = pd.DataFrame(extracted_vectors)
    df_vectors.insert(0, 'Submission ID', submission_ids)
    df_vectors.to_csv(vector_output_path, index=False)
    
    df_weights = pd.DataFrame({'Submission ID': submission_ids, 'Attention Weights': extracted_weights})
    df_weights.to_csv(weights_output_path, index=False)
    
    print(f"\nExtracted features for {len(submission_ids)} incorrect submissions.")
    print(f"Vectors saved to: {vector_output_path}")
    print(f"Weights saved to: {weights_output_path}")

# --- 4. Training Execution ---
if __name__ == '__main__':
    print("Initializing Full Training Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing computations on: {device}")

    base_dir = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data"
    vocab_path = os.path.join(base_dir, "node_vocabulary.json")
    subtree_vocab_path = os.path.join(base_dir, "global_subtree_vocabulary.json") # Added global vocab
    tensor_dir = os.path.join(base_dir, "submission_tensors")
    metadata_path = os.path.join(base_dir, "submissions_metadata_labels_cleaned.csv") 
    model_save_path = os.path.join(base_dir, "sann_model_weights.pth")
    metrics_log_path = os.path.join(base_dir, "training_metrics_log.csv")
    
    extracted_vectors_path = os.path.join(base_dir, "incorrect_source_code_vectors.csv")
    extracted_weights_path = os.path.join(base_dir, "incorrect_attention_weights.csv")

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab)
    
    # Load global subtree vocabulary to determine max unique subtrees
    with open(subtree_vocab_path, 'r', encoding='utf-8') as f:
        subtree_vocab = json.load(f)
    MAX_UNIQUE_SUBTREES = len(subtree_vocab) + 1 
    
    BATCH_SIZE = 8
    epochs = 175 # Updated

    full_dataset = SubtreeDatasetWithLabels(tensor_dir, metadata_path)
    
    labels = full_dataset.metadata['Accuracy'].tolist()
    indices = list(range(len(full_dataset)))
    
    train_idx, temp_idx, _, temp_labels = train_test_split(indices, labels, test_size=0.2, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, num_workers=4, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=4, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=4, pin_memory=False)
    
    model = ModifiedSANN(vocab_size=VOCAB_SIZE, num_unique_subtrees=MAX_UNIQUE_SUBTREES).to(device)

    train_labels = [labels[i] for i in train_idx]
    num_negatives = train_labels.count(0.0)
    num_positives = train_labels.count(1.0)
    pos_weight_val = num_negatives / num_positives if num_positives > 0 else 1.0

    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device)) 
    lambda_reg = 3.5e-5 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15) # Updated

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 35 # Updated

    with open(metrics_log_path, 'w', encoding='utf-8') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Accuracy,Duration_Seconds\n")

    print(f"\nBeginning Training on {len(train_dataset)} submissions...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        progress_bar = tqdm(train_dataloader, desc="Training Batches", leave=True)
        
        accumulation_steps = 4 
        optimizer.zero_grad() 
        
        for step, (node_seq, sub_id, label, fname) in enumerate(progress_bar, 1):
            node_seq, sub_id, label = node_seq.to(device), sub_id.to(device), label.to(device)
            
            prediction, attention_weights, _ = model(node_seq, sub_id)
            
            label = label.view_as(prediction).float()
            bce_loss = bce_loss_fn(prediction, label)
            
            epsilon = 1e-8
            mask = (sub_id != 0).float()

            entropy_per_submission = torch.sum(attention_weights * torch.log(attention_weights + epsilon) * mask, dim=1)
            entropy_reg = -lambda_reg * torch.mean(entropy_per_submission)
            
            loss = (bce_loss + entropy_reg) / accumulation_steps
            loss.backward()
            
            if step % accumulation_steps == 0 or step == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            progress_bar.set_postfix({'avg_train_loss': f"{train_loss/step:.4f}"})
            
        avg_train_loss = train_loss / len(train_dataloader)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for node_seq, sub_id, label, fname in val_dataloader:
                node_seq, sub_id, label = node_seq.to(device), sub_id.to(device), label.to(device)
                
                prediction, attention_weights, _ = model(node_seq, sub_id)
                label = label.view_as(prediction).float()
                
                bce = bce_loss_fn(prediction, label)
                mask = (sub_id != 0).float()
                
                entropy_per_submission = torch.sum(attention_weights * torch.log(attention_weights + epsilon) * mask, dim=1)
                entropy = -lambda_reg * torch.mean(entropy_per_submission)
                
                val_loss += (bce + entropy).item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        
        val_accuracy, _, _, _ = evaluate_model(model, val_dataloader, device)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Time: {duration:.2f}s")
        
        with open(metrics_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_accuracy:.4f},{duration:.2f}\n")

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  --> Early stopping triggered after {epoch+1} epochs.")
                break

    print("\nEvaluating best model on unseen test data...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    
    accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader, device)
    
    print("\n--- Program Correctness Prediction Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    with open(metrics_log_path, 'a', encoding='utf-8') as f:
        f.write("\n--- Final Test Metrics ---\n")
        f.write(f"Accuracy,{accuracy:.4f}\n")
        f.write(f"Precision,{precision:.4f}\n")
        f.write(f"Recall,{recall:.4f}\n")
        f.write(f"F1-Score,{f1:.4f}\n")
        
    print("\nInitiating Feature Extraction for Clustering...")
    full_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=4, pin_memory=False)
    extract_and_save_features(model, full_dataloader, device, extracted_vectors_path, extracted_weights_path)