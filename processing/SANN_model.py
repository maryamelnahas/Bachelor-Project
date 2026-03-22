import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from data_loader import SubtreeDatasetWithLabels

class ModifiedSANN(nn.Module):
    def __init__(self, vocab_size, num_unique_subtrees, embedding_dim=128, hidden_dim=64):
        super(ModifiedSANN, self).__init__()
        
        # Two-way Embedding Pathways
        self.node_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.node_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.subtree_embedding = nn.Embedding(num_unique_subtrees, hidden_dim)
        
        self.fc_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_sequences, subtree_ids):
        batch_size, num_subtrees, seq_len = node_sequences.size()
        
        # Node-based embedding processing
        flat_nodes = node_sequences.view(batch_size * num_subtrees, seq_len)
        embedded_nodes = self.node_embedding(flat_nodes)
        _, (hidden, _) = self.node_lstm(embedded_nodes)
        node_repr = hidden[-1].view(batch_size, num_subtrees, -1) 
        
        # Subtree-based embedding processing
        subtree_repr = self.subtree_embedding(subtree_ids) 
        
        # Concatenation and Time-Distributed fusion
        combined_repr = torch.cat((node_repr, subtree_repr), dim=2)
        subtree_vectors = torch.relu(self.fc_combine(combined_repr)) 
        
        # Modified Attention Mechanism (Sigmoid Activation)
        attention_scores = torch.matmul(subtree_vectors, self.context_vector).squeeze(2) 
        attention_weights = torch.sigmoid(attention_scores) 
        
        # Source Code Vector Aggregation
        weighted_subtrees = subtree_vectors * attention_weights.unsqueeze(2)
        source_code_vector = torch.sum(weighted_subtrees, dim=1) 
        
        # Binary Correctness Prediction
        prediction = torch.sigmoid(self.classifier(source_code_vector))
        
        return prediction, attention_weights, subtree_vectors


# --- Training Execution ---
if __name__ == '__main__':
    print("Initializing Full Training Pipeline...")

    vocab_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\node_vocabulary.json"
    tokenized_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\extracted_subtrees_tokenized.csv"
    metadata_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\p0_metadata.csv" 

    # Load Vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab)
    MAX_UNIQUE_SUBTREES = 50000 

    # Initialize Dataset and DataLoader
    dataset = SubtreeDatasetWithLabels(tokenized_path, metadata_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize Model
    model = ModifiedSANN(vocab_size=VOCAB_SIZE, num_unique_subtrees=MAX_UNIQUE_SUBTREES)

    # Define Optimizer and Base Loss Function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bce_loss_fn = nn.BCELoss()
    lambda_reg = 3.5e-5 # Entropy regularization weight from the literature

    epochs = 5 

    print(f"\nBeginning Training on {len(dataset)} submissions...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for node_seq, sub_id, label, fname in dataloader:
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            prediction, attention_weights, _ = model(node_seq, sub_id)
            
            # 3. Calculate Binary Cross Entropy Loss
            bce_loss = bce_loss_fn(prediction, label)
            
            # 4. Calculate Entropy Regularization Term
            epsilon = 1e-8
            entropy_reg = -lambda_reg * torch.sum(attention_weights * torch.log(attention_weights + epsilon))
            
            # 5. Combine and Backpropagate
            loss = bce_loss + entropy_reg
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 6. Calculate accuracy (Threshold at 0.5)
            predicted_class = 1.0 if prediction.item() >= 0.5 else 0.0
            if predicted_class == label.item():
                correct_predictions += 1
                
        # Epoch Results
        epoch_accuracy = (correct_predictions / len(dataset)) * 100
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss:.4f} | Training Accuracy: {epoch_accuracy:.2f}%")
        
    print("\nSaving trained model weights to disk...")
    torch.save(model.state_dict(), r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\sann_model_weights.pth")
    print("Model saved successfully!")