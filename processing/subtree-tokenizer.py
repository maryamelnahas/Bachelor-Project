import pandas as pd
import re
import json

def build_vocabulary_and_tokenize(csv_path, max_sequence_length=50):
    print("Loading extracted subtrees...")
    df = pd.read_csv(csv_path)

    print("Cleaning S-expressions and extracting tokens...")
    def clean_s_exp(s_exp):
        # Remove parentheses and split by whitespace
        clean_text = re.sub(r'[()]', ' ', str(s_exp))
        return clean_text.split()

    df['Tokens'] = df['Subtree_Expression'].apply(clean_s_exp)

    # Build the vocabulary dictionary
    print("Constructing vocabulary...")
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    current_index = 2

    for tokens in df['Tokens']:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = current_index
                current_index += 1

    # Save vocabulary for downstream mapping and model interpretation
    with open('node_vocabulary.json', 'w') as f:
        json.dump(vocabulary, f, indent=4)

    # Convert text tokens to integer sequences
    print("Encoding tokens into integer sequences...")
    def encode_sequence(tokens):
        return [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens]

    df['Integer_Sequence'] = df['Tokens'].apply(encode_sequence)

    # Pad sequences to a fixed length for neural network tensor compatibility
    print("Padding sequences...")
    def pad_sequence(seq):
        if len(seq) > max_sequence_length:
            return seq[:max_sequence_length]
        return seq + [vocabulary["<PAD>"]] * (max_sequence_length - len(seq))

    df['Padded_Sequence'] = df['Integer_Sequence'].apply(pad_sequence)

    # Save the tokenized dataset
    output_path = csv_path.replace('.csv', '_tokenized.csv')
    df[['File Name', 'Subtree_ID', 'Padded_Sequence']].to_csv(output_path, index=False)
    print(f"Tokenization complete. Output saved to {output_path}")
    print(f"Total unique node tokens identified: {len(vocabulary)}")

# Execution with exact absolute paths configured as raw strings
input_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\extracted_subtrees.csv"
build_vocabulary_and_tokenize(input_path)