import pandas as pd
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
import gc
import csv

JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser()
parser.language = JAVA_LANGUAGE

def yield_code_snippets(csv_path):
    for chunk in pd.read_csv(csv_path, chunksize=500):
        for index, row in chunk.iterrows():
            yield row['File Name'], row['Code Snippet']

# Recursive subtree extraction
def extract_all_subtrees(node):
    subtrees = []
    if node.is_named:
        subtrees.append(str(node))
        for child in node.children:
            subtrees.extend(extract_all_subtrees(child))
    return subtrees

# Main processing function
def process_dataset_for_subtrees(input_csv, output_csv):
    print("Starting recursive subtree extraction...")
    success_count = 0
    error_count = 0
    total_subtrees = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File Name', 'Subtree_ID', 'Subtree_Expression'])

    for filename, code in yield_code_snippets(input_csv):
        try:
            tree = parser.parse(bytes(str(code), "utf8"))
            root_node = tree.root_node
            
            if root_node.type == 'program':
                success_count += 1
                file_subtrees = extract_all_subtrees(root_node)
                
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for idx, subtree in enumerate(file_subtrees):
                        writer.writerow([filename, f"subtree_{idx}", subtree])
                        total_subtrees += 1
            
            del tree
            del root_node

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            error_count += 1

    gc.collect()
    print(f"\nExtraction complete. Processed {success_count} files.")
    print(f"Generated a total of {total_subtrees} subtrees.")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")

input_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Analytics on Dataset\Programming Mistakes Dataset - Java P0 Solutions.csv"
output_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\extracted_subtrees.csv"

process_dataset_for_subtrees(input_path, output_path)