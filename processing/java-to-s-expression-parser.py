import pandas as pd
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
import gc
import csv

# 1. Initialize the Parser
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser()
parser.language = JAVA_LANGUAGE

# 2. Create a memory-efficient generator
def yield_code_snippets(csv_path):
    for chunk in pd.read_csv(csv_path, chunksize=500):
        for index, row in chunk.iterrows():
            yield row['File Name'], row['Code Snippet']

# 3. Recursive function to generate AST string
def generate_ast_string(node):
    if not node.is_named:
        return ""
    
    children_strings = []
    for child in node.children:
        if child.is_named:
            child_str = generate_ast_string(child)
            if child_str:
                children_strings.append(child_str)
    
    if children_strings:
        return f"({node.type} {' '.join(children_strings)})"
    else:
        return f"({node.type})"

# 4. Main processing function
def process_dataset(input_csv, output_csv):
    print("Starting to parse the dataset for ASTs...")
    success_count = 0
    error_count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File Name', 'AST-Expression', 'Status'])

    for filename, code in yield_code_snippets(input_csv):
        try:
            tree = parser.parse(bytes(str(code), "utf8"))
            root_node = tree.root_node
            
            ast_expression = ""
            status = "Failed"

            if root_node.type == 'program':
                success_count += 1
                # Pass the root node to the recursive filtering function
                ast_expression = generate_ast_string(root_node)
                status = "Success"
            
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([filename, ast_expression, status])
            
            del tree
            del root_node

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            error_count += 1
            
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([filename, f"Error: {str(e)}", "Error"])

    gc.collect()
    print(f"\nFinished! Successfully generated ASTs for: {success_count} files.")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")

# 5. Run the function with exact absolute paths configured as raw strings
input_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Analytics on Dataset\Programming Mistakes Dataset - Java P0 Solutions.csv"
output_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\code-to-AST-parsing-results.csv"

process_dataset(input_path, output_path)