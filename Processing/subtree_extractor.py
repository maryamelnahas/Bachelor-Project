import pandas as pd
import csv
import gc

def yield_s_expressions(csv_path):
    # Reads the dataset in chunks to preserve memory
    for chunk in pd.read_csv(csv_path, chunksize=500):
        for index, row in chunk.iterrows():
            yield row['Submission ID'], row['AST-Expression'], row['Status']

def extract_subtrees_from_string(sexp_string):
    # Extracts subtrees by matching nested parentheses
    subtrees = []
    stack = []
    
    if not isinstance(sexp_string, str):
        return subtrees

    for i, char in enumerate(sexp_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start_idx = stack.pop()
                # Slices the string from the matched open parenthesis to the current close parenthesis
                subtrees.append(sexp_string[start_idx:i+1])
                
    return subtrees

def process_preparsed_dataset(input_csv, output_csv):
    print("Starting direct subtree extraction from S-Expressions...")
    success_count = 0
    total_subtrees = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Submission ID', 'Subtree ID', 'Subtree Expression'])

        for submission_id, sexp, status in yield_s_expressions(input_csv):
            # Skip null S-Expressions or rows where the initial AST creation failed
            if pd.isna(sexp) or str(status).strip().lower() != 'success':
                continue
            
            file_subtrees = extract_subtrees_from_string(sexp)
            
            if file_subtrees:
                success_count += 1
                for idx, subtree in enumerate(file_subtrees):
                    writer.writerow([submission_id, f"subtree_{idx}", subtree])
                    total_subtrees += 1
            
            if success_count > 0 and success_count % 500 == 0:
                gc.collect()
                print(f"Processed {success_count} valid rows...")

    print(f"\nExtraction complete. Processed {success_count} valid submissions.")
    print(f"Generated a total of {total_subtrees} subtrees.")

input_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\code_to_s_expression_results.csv"
output_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\Data\all_extracted_subtrees.csv"

process_preparsed_dataset(input_path, output_path)