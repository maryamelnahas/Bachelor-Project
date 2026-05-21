import os
import csv

def main():
    # 1. Define Paths
    metadata_folder_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project CodeNet Dataset\Project_CodeNet\Project_CodeNet\metadata"
    output_file = "metadata_P0_to_P499.csv"

    # Verify the metadata folder exists
    if not os.path.exists(metadata_folder_path):
        print(f"Error: The directory {metadata_folder_path} does not exist.")
        return

    # 2. Define the desired output columns in the requested order
    output_headers = [
        "Problem ID", "Submission ID", "User ID", 
        "Date", "Status", "Accuracy"
    ]

    total_extracted_count = 0

    # 3. Open the new consolidated CSV for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(output_headers) # Write the header row

        # 4. Loop through p00000 to p00499
        for i in range(500):
            problem_id = f"p{i:05d}"
            # CodeNet metadata files are typically named with a .csv extension
            file_path = os.path.join(metadata_folder_path, f"{problem_id}.csv")

            # Skip to the next problem if the file is missing
            if not os.path.exists(file_path):
                continue

            try:
                # 5. Read the individual problem's metadata file
                with open(file_path, mode='r', encoding='utf-8') as in_f:
                    reader = csv.DictReader(in_f)
                    
                    for row in reader:
                        # 6. Filter by language = Java
                        if row.get("language") == "Java":
                            
                            # Extract only the specified columns and map them to the new row
                            extracted_row = [
                                row.get("problem_id", ""),
                                row.get("submission_id", ""),
                                row.get("user_id", ""),
                                row.get("date", ""),
                                row.get("status", ""),
                                row.get("accuracy", "")
                            ]
                            
                            writer.writerow(extracted_row)
                            total_extracted_count += 1
                            
            except Exception as e:
                print(f"Error reading or processing file {file_path}: {e}")
                continue

    print(f"Extraction complete! Saved {total_extracted_count} Java submission records to '{output_file}'.")

if __name__ == "__main__":
    main()