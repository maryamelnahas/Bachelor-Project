import os
import time

output_path = r"D:\Gam3a\Semester 8\Bachelor Thesis\Project Implementation\code_to_s_expression_results.csv"

# 1. Check if the file is the "old" one or the "new" one
if os.path.exists(output_path):
    mod_time = os.path.getmtime(output_path)
    print("Last Modified Time:", time.ctime(mod_time))

# 2. Check the environment's current working directory
print("Python Execution Directory:", os.getcwd())