import pandas as pd
import requests
import pathlib
import gzip
import pickle
import numpy as np
import os

def get_miplib_best_objective(instance_name):
    url = f"https://miplib.zib.de/instance_details_{instance_name}.html"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Error: Could not find instance '{instance_name}' (Status {response.status_code})"
            
        # Parse all HTML tables on the webpage
        tables = pd.read_html(response.text)
        
        summary_objective = None
        
        for df in tables:
            # Method 1: Try the "Best Known Solution(s)" table first 
            # (Usually cleaner data, tracks 'Exact' matches)
            if 'Objective' in df.columns and 'Exact' in df.columns:
                return df['Objective'].iloc[0]
            
            # Method 2: Capture the main summary table at the top of the page
            if 'Objective' in df.columns and 'Status' in df.columns:
                summary_objective = df['Objective'].iloc[0]
                
        # If the Best Known Solutions table was missing, return the summary table value
        if summary_objective is not None:
            # Strip out asterisks (e.g., "186.0*") which MIPLIB sometimes uses to denote open problems
            return str(summary_objective).replace('*', '').strip()
            
        return "Error: Could not find the Objective value in any table on the page."
        
    except Exception as e:
        return f"An error occurred: {e}"

def update_objective_in_file(file_path, instance_name, new_value):
    import gzip
    import pickle
    # Load existing data
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Update the value
    data[instance_name] = new_value
    # Save back to the same file
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Updated {instance_name} to {new_value} in {file_path}")

# file_directory = './result/miplib2017/miplib2017_purebinary_solved.txt'
# print(file_directory)
# dict_miplib2017binary_solved_objectives = {}
# with open(file_directory) as fp:
#     Lines = fp.readlines()
#     i = 0
#     for line in Lines:
#         # if i > 0: #  start from i==56
#         instance_str = line.strip()
#         print('instance index = ', i)
#         i += 1
#         instance_name = instance_str.split('.')[0]
#         # print(instance_name)
#         best_obj = get_miplib_best_objective(instance_name)
#         # print(f"The best objective value for {instance_name} is: {best_obj}")
#         dict_miplib2017binary_solved_objectives[instance_name] = best_obj


# --- Adapted for miplib2017_binary39 ---
npz_path = './result/miplib2017/miplib2017_binary39.npz'
data = np.load(npz_path)
instance_names = data['miplib2017_binary39']

dict_miplib2017binary39_objectives = {}
for i, instance_name in enumerate(instance_names):
    print(f'Instance {i}: {instance_name}')
    best_obj = get_miplib_best_objective(instance_name)
    print(f"  Best objective: {best_obj}")
    dict_miplib2017binary39_objectives[instance_name] = best_obj

# Save the objective dictionary to a file using gzip and pickle
save_path = './result/miplib2017/miplib2017_binary39_objective.pkl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with gzip.open(save_path, 'wb') as f:
    pickle.dump(dict_miplib2017binary39_objectives, f)


# Example usage:
update_objective_in_file('./result/miplib2017/miplib2017_binary39_objective.pkl', 'harp2', '-7.38998e+07')
update_objective_in_file('./result/miplib2017/miplib2017_binary39_objective.pkl', 'p2756', '3124')
update_objective_in_file('./result/miplib2017/miplib2017_binary39_objective.pkl', 'protfold', '-31')

# (Optional) Load and print a summary for verification
with gzip.open(save_path, 'rb') as f:
    loaded_data = pickle.load(f)
print(f"Loaded {len(loaded_data)} objectives from {save_path}")
for k, v in list(loaded_data.items()):
    print(f"{k}: {v}")