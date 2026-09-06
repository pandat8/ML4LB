import pandas as pd
import requests
import pathlib
import gzip
import pickle

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

# Testing the edge case
print(f"Seymour: {get_miplib_best_objective('seymour')}")
print(f"stp3d: {get_miplib_best_objective('stp3d')}")

# raw_obj = get_miplib_best_objective("seymour")
# # Print the data type of raw_obj
# print(type(raw_obj))

file_directory = './result/miplib2017/miplib2017_purebinary_solved.txt'
print(file_directory)
dict_miplib2017binary_solved_objectives = {}
with open(file_directory) as fp:
    Lines = fp.readlines()
    i = 0
    for line in Lines:
        # if i > 0: #  start from i==56
        instance_str = line.strip()
        print('instance index = ', i)
        i += 1
        instance_name = instance_str.split('.')[0]
        # print(instance_name)
        best_obj = get_miplib_best_objective(instance_name)
        # print(f"The best objective value for {instance_name} is: {best_obj}")
        dict_miplib2017binary_solved_objectives[instance_name] = best_obj

# Save the objective dictionary to a file using gzip and pickle
filename = f'./result/miplib2017/miplib2017_purebinary_solved_objective.pkl'  # instance 100-199
with gzip.open(filename, 'wb') as f:
    pickle.dump(dict_miplib2017binary_solved_objectives, f)

# Load the objective dictionary from the file
filename = f'./result/miplib2017/miplib2017_purebinary_solved_objective.pkl'
with gzip.open(filename, 'rb') as f:
    dict_data = pickle.load(f)

with open(file_directory) as fp:
    Lines = fp.readlines()
    i = 0
    for line in Lines:
        # if i > 0: #  start from i==56
        instance_str = line.strip()
        print('instance index = ', i)
        i += 1
        instance_name = instance_str.split('.')[0]
        print(instance_name)
        best_obj = dict_data[instance_name]
        print(f"The best objective value for {instance_name} is: {best_obj}")
        
