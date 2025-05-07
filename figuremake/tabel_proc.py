import os
import glob
import re

def remove_energy_lines(file_path):
    """Read a file, remove all lines containing 'energy', and save back to the same file."""
    try:
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Filter out lines containing 'energy'
        filtered_lines = [line for line in lines if 'energy' not in line.lower()]
        
        # Write the filtered content back to the original file
        with open(file_path, 'w') as file:
            file.writelines(filtered_lines)
            
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_pipeline_files(search_pattern):
    """Find all files matching the pattern and remove energy lines from each."""
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {search_pattern}")
        return
    
    # Process each file
    success_count = 0
    for file_path in matching_files:
        if remove_energy_lines(file_path):
            success_count += 1
            print(f"Processed: {file_path}")
    
    print(f"Completed: {success_count} of {len(matching_files)} files processed successfully.")

# Example usage
process_pipeline_files("./pipline_data/apipline_data*.txt")