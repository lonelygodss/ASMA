import os
import glob
import re

def process_bb_file(file_path):
    """Process a file with 'bb' in its name according to the requirements."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract data
    total_transfer_time = 0
    active_times = {}
    current_subfunction = None
    
    for line in lines:
        line = line.strip()
        
        # Extract subfunction name
        subfunction_match = re.match(r'excecute subfunction: (.+)', line)
        if subfunction_match:
            current_subfunction = subfunction_match.group(1)
            if current_subfunction not in active_times:
                active_times[current_subfunction] = 0
            continue
        
        # Extract execution time
        exec_time_match = re.match(r'execution time: (.+)', line)
        if exec_time_match and current_subfunction:
            active_times[current_subfunction] += float(exec_time_match.group(1))
            continue
        
        # Extract transfer time
        transfer_time_match = re.match(r'transfer time: (.+)', line)
        if transfer_time_match:
            total_transfer_time += float(transfer_time_match.group(1))
            continue
    
    # Format time values with 1 decimal place
    total_transfer_time = round(total_transfer_time, 1)
    for subfunction in active_times:
        active_times[subfunction] = round(active_times[subfunction], 1)
    
    # Calculate utilization (keep original precision)
    utilization = {}
    for subfunction, active_time in active_times.items():
        if total_transfer_time > 0:
            utilization[subfunction] = active_time / total_transfer_time
        else:
            utilization[subfunction] = 0.0
    
    # Prepare new lines to add
    new_lines = []
    new_lines.append(f"total_transfer_time: {total_transfer_time:.1f}\n")
    for subfunction, active_time in active_times.items():
        new_lines.append(f"active_time_{subfunction}: {active_time:.1f}\n")
    for subfunction, util in utilization.items():
        new_lines.append(f"utilization_{subfunction}: {util}\n")
    
    # Insert new lines before the "Simulation finished" line
    simulation_line_index = next((i for i, line in enumerate(lines) if "Simulation finished" in line), len(lines))
    lines = lines[:simulation_line_index] + new_lines + lines[simulation_line_index:]
    
    # Format numerical values to have 1 decimal place (only for time values)
    formatted_lines = []
    for line in lines:
        if "time:" in line and "utilization" not in line:
            # Format time values with 1 decimal place
            formatted_line = re.sub(r':\s*(\d+\.\d+)', lambda m: f": {float(m.group(1)):.1f}", line)
        else:
            formatted_line = line
        formatted_lines.append(formatted_line)
    
    # Write back to file
    with open(file_path, 'w') as file:
        file.writelines(formatted_lines)
    
    return True

def process_non_bb_file(file_path):
    """Process a file without 'bb' in its name according to the requirements."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # First pass: extract all required data
    active_times = {}  # Track execution time by subfunction
    special_transfer_times = []
    current_subfunction = None
    
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        
        # Extract subfunction name
        subfunction_match = re.match(r'excecute subfunction: (.+)', current_line)
        if subfunction_match:
            current_subfunction = subfunction_match.group(1)
            if current_subfunction not in active_times:
                active_times[current_subfunction] = 0
        
        # Extract execution time
        exec_time_match = re.match(r'execution time: (.+)', current_line)
        if exec_time_match and current_subfunction:
            active_times[current_subfunction] += float(exec_time_match.group(1))
        
        # Find transfer times after Distribution
        if i > 0 and "excecute subfunction: DISTRIBUTE" in lines[i-1]:
            if "transfer time:" in current_line:
                transfer_time_match = re.match(r'transfer time: (.+)', current_line)
                if transfer_time_match:
                    special_transfer_times.append(float(transfer_time_match.group(1)))
        
        # Find transfer times before Concat
        if i < len(lines) - 1 and "excecute subfunction: CONCAT" in lines[i+1]:
            if "transfer time:" in current_line:
                transfer_time_match = re.match(r'transfer time: (.+)', current_line)
                if transfer_time_match:
                    special_transfer_times.append(float(transfer_time_match.group(1)))
        
        i += 1
    
    # Calculate totals
    total_transfer_time = sum(special_transfer_times)
    total_transfer_time = round(total_transfer_time, 1)
    
    # Round active times to 1 decimal place
    for subfunction in active_times:
        active_times[subfunction] = round(active_times[subfunction], 1)
    
    # Calculate total_execution_time
    total_execution_time = sum(active_times.values())
    total_execution_time = round(total_execution_time, 1)
    
    # Calculate utilization for each subfunction
    utilization = {}
    for subfunction, active_time in active_times.items():
        if total_transfer_time > 0:
            utilization[subfunction] = active_time / total_transfer_time
        else:
            utilization[subfunction] = 0.0
    
    # Prepare new lines to add
    new_lines = []
    new_lines.append(f"total_transfer_time: {total_transfer_time:.1f}\n")
    new_lines.append(f"total_execution_time: {total_execution_time:.1f}\n")
    for subfunction, active_time in active_times.items():
        new_lines.append(f"active_time_{subfunction}: {active_time:.1f}\n")
    for subfunction, util in utilization.items():
        new_lines.append(f"utilization_{subfunction}: {util}\n")
    
    # Insert new lines before the "Simulation finished" line
    simulation_line_index = next((i for i, line in enumerate(lines) if "Simulation finished" in line), len(lines))
    lines = lines[:simulation_line_index] + new_lines + lines[simulation_line_index:]
    
    # Format numerical values to have 1 decimal place (only for time values)
    formatted_lines = []
    for line in lines:
        if "time:" in line and "utilization" not in line:
            # Format time values with 1 decimal place
            formatted_line = re.sub(r':\s*(\d+\.\d+)', lambda m: f": {float(m.group(1)):.1f}", line)
        else:
            formatted_line = line
        formatted_lines.append(formatted_line)
    
    # Write back to file
    with open(file_path, 'w') as file:
        file.writelines(formatted_lines)
    
    return True

def process_pipeline_files(search_pattern):
    """Process all pipeline data files according to requirements."""
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {search_pattern}")
        return
    
    bb_files = []
    non_bb_files = []
    
    for file_path in matching_files:
        if 'bb' in os.path.basename(file_path):
            bb_files.append(file_path)
        else:
            non_bb_files.append(file_path)
    
    print(f"Found {len(bb_files)} files with 'bb' in name")
    print(f"Found {len(non_bb_files)} files without 'bb' in name")
    
    # Process each file
    success_count = 0
    for file_path in bb_files:
        if process_bb_file(file_path):
            success_count += 1
            print(f"Processed BB file: {file_path}")
    
    for file_path in non_bb_files:
        if process_non_bb_file(file_path):
            success_count += 1
            print(f"Processed non-BB file: {file_path}")
    
    print(f"Completed: {success_count} of {len(matching_files)} files processed successfully.")

# Example usage
process_pipeline_files("./pipline_data/apipline_data*.txt")