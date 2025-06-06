from analysis.data_read import SimulationDataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter  # Add FormatStrFormatter
import math

datapath = './'
filename1 = 'simulation_results_hidden.csv'
filename2 = 'simulation_results_ffn.csv'
dataloader1 = SimulationDataLoader(datapath,filename1,'Hidden Dimension')
dataloader2 = SimulationDataLoader(datapath,filename2,'FFn Dimension')
grouped_time_hidden = dataloader1.prepare_data_for_grouped_comparison(metric='Time')
grouped_energy_hidden = dataloader1.prepare_data_for_grouped_comparison(metric='Energy')
grouped_time_ffn = dataloader2.prepare_data_for_grouped_comparison(metric='Time')
grouped_energy_hidden = dataloader2.prepare_data_for_grouped_comparison(metric='Energy')

def save_normalized_improvements_to_csv(hidden_time, hidden_energy, ffn_time, ffn_energy, baseline_compiler='b_b', output_file='normalized_improvements.csv'):
    """
    Save normalized improvements (relative to baseline) to a CSV file, including statistical analysis.
    
    Parameters:
    -----------
    hidden_time, hidden_energy, ffn_time, ffn_energy : dict
        The grouped data dictionaries from the dataloader
    baseline_compiler : str
        The compiler to use as baseline for normalization
    output_file : str
        Path to save the output CSV file
    """
    import csv
    import numpy as np
    
    # Extract dimensions from the data
    array_sizes = hidden_time['dimensions']['array_sizes']
    hidden_dims = hidden_time['dimensions']['scale_dims']
    ffn_dims = ffn_time['dimensions']['scale_dims']
    compilers = hidden_time['dimensions']['compilers']
    
    # Define compiler friendly names
    compiler_names = {
        'b_b': 'Baseline',
        's_p': 'Standard-Parallel',
        's_s': 'Scatter-Parallel',
        's_sp': 'Mutual-Parallel'
    }
    
    # Dictionary to collect statistics for each compiler (excluding baseline)
    stats = {}
    for compiler in [c for c in compilers if c != baseline_compiler]:
        stats[compiler] = {
            'Hidden_Time': [],
            'Hidden_Energy': [],
            'FFN_Time': [],
            'FFN_Energy': []
        }
    
    # Open the file for writing
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        header = ['Dimension Type', 'Array Size', 'Dimension Value', 'Metric', 'Compiler', 'Raw Value', 'Normalized Value']
        writer.writerow(header)
        
        # Process Hidden Dimension data
        for array_size in array_sizes:
            for dim in hidden_dims:
                for compiler in compilers:
                    # Process Time
                    baseline_time = float(hidden_time[array_size][dim][baseline_compiler])
                    compiler_time = float(hidden_time[array_size][dim][compiler])
                    normalized_time = compiler_time / baseline_time
                    
                    # Write time row
                    writer.writerow([
                        'Hidden', 
                        array_size, 
                        dim, 
                        'Time', 
                        compiler_names[compiler], 
                        f"{compiler_time:.6f}", 
                        f"{normalized_time:.6f}"
                    ])
                    
                    # Collect statistics (except for baseline)
                    if compiler != baseline_compiler:
                        stats[compiler]['Hidden_Time'].append(normalized_time)
                    
                    # Process Energy
                    baseline_energy = float(hidden_energy[array_size][dim][baseline_compiler])
                    compiler_energy = float(hidden_energy[array_size][dim][compiler])
                    normalized_energy = compiler_energy / baseline_energy
                    
                    # Write energy row
                    writer.writerow([
                        'Hidden', 
                        array_size, 
                        dim, 
                        'Energy', 
                        compiler_names[compiler], 
                        f"{compiler_energy:.6f}", 
                        f"{normalized_energy:.6f}"
                    ])
                    
                    # Collect statistics (except for baseline)
                    if compiler != baseline_compiler:
                        stats[compiler]['Hidden_Energy'].append(normalized_energy)
        
        # Process FFN Dimension data
        for array_size in array_sizes:
            for dim in ffn_dims:
                for compiler in compilers:
                    # Process Time
                    baseline_time = float(ffn_time[array_size][dim][baseline_compiler])
                    compiler_time = float(ffn_time[array_size][dim][compiler])
                    normalized_time = compiler_time / baseline_time
                    
                    # Write time row
                    writer.writerow([
                        'FFN', 
                        array_size, 
                        dim, 
                        'Time', 
                        compiler_names[compiler], 
                        f"{compiler_time:.6f}", 
                        f"{normalized_time:.6f}"
                    ])
                    
                    # Collect statistics (except for baseline)
                    if compiler != baseline_compiler:
                        stats[compiler]['FFN_Time'].append(normalized_time)
                    
                    # Process Energy
                    baseline_energy = float(ffn_energy[array_size][dim][baseline_compiler])
                    compiler_energy = float(ffn_energy[array_size][dim][compiler])
                    normalized_energy = compiler_energy / baseline_energy
                    
                    # Write energy row
                    writer.writerow([
                        'FFN', 
                        array_size, 
                        dim, 
                        'Energy', 
                        compiler_names[compiler], 
                        f"{compiler_energy:.6f}", 
                        f"{normalized_energy:.6f}"
                    ])
                    
                    # Collect statistics (except for baseline)
                    if compiler != baseline_compiler:
                        stats[compiler]['FFN_Energy'].append(normalized_energy)
        
        # Add empty row as separator
        writer.writerow([])
        
        # Write statistics header
        writer.writerow(['STATISTICS'])
        writer.writerow(['Compiler', 'Dimension Type', 'Metric', 'Average', 'Min (Best)', 'Max (Worst)', 'Range'])
        
        # Calculate and write statistics
        for compiler in [c for c in compilers if c != baseline_compiler]:
            for dim_type in ['Hidden', 'FFN']:
                for metric in ['Time', 'Energy']:
                    key = f"{dim_type}_{metric}"
                    values = stats[compiler][key]
                    
                    if values:  # Make sure we have data to calculate statistics
                        avg = np.mean(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        range_val = max_val - min_val
                        
                        writer.writerow([
                            compiler_names[compiler],
                            dim_type,
                            metric,
                            f"{avg:.6f}",
                            f"{min_val:.6f}",
                            f"{max_val:.6f}",
                            f"{range_val:.6f}"
                        ])
        
        # Add overall improvement statistics
        writer.writerow([])
        writer.writerow(['OVERALL STATISTICS'])
        writer.writerow(['Compiler', 'Average Improvement', 'Best Improvement', 'Worst Performance', 'Improvement Range'])
        
        for compiler in [c for c in compilers if c != baseline_compiler]:
            # Combine all values across dimension types and metrics
            all_values = (
                stats[compiler]['Hidden_Time'] + 
                stats[compiler]['Hidden_Energy'] + 
                stats[compiler]['FFN_Time'] + 
                stats[compiler]['FFN_Energy']
            )
            
            if all_values:
                avg = np.mean(all_values)
                min_val = np.min(all_values)
                max_val = np.max(all_values)
                range_val = max_val - min_val
                
                # Calculate improvement percentage (lower is better, so improvement is 1 - value)
                avg_improvement = (1 - avg) * 100  # Convert to percentage
                best_improvement = (1 - min_val) * 100
                worst_improvement = (1 - max_val) * 100  # Could be negative if worse than baseline
                
                writer.writerow([
                    compiler_names[compiler],
                    f"{avg_improvement:.2f}%",
                    f"{best_improvement:.2f}%",
                    f"{worst_improvement:.2f}%",
                    f"{range_val * 100:.2f}%"
                ])
    
    print(f"Normalized improvements and statistics saved to {output_file}")

def plot_combined_comparison(hidden_time, hidden_energy, ffn_time, ffn_energy, baseline_compiler='b_b', save_path=None):
    # Extract dimensions from the data
    array_sizes = hidden_time['dimensions']['array_sizes']
    hidden_dims = hidden_time['dimensions']['scale_dims']
    ffn_dims = ffn_time['dimensions']['scale_dims']
    compilers = hidden_time['dimensions']['compilers']
    
    # Define compiler colors and friendly names
    compiler_colors = {
        'b_b': '#AAAACC',    # Baseline - blue-gray
        's_p': '#AACCDD',    # Parallel - light blue
        's_s': '#E6DFB8',    # Scatter - tan
        's_sp': '#F2C1B6'    # Scatter-Parallel - salmon
    }
    
    compiler_names = {
        'b_b': 'Baseline',
        's_p': 'Standard-Parallel',
        's_s': 'Scatter-Parallel',
        's_sp': 'Mutual-Parallel'
    }
    
    # Function to convert dimension to power of 2 format
    def format_as_power_of_two(dim):
        if dim == 0:
            return "0"
        power = int(math.log2(dim))
        if 2**power == dim:  # Perfect power of 2
            return f"$2^{{{power}}}$"
        else:
            return str(dim)
    
    # Calculate figure size to make subplots square
    subplot_size = 3  # Base size for each subplot in inches
    fig_width = subplot_size * len(array_sizes) * 2
    fig_height = subplot_size * 2  # 2 rows
    
    # Create figure with calculated dimensions - don't use constrained_layout here
    fig, axes = plt.subplots(2, len(array_sizes)*2, figsize=(fig_width, fig_height))
    
    # Process each dataset separately
    #-------------------------------------------------
    # FIRST HALF: Process hidden dimension data (left side)
    #-------------------------------------------------
    for col_idx, array_size in enumerate(array_sizes):
        # Get the subplot indexes
        time_ax_idx = (0, col_idx)
        energy_ax_idx = (1, col_idx)
        
        #------------------------------
        # Process TIME data (top row)
        #------------------------------
        ax_time = axes[time_ax_idx]
        
        # Set aspect ratio to make plot square - this is the key to square subplots
        ax_time.set_aspect('auto', adjustable='box')
        
        # Set y-axis formatter to show 1 decimal place
        ax_time.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Increase tick label size
        ax_time.tick_params(axis='both', labelsize=12)
        
        # Position of each group of bars on the x-axis
        positions = np.arange(len(hidden_dims))
        width = 0.2  # Width of each bar
        
        # Plot bars for each compiler
        for comp_idx, compiler in enumerate(compilers):
            offset = (comp_idx - 1.5) * width
            normalized_values = []
            
            for i, dim in enumerate(hidden_dims):
                # Get baseline value for normalization
                baseline_value = float(hidden_time[array_size][dim][baseline_compiler])
                
                # Get current compiler value
                compiler_value = float(hidden_time[array_size][dim][compiler])
                
                # Normalize to baseline
                normalized_value = compiler_value / baseline_value
                normalized_values.append(normalized_value)
                
                # Annotate the unnormalized value on top of baseline bar
                if compiler == baseline_compiler:
                    text = f"{baseline_value:.2e}" if baseline_value >= 1000 else f"{baseline_value:.2f}"
                    ax_time.text(positions[i] + offset, normalized_value + 0.05, text, 
                           ha='center', va='bottom', fontsize=7, rotation=45)
            
            # Create bars
            ax_time.bar(positions + offset, normalized_values, 
                    width=width, color=compiler_colors[compiler], 
                    label=compiler_names[compiler] if col_idx == 0 else "")
        
        # Add subplot title (only for top row) with larger font
        ax_time.set_title(f"Array Size: {array_size}", fontsize=14)
        
        # Format x-axis (no labels for top row)
        ax_time.set_xticks(positions)
        ax_time.set_xticklabels([])
        
        # Set y-axis range
        max_y_time = max([float(hidden_time[array_size][hd][c]) / 
                     float(hidden_time[array_size][hd][baseline_compiler])
                     for hd in hidden_dims for c in compilers
                     if hidden_time[array_size][hd][c] is not None])
        
        ax_time.set_ylim(0, max_y_time * 1.2)  # 20% padding
        
        # Add horizontal line at y=1.0 to indicate baseline
        ax_time.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add y-axis label only on the leftmost subplot with larger font
        if col_idx == 0:
            ax_time.set_ylabel("Normalized Time\n(relative to baseline)", fontsize=14)
        
        # Add gridlines
        ax_time.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        #--------------------------------
        # Process ENERGY data (bottom row)
        #--------------------------------
        ax_energy = axes[energy_ax_idx]
        
        # Set aspect ratio to make plot square
        ax_energy.set_aspect('auto', adjustable='box')
        
        # Set y-axis formatter to show 1 decimal place
        ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Increase tick label size
        ax_energy.tick_params(axis='both', labelsize=12)
        
        # Plot bars for each compiler
        for comp_idx, compiler in enumerate(compilers):
            offset = (comp_idx - 1.5) * width
            normalized_values = []
            
            for i, dim in enumerate(hidden_dims):
                # Get baseline value for normalization
                baseline_value = float(hidden_energy[array_size][dim][baseline_compiler])
                
                # Get current compiler value
                compiler_value = float(hidden_energy[array_size][dim][compiler])
                
                # Normalize to baseline
                normalized_value = compiler_value / baseline_value
                normalized_values.append(normalized_value)
                
                # Annotate the unnormalized value on top of baseline bar
                if compiler == baseline_compiler:
                    text = f"{baseline_value:.2e}" if baseline_value >= 1000 else f"{baseline_value:.2f}"
                    ax_energy.text(positions[i] + offset, normalized_value + 0.05, text, 
                           ha='center', va='bottom', fontsize=7, rotation=45)
            
            # Create bars 
            ax_energy.bar(positions + offset, normalized_values, 
                    width=width, color=compiler_colors[compiler])
        
        # Format x-axis (labels for bottom row) with power of 2 format
        ax_energy.set_xticks(positions)
        ax_energy.set_xticklabels([format_as_power_of_two(dim) for dim in hidden_dims])
        ax_energy.set_xlabel("Model Dimension", fontsize=14)
        
        # Set y-axis range
        max_y_energy = max([float(hidden_energy[array_size][hd][c]) / 
                     float(hidden_energy[array_size][hd][baseline_compiler])
                     for hd in hidden_dims for c in compilers
                     if hidden_energy[array_size][hd][c] is not None])
        
        ax_energy.set_ylim(0, max_y_energy * 1.2)  # 20% padding
        
        # Add horizontal line at y=1.0 to indicate baseline
        ax_energy.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add y-axis label only on the leftmost subplot with larger font
        if col_idx == 0:
            ax_energy.set_ylabel("Normalized Energy\n(relative to baseline)", fontsize=14)
        
        # Add gridlines
        ax_energy.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    #-------------------------------------------------
    # SECOND HALF: Process FFN dimension data (right side)
    #-------------------------------------------------
    for col_idx, array_size in enumerate(array_sizes):
        # Get the subplot indexes (offset by len(array_sizes) to place in right half)
        time_ax_idx = (0, col_idx + len(array_sizes))
        energy_ax_idx = (1, col_idx + len(array_sizes))
        
        #------------------------------
        # Process TIME data (top row)
        #------------------------------
        ax_time = axes[time_ax_idx]
        
        # Set aspect ratio to make plot square
        ax_time.set_aspect('auto', adjustable='box')
        
        # Set y-axis formatter to show 1 decimal place
        ax_time.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Increase tick label size
        ax_time.tick_params(axis='both', labelsize=12)
        
        # Position of each group of bars on the x-axis
        positions = np.arange(len(ffn_dims))
        width = 0.2  # Width of each bar
        
        # Plot bars for each compiler
        for comp_idx, compiler in enumerate(compilers):
            offset = (comp_idx - 1.5) * width
            normalized_values = []
            
            for i, dim in enumerate(ffn_dims):
                # Get baseline value for normalization
                baseline_value = float(ffn_time[array_size][dim][baseline_compiler])
                
                # Get current compiler value
                compiler_value = float(ffn_time[array_size][dim][compiler])
                
                # Normalize to baseline
                normalized_value = compiler_value / baseline_value
                normalized_values.append(normalized_value)
                
                # Annotate the unnormalized value on top of baseline bar
                if compiler == baseline_compiler:
                    text = f"{baseline_value:.2e}" if baseline_value >= 1000 else f"{baseline_value:.2f}"
                    ax_time.text(positions[i] + offset, normalized_value + 0.05, text, 
                           ha='center', va='bottom', fontsize=7, rotation=45)
            
            # Create bars
            ax_time.bar(positions + offset, normalized_values, 
                    width=width, color=compiler_colors[compiler])
        
        # Add subplot title (only for top row) with larger font
        ax_time.set_title(f"Array Size: {array_size}", fontsize=14)
        
        # Format x-axis (no labels for top row)
        ax_time.set_xticks(positions)
        ax_time.set_xticklabels([])
        
        # Set y-axis range
        max_y_time = max([float(ffn_time[array_size][hd][c]) / 
                     float(ffn_time[array_size][hd][baseline_compiler])
                     for hd in ffn_dims for c in compilers
                     if ffn_time[array_size][hd][c] is not None])
        
        ax_time.set_ylim(0, max_y_time * 1.2)  # 20% padding
        
        # Add horizontal line at y=1.0 to indicate baseline
        ax_time.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add gridlines
        ax_time.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        #--------------------------------
        # Process ENERGY data (bottom row)
        #--------------------------------
        ax_energy = axes[energy_ax_idx]
        
        # Set aspect ratio to make plot square
        ax_energy.set_aspect('auto', adjustable='box')
        
        # Set y-axis formatter to show 1 decimal place
        ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Increase tick label size
        ax_energy.tick_params(axis='both', labelsize=12)
        
        # Plot bars for each compiler
        for comp_idx, compiler in enumerate(compilers):
            offset = (comp_idx - 1.5) * width
            normalized_values = []
            
            for i, dim in enumerate(ffn_dims):
                # Get baseline value for normalization
                baseline_value = float(ffn_energy[array_size][dim][baseline_compiler])
                
                # Get current compiler value
                compiler_value = float(ffn_energy[array_size][dim][compiler])
                
                # Normalize to baseline
                normalized_value = compiler_value / baseline_value
                normalized_values.append(normalized_value)
                
                # Annotate the unnormalized value on top of baseline bar
                if compiler == baseline_compiler:
                    text = f"{baseline_value:.2e}" if baseline_value >= 1000 else f"{baseline_value:.2f}"
                    ax_energy.text(positions[i] + offset, normalized_value + 0.05, text, 
                           ha='center', va='bottom', fontsize=7, rotation=45)
            
            # Create bars 
            ax_energy.bar(positions + offset, normalized_values, 
                    width=width, color=compiler_colors[compiler])
        
        # Format x-axis (labels for bottom row) with power of 2 format
        ax_energy.set_xticks(positions)
        ax_energy.set_xticklabels([format_as_power_of_two(dim) for dim in ffn_dims])
        ax_energy.set_xlabel("FFN Dimension", fontsize=14)
        
        # Set y-axis range
        max_y_energy = max([float(ffn_energy[array_size][hd][c]) / 
                     float(ffn_energy[array_size][hd][baseline_compiler])
                     for hd in ffn_dims for c in compilers
                     if ffn_energy[array_size][hd][c] is not None])
        
        ax_energy.set_ylim(0, max_y_energy * 1.2)  # 20% padding
        
        # Add horizontal line at y=1.0 to indicate baseline
        ax_energy.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Add gridlines
        ax_energy.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Apply tight_layout first to arrange subplots
    plt.tight_layout()
    
    # Then adjust with subplots_adjust for additional spacing
    plt.subplots_adjust(top=0.85, wspace=0.3, hspace=0.3)
    
    # Then add section headers using fig.text 
    left_center = 0.25
    right_center = 0.75
    
    # Create custom legend
    legend_labels = [compiler_names[c] for c in compilers]
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=compiler_colors[c]) for c in compilers]
    
    # Place the legend at the top of the figure
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.03),
               ncol=len(compilers),
               frameon=True,
               fancybox=True,
               shadow=False,
               fontsize=15)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Show the figure
    plt.show()

# Usage
plot_combined_comparison(
    hidden_time=grouped_time_hidden, 
    hidden_energy=grouped_energy_hidden, 
    ffn_time=grouped_time_ffn, 
    ffn_energy=grouped_energy_hidden,  # This should be grouped_energy_ffn in your actual code  
    save_path='combined_comparison.png'
)

# # Save normalized improvements to CSV
# save_normalized_improvements_to_csv(
#     hidden_time=grouped_time_hidden, 
#     hidden_energy=grouped_energy_hidden, 
#     ffn_time=grouped_time_ffn, 
#     ffn_energy=grouped_energy_hidden,  # This should be grouped_energy_ffn in your actual code
#     output_file='normalized_improvements.csv'
# )