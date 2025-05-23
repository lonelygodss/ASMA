from analysis.model_performance_utils import ModelPerformanceLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Load the data
loader = ModelPerformanceLoader('./model_performance.csv')

# Get data grouped by batch size (with models together within each batch size)
batch_data = loader.prepare_data_by_batch_size()

# Prepare data for latency plotting
latency_data = loader.prepare_data_for_plotting(metric='Latency(ms)')

# Get speedup factors compared to 100% memory utilization
speedup_data = loader.prepare_speedup_data(reference_mem_util='100%')

# Convert to DataFrame for easier manipulation
df = loader.convert_to_dataframe()

def save_model_performance_to_csv(df, reference_mem_util='100%', output_file='model_performance_improvements.csv'):
    """
    Save model performance metrics and improvements to a CSV file, including statistical analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model performance data
    reference_mem_util : str
        The memory utilization to use as reference for normalization (default: '100%')
    output_file : str
        Path to save the output CSV file
    """
    import csv
    import numpy as np
    import pandas as pd
    
    # Get unique values
    batch_sizes = sorted(df['Batch Size'].unique())
    mem_utils = sorted(df['Mem Util'].unique())
    models = sorted(df['Model'].unique())
    
    # Define friendly names for memory utilizations
    mem_util_names = {
        '70%': 'A100 MBU 70%',
        '100%': 'A100 MBU 100%',
        'Proposed': 'ASMA'
    }
    
    # Dictionary to collect statistics for improvements
    stats = {
        'Latency(ms)': {},
        'Energy(mJ)': {}
    }
    
    for mem_util in mem_utils:
        if mem_util != reference_mem_util:
            stats['Latency(ms)'][mem_util] = []
            stats['Energy(mJ)'][mem_util] = []
    
    # Open the file for writing
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        header = ['Batch Size', 'Model', 'Memory Utilization', 'Metric', 'Raw Value', 
                 f'Normalized to {reference_mem_util}', 'Improvement (%)']
        writer.writerow(header)
        
        # Process data for each combination
        for batch_size in batch_sizes:
            for model in models:
                # Get reference values for normalization
                ref_data = df[(df['Batch Size'] == batch_size) & 
                             (df['Model'] == model) & 
                             (df['Mem Util'] == reference_mem_util)]
                
                if ref_data.empty:
                    continue  # Skip if reference data is missing
                
                ref_latency = ref_data['Latency(ms)'].values[0]
                ref_energy = ref_data['Energy(mJ)'].values[0]
                
                for mem_util in mem_utils:
                    # Get data for this configuration
                    config_data = df[(df['Batch Size'] == batch_size) & 
                                   (df['Model'] == model) & 
                                   (df['Mem Util'] == mem_util)]
                    
                    if config_data.empty:
                        continue  # Skip if configuration data is missing
                    
                    # Process latency
                    raw_latency = config_data['Latency(ms)'].values[0]
                    norm_latency = raw_latency / ref_latency
                    latency_improvement = (1 - norm_latency) * 100
                    
                    # Write latency row
                    writer.writerow([
                        batch_size,
                        model,
                        mem_util_names[mem_util],
                        'Latency',
                        f"{raw_latency:.6f}",
                        f"{norm_latency:.6f}",
                        f"{latency_improvement:.2f}%"
                    ])
                    
                    # Collect statistics (except for reference)
                    if mem_util != reference_mem_util:
                        stats['Latency(ms)'][mem_util].append(norm_latency)
                    
                    # Process energy
                    raw_energy = config_data['Energy(mJ)'].values[0]
                    norm_energy = raw_energy / ref_energy
                    energy_improvement = (1 - norm_energy) * 100
                    
                    # Write energy row
                    writer.writerow([
                        batch_size,
                        model,
                        mem_util_names[mem_util],
                        'Energy',
                        f"{raw_energy:.6f}",
                        f"{norm_energy:.6f}",
                        f"{energy_improvement:.2f}%"
                    ])
                    
                    # Collect statistics (except for reference)
                    if mem_util != reference_mem_util:
                        stats['Energy(mJ)'][mem_util].append(norm_energy)
        
        # Add empty row as separator
        writer.writerow([])
        
        # Write statistics header
        writer.writerow(['STATISTICS (NORMALIZED TO ' + reference_mem_util + ')'])
        writer.writerow(['Memory Utilization', 'Metric', 'Average', 'Min', 'Max', 'Range'])
        
        # Calculate and write statistics
        for metric in ['Latency(ms)', 'Energy(mJ)']:
            metric_display = 'Latency' if 'Latency' in metric else 'Energy'
            
            for mem_util in mem_utils:
                if mem_util != reference_mem_util:
                    values = stats[metric][mem_util]
                    
                    if values:  # Make sure we have data
                        avg = np.mean(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        range_val = max_val - min_val
                        
                        writer.writerow([
                            mem_util_names[mem_util],
                            metric_display,
                            f"{avg:.6f}",
                            f"{min_val:.6f}",
                            f"{max_val:.6f}",
                            f"{range_val:.6f}"
                        ])
        
        # Add empty row as separator
        writer.writerow([])
        
        # Write improvement statistics header
        writer.writerow(['IMPROVEMENT STATISTICS (RELATIVE TO ' + reference_mem_util + ')'])
        writer.writerow(['Memory Utilization', 'Metric', 'Average Improvement', 'Best Improvement', 'Worst Improvement', 'Improvement Range'])
        
        # Calculate and write improvement statistics
        for metric in ['Latency(ms)', 'Energy(mJ)']:
            metric_display = 'Latency' if 'Latency' in metric else 'Energy'
            
            for mem_util in mem_utils:
                if mem_util != reference_mem_util:
                    values = stats[metric][mem_util]
                    
                    if values:  # Make sure we have data
                        # Calculate improvement percentages
                        improvements = [(1 - val) * 100 for val in values]
                        
                        avg_improvement = np.mean(improvements)
                        best_improvement = np.max(improvements)
                        worst_improvement = np.min(improvements)
                        range_val = best_improvement - worst_improvement
                        
                        writer.writerow([
                            mem_util_names[mem_util],
                            metric_display,
                            f"{avg_improvement:.2f}%",
                            f"{best_improvement:.2f}%",
                            f"{worst_improvement:.2f}%",
                            f"{range_val:.2f}%"
                        ])
        
        # Add empty row as separator
        writer.writerow([])
        
        # Write batch size comparison header
        writer.writerow(['BATCH SIZE COMPARISON'])
        writer.writerow(['Memory Utilization', 'Model', 'Metric', 'Small Batch (1) Value', 'Large Batch (100) Value', 'Scaling Factor'])
        
        # Calculate scaling between batch sizes
        for mem_util in mem_utils:
            for model in models:
                # Get data for batch size 1
                small_batch = df[(df['Batch Size'] == 1) & 
                                (df['Model'] == model) & 
                                (df['Mem Util'] == mem_util)]
                
                # Get data for batch size 100
                large_batch = df[(df['Batch Size'] == 100) & 
                                (df['Model'] == model) & 
                                (df['Mem Util'] == mem_util)]
                
                if small_batch.empty or large_batch.empty:
                    continue  # Skip if data is missing
                
                # Process latency
                small_latency = small_batch['Latency(ms)'].values[0]
                large_latency = large_batch['Latency(ms)'].values[0]
                latency_scaling = large_latency / small_latency
                
                writer.writerow([
                    mem_util_names[mem_util],
                    model,
                    'Latency',
                    f"{small_latency:.6f}",
                    f"{large_latency:.6f}",
                    f"{latency_scaling:.6f}"
                ])
                
                # Process energy
                small_energy = small_batch['Energy(mJ)'].values[0]
                large_energy = large_batch['Energy(mJ)'].values[0]
                energy_scaling = large_energy / small_energy
                
                writer.writerow([
                    mem_util_names[mem_util],
                    model,
                    'Energy',
                    f"{small_energy:.6f}",
                    f"{large_energy:.6f}",
                    f"{energy_scaling:.6f}"
                ])
    
    print(f"Model performance metrics, improvements, and statistics saved to {output_file}")


def plot_model_performance(df, textsize=8, save_path=None):
    """
    Create a visualization of model performance data using raw values and log scale,
    with a 2x2 layout and pastel colors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model performance data
    textsize : int
        Font size for all text elements
    save_path : str, optional
        Path to save the figure, if provided
    """
    # Get unique values
    batch_sizes = df['Batch Size'].unique()
    mem_utils = df['Mem Util'].unique()
    models = df['Model'].unique()
    
    # Define pastel colors for memory utilization (based on reference image)
    mem_util_colors = {
        '70%': 'thistle',      # Light pink
        '100%': 'cadetblue',   # Light blue
        'Proposed': 'peachpuff'  # Light peach
    }
    
    # Define custom legend labels
    legend_mapping = {
        '70%': 'A100 MBU 70%',
        '100%': 'A100 MBU 100%',
        'Proposed': 'ASMA'
    }
    
    # Create figure with 2x2 subplots layout
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    # Flatten the 2D array of axes for easier indexing
    axes = axes.flatten()
    
    # Define subplot positions for 2x2 layout
    subplot_positions = [
        (0, "Latency (ms)", 1),      # Top-left: Batch Size 1, Latency
        (1, "Energy (mJ)", 1),       # Top-right: Batch Size 1, Energy
        (2, "Latency (ms)", 100),    # Bottom-left: Batch Size 100, Latency
        (3, "Energy (mJ)", 100)      # Bottom-right: Batch Size 100, Energy
    ]
    
    # Process each subplot
    for subplot_idx, (ax_idx, metric_name, batch_size) in enumerate(subplot_positions):
        # Get the axis
        ax = axes[ax_idx]
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        # Determine which metric to plot
        metric = 'Latency(ms)' if 'Latency' in metric_name else 'Energy(mJ)'
        
        # Filter data for this batch size
        batch_df = df[df['Batch Size'] == batch_size]
        
        # Position of each group of bars on the x-axis
        positions = np.arange(len(models))
        width = 0.25  # Width of each bar
        
        # Plot bars for each memory utilization
        for mem_idx, mem_util in enumerate(mem_utils):
            offset = (mem_idx - 1) * width
            
            # Filter data for this memory utilization
            mem_df = batch_df[batch_df['Mem Util'] == mem_util]
            
            # Get raw values
            values = []
            for model in models:
                model_df = mem_df[mem_df['Model'] == model]
                if not model_df.empty:
                    value = model_df[metric].values[0]
                    values.append(value)
                else:
                    values.append(None)  # Handle missing data
            
            # Create bars
            ax.bar(positions + offset, values, 
                  width=width, color=mem_util_colors[mem_util], 
                  label=legend_mapping[mem_util] if subplot_idx == 0 else "")
        
        # Set subplot title with larger font
        ax.set_title(f"Batch Size: {batch_size}", fontsize=textsize)
        
        # Format x-axis with larger font
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=textsize)
        
        # Set y-axis labels and ticks with larger font
        ax.set_ylabel(metric_name, fontsize=textsize)
        ax.tick_params(axis='y', labelsize=textsize)
        
        # Add gridlines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Apply tight_layout
    plt.tight_layout()
    
    # Create custom legend with new labels
    legend_labels = [legend_mapping[mu] for mu in mem_utils]
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=mem_util_colors[mu]) for mu in mem_utils]
    
    # Place the legend at the top of the figure
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.05),
               ncol=len(mem_utils),
               frameon=True,
               fancybox=True,
               shadow=False,
               fontsize=textsize)
    
    # Add title for the entire figure
    fig.suptitle("Model Performance Across Different Memory Utilizations", 
                y=1.1, fontsize=textsize+2)
    
    # Adjust the layout to make room for the title and legend
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Show the figure
    plt.show()

# # Call the function with the dataframe
# plot_model_performance(df, textsize=11, save_path='model_performance_comparison.png')

# Save model performance metrics to CSV
save_model_performance_to_csv(
    df,
    reference_mem_util='100%',
    output_file='model_performance_improvements.csv'
)