#%%
from analysis.detailed_metrics_utils import DetailedMetricsLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

# Load data
loader = DetailedMetricsLoader('./pipeline_time.csv')

# Get data grouped by array size
data_by_array_size = loader.prepare_data_by_array_size()

# Prepare data for stacked bar charts (showing Bank-Tile, Intra-Tile, Execution)
stacked_data = loader.prepare_stacked_bar_data()
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.colors as mcolors

def plot_normalized_pipeline_metrics(stacked_data, baseline_compiler='bb', save_path=None):
    """
    Create a visualization with three subplots:
    1. Max times per compiler (log scale, normalized)
    2. Stacked times showing composition with color variations (log scale, normalized)
    3. Scatter plot of Intra-Tile/Bank-Tile vs Execution/Bank-Tile ratios (excluding baseline)
    
    Parameters:
    -----------
    stacked_data : dict
        Data returned by DetailedMetricsLoader.prepare_stacked_bar_data()
    baseline_compiler : str
        The compiler to use as baseline for normalization (default: 'bb')
    save_path : str, optional
        Path to save the figure, if provided
    """
    # Extract dimensions from the data
    array_sizes = sorted(list(stacked_data.keys()))
    first_array_size = array_sizes[0]
    first_compiler = list(stacked_data[first_array_size].keys())[0]
    
    # Get the metrics (assuming all array sizes and compilers have the same metrics)
    metrics = list(stacked_data[first_array_size][first_compiler].keys())
    compilers = list(stacked_data[first_array_size].keys())
    
    # Define compiler colors and friendly names
    compiler_colors = {
        'bb': '#AAAACC',    # Baseline - blue-gray
        'sp': '#AACCDD',    # Parallel - light blue
        'ss': '#E6DFB8',    # Scatter - tan
        'ssp': '#F2C1B6'    # Scatter-Parallel - salmon
    }
    
    # Function to create lighter/darker variations of a color
    def adjust_color(color, factor):
        """Make a color lighter (factor>1) or darker (factor<1)"""
        c = mcolors.to_rgb(color)
        c_adjusted = [min(1.0, max(0.0, x * factor)) for x in c]
        return c_adjusted
    
    # Create color variations for each metric of each compiler
    stage_colors = {}
    for compiler, base_color in compiler_colors.items():
        stage_colors[compiler] = {
            'Bank-Tile': adjust_color(base_color, 1.0),   # Lighter
            'Intra-Tile': adjust_color(base_color, 1.1),                     # Original
            'Excecution': adjust_color(base_color, 1.2)   # Darker
        }
    
    compiler_names = {
        'bb': 'Baseline',
        'sp': 'Proposed-Parallel',
        'ss': 'Proposed-Scatter',
        'ssp': 'Proposed-Mutual'
    }
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set position of bars on x-axis
    x_positions = np.arange(len(array_sizes))
    width = 0.2  # Width of each bar
    
    #------------------------------------
    # SUBPLOT 1: Maximum values (log scale, normalized)
    #------------------------------------
    ax1 = axes[0]
    
    # Set log scale for y-axis
    ax1.set_yscale('linear')
    
    # Set y-axis formatter to show 1 decimal place
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Calculate max values for each compiler and normalize to baseline
    normalized_max_values = []
    baseline_values = []  # To store original baseline values for annotation
    
    for array_size in array_sizes:
        normalized_max_values.append({})
        
        # First, calculate the baseline max value
        baseline_max = max([stacked_data[array_size][baseline_compiler][metric] for metric in metrics])
        baseline_values.append(baseline_max)
        
        # Then normalize each compiler's max value to the baseline
        for compiler in compilers:
            compiler_max = max([stacked_data[array_size][compiler][metric] for metric in metrics])
            normalized_max_values[-1][compiler] = compiler_max / baseline_max
    
    # Plot bars for each compiler
    for idx, compiler in enumerate(compilers):
        # Offset the bars for each compiler
        offset = (idx - 1.5) * width
        
        # Get normalized max values for this compiler across array sizes
        values = [max_data[compiler] for max_data in normalized_max_values]
        
        # Plot the bars
        ax1.bar(x_positions + offset, values, 
                width=width, color=compiler_colors[compiler], 
                label=compiler_names[compiler],edgecolor='black', linewidth=0.5)
        
        # Annotate the unnormalized baseline values
        if compiler == baseline_compiler:
            for i, val in enumerate(baseline_values):
                text = f"{val:.2e}" if val >= 1000 else f"{val:.2f}"
                ax1.text(x_positions[i] + offset, values[i] + 0.05, text, 
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Add horizontal line at y=1.0 to indicate baseline
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Set x-axis labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(size) for size in array_sizes])
    ax1.set_xlabel("Array Size")
    
    # Set labels and title
    ax1.set_ylabel("Normalized Maximum Time\n(relative to baseline)")
    ax1.set_title("Maximum Pipeline Stage Time")
    
    # Add grid for better readability
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    #------------------------------------
    # SUBPLOT 2: Stacked proportions (log scale, normalized)
    #------------------------------------
    ax2 = axes[1]
    
    # Set log scale for y-axis
    ax2.set_yscale('linear')
    
    # Set y-axis formatter to show 1 decimal place
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Create empty list to store bar objects for the color legend
    stage_patches = []
    
    # Calculate total values for baseline (for normalization)
    baseline_totals = []
    for array_size in array_sizes:
        total = sum([stacked_data[array_size][baseline_compiler][metric] for metric in metrics])
        baseline_totals.append(total)
    
    # Plot stacked bars for each compiler
    for idx, compiler in enumerate(compilers):
        # Offset the bars for each compiler
        offset = (idx - 1.5) * width
        
        # For each array size
        for i, array_size in enumerate(array_sizes):
            # Calculate normalization factor for this compiler+array_size
            baseline_total = baseline_totals[i]
            
            # Start bottom at 0
            bottom = 0
            
            # Process each metric to create the stacked bar
            for metric_idx, metric in enumerate(metrics):
                # Get the value for this metric
                value = stacked_data[array_size][compiler][metric]
                
                # Normalize to the baseline total
                normalized_value = value / baseline_total
                
                # Plot the bar segment
                bar = ax2.bar(x_positions[i] + offset, normalized_value, width=width, 
                      bottom=bottom, color=stage_colors[compiler][metric], 
                      edgecolor='black', linewidth=0.5)
                
                # Save first instance for legend
                if idx == 0 and i == 0 and metric_idx == 0:
                    stage_patches.append(plt.Rectangle((0,0), 1, 1, color=stage_colors[compiler][metric]))
                
                # Update bottom for next segment
                bottom += normalized_value
            
            # Annotate the total unnormalized value for baseline
            if compiler == baseline_compiler:
                text = f"{baseline_total:.2e}" if baseline_total >= 1000 else f"{baseline_total:.2f}"
                ax2.text(x_positions[i] + offset, 1.05, text, 
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Add horizontal line at y=1.0 to indicate baseline
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Set x-axis labels
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(size) for size in array_sizes])
    ax2.set_xlabel("Array Size")
    
    # Set labels and title
    ax2.set_ylabel("Normalized Total Time\n(relative to baseline)")
    ax2.set_title("Composition of Pipeline Stages")
    
    # Add grid for better readability
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Create legend for pipeline stages
    ax2.legend([stage_patches[0], 
               plt.Rectangle((0,0), 1, 1, color=stage_colors[compilers[0]]['Intra-Tile']),
               plt.Rectangle((0,0), 1, 1, color=stage_colors[compilers[0]]['Excecution'])], 
              metrics, title="Pipeline Stages", loc='upper right')
    
    #------------------------------------
    # SUBPLOT 3: Scatter plot of ratios (EXCLUDING BASELINE)
    #------------------------------------
    ax3 = axes[2]
    
    # Define marker styles for different array sizes
    markers = {
        1024: 'o',     # Circle
        2048: 's',     # Square
        768: 'd',
        512: '^'       # Triangle (in case there's a 512 array size)
    }
    
    # Set log scale for both axes
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Set axis formatters to show 1 decimal place
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Create scatter plot - EXCLUDING BASELINE ('bb')
    for array_size in array_sizes:
        for compiler in compilers:
            # Skip baseline compiler (bb) for this plot
            if compiler == baseline_compiler:
                continue
                
            # Get values for this compiler and array size
            bank_tile = stacked_data[array_size][compiler]['Bank-Tile']
            intra_tile = stacked_data[array_size][compiler]['Intra-Tile']
            execution = stacked_data[array_size][compiler]['Excecution']
            
            # Skip if bank_tile is zero to avoid division by zero
            if bank_tile == 0:
                continue
                
            # Calculate ratios
            intra_bank_ratio = intra_tile / bank_tile
            exec_bank_ratio = execution / bank_tile
            
            # Add small epsilon to zero values to show them on log scale
            if intra_bank_ratio == 0:
                intra_bank_ratio = 0.01
            
            # Plot the point
            ax3.scatter(intra_bank_ratio, exec_bank_ratio, 
                       color=compiler_colors[compiler], 
                       marker=markers[array_size],
                       s=300,  # Size
                       edgecolors='black', linewidth=0.5,
                       alpha=0.8)



    # Reference lines for subplot 3:
    # 1. Vertical line at x=1 from y=0 to y=1
    vertical_line = Line2D([1, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)
    ax3.add_line(vertical_line)

    # 2. Horizontal line at y=1 from x=0 to x=1
    horizontal_line = Line2D([0, 1], [1, 1], color='gray', linestyle='--', alpha=0.7)
    ax3.add_line(horizontal_line)

    # 3. Diagonal line y=x for x>1
    diagonal_line = Line2D([1, 10], [1, 10], color='gray', linestyle='--', alpha=0.7)
    ax3.add_line(diagonal_line)
    # Add grid for better readability
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax3.set_xlabel("Intra-Tile / Bank-Tile Ratio")
    ax3.set_ylabel("Execution / Bank-Tile Ratio")
    ax3.set_title("Communication-Computation Balance\n(Proposed Methods Only)")
    
    # Add legend for array sizes
    array_size_handles = [plt.Line2D([0], [0], marker=markers[size], color='gray', 
                                    linestyle='None', markersize=8, 
                                    markerfacecolor='gray', markeredgecolor='black') 
                         for size in array_sizes]
    array_size_labels = [f"Array Size: {size}" for size in array_sizes]
    ax3.legend(array_size_handles, array_size_labels, loc='upper left')

    
    # Adjust layout
    plt.tight_layout()
    
    # Create a legend for compilers above the figure
    compiler_handles = [plt.Rectangle((0,0), 1, 1, color=compiler_colors[c]) for c in compilers]
    compiler_labels = [compiler_names[c] for c in compilers]
    
    fig.legend(compiler_handles, compiler_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=len(compilers), frameon=True,
               fancybox=True, shadow=False, fontsize=15)
    
    plt.subplots_adjust(top=0.85)  # Make room for the legend
    
    # Add title for the entire figure
    fig.suptitle("Pipeline Performance Analysis Across Different Array Sizes", 
                y=1.12, fontsize=16)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Show the figure
    plt.show()

# Usage
plot_normalized_pipeline_metrics(stacked_data, baseline_compiler='bb', save_path='pipeline_analysis.png')