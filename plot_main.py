#%%
from analysis.data_read import SimulationDataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

datapath = './'
filename = 'simulation_results_hidden.csv'
dataloader = SimulationDataLoader(datapath,filename)
# %%
grouped_data = dataloader.prepare_data_for_grouped_comparison()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

def plot_grouped_comparison(grouped_data, metric='Time', save_path=None):
    """
    Create a visualization of grouped comparison data with subplots for each array size.
    
    Parameters:
    -----------
    grouped_data : dict
        Data returned by prepare_data_for_grouped_comparison()
    metric : str
        The metric being visualized ('Time' or 'Energy')
    save_path : str, optional
        Path to save the figure, if provided
    """
    # Extract dimensions from the data
    array_sizes = grouped_data['dimensions']['array_sizes']
    hidden_dims = grouped_data['dimensions']['hidden_dims']
    compilers = grouped_data['dimensions']['compilers']
    
    # Define compiler colors and friendly names
    compiler_colors = {
        'b_b': '#AAAACC',    # Baseline - blue-gray
        's_p': '#AACCDD',    # Parallel - light blue
        's_s': '#E6DFB8',    # Scatter - tan
        's_sp': '#F2C1B6'    # Scatter-Parallel - salmon
    }
    
    compiler_names = {
        'b_b': 'Baseline Compiler',
        's_p': 'Parallel Compiler',
        's_s': 'Scatter Compiler',
        's_sp': 'Scatter-Parallel Compiler'
    }
    
    # Create figure with extra space at the top for the legend
    fig, axes = plt.subplots(1, len(array_sizes), figsize=(15, 7))
    
    # Make axes an array even if there's only one subplot
    if len(array_sizes) == 1:
        axes = [axes]
    
    # Create bar plots for each compiler in each subplot
    for idx, array_size in enumerate(array_sizes):
        ax = axes[idx]
        
        # Position of each group of bars on the x-axis
        positions = np.arange(len(hidden_dims))
        width = 0.2  # Width of each bar
        
        # Store bars for legend
        bar_handles = []
        
        # Plot bars for each compiler
        for comp_idx, compiler in enumerate(compilers):
            offset = (comp_idx - 1.5) * width
            
            # Get values for this compiler across all hidden dimensions
            values = [grouped_data[array_size][hidden_dim][compiler] for hidden_dim in hidden_dims]
            
            # Convert from numpy types to Python native types
            values = [float(v) for v in values]
            
            # Create bars and add to handles for legend
            bars = ax.bar(positions + offset, values, 
                    width=width, color=compiler_colors[compiler], 
                    label=compiler_names[compiler])
            
            bar_handles.append(bars)
        
        # Add subplot title
        ax.set_title(f"Array Size: {array_size}")
        
        # Format x-axis
        ax.set_xticks(positions)
        ax.set_xticklabels([str(dim) for dim in hidden_dims])
        ax.set_xlabel("Hidden Dimension")
        
        # Format y-axis with scientific notation if values are large
        max_value = max([max([float(grouped_data[array_size][hd][c]) 
                              for c in compilers if grouped_data[array_size][hd][c] is not None]) 
                         for hd in hidden_dims])
        
        if max_value > 1000:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Add y-axis label only on the leftmost subplot
        if idx == 0:
            ax.set_ylabel(f"{metric} (arbitrary units)")
        
        # Add gridlines for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend above the subplots
    # This ensures the legend will be visible and properly positioned
    legend_labels = [compiler_names[c] for c in compilers]
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=compiler_colors[c]) for c in compilers]
    
    # Place the legend at the top of the figure
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.98),
               ncol=len(compilers),
               frameon=True,
               fancybox=True,
               shadow=True,
               fontsize=12)
    
    # Add title for the entire figure
    fig.suptitle(f"Comparison of {metric} Across Different Configurations", 
                y=1.05, fontsize=16)
    
    # Adjust layout, but leave space for the legend and title
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Show the figure
    plt.show()

# Usage
plot_grouped_comparison(grouped_data, metric='Time', save_path='compiler_comparison.png')