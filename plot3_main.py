#%%
from analysis.model_performance_utils import ModelPerformanceLoader

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
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_model_performance(df, save_path=None):
    """
    Create a visualization of model performance data using raw values and log scale,
    with a horizontal layout and pastel colors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model performance data
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
        '100%': 'cadetblue',     # Light blue
        'Proposed': 'peachpuff'  # Light peach
    }
    
    # Define custom legend labels
    legend_mapping = {
        '70%': 'A100 MBU 70%',
        '100%': 'A100 MBU 100%',
        'Proposed': 'ASMA'
    }
    
    # Create figure with 4 subplots in a row (1×4)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # Define subplot positions
    subplot_positions = [
        (0, "Latency (ms)", 1),      # Batch Size 1, Latency
        (1, "Energy (mJ)", 1),       # Batch Size 1, Energy
        (2, "Latency (ms)", 100),    # Batch Size 100, Latency
        (3, "Energy (mJ)", 100)      # Batch Size 100, Energy
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
        
        # Set subplot title
        ax.set_title(f"Batch Size: {batch_size}")
        
        # Format x-axis
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlabel("Model")
        
        # Add y-axis label
        ax.set_ylabel(metric_name)
        
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
               bbox_to_anchor=(0.5, 1.1),
               ncol=len(mem_utils),
               frameon=True,
               fancybox=True,
               shadow=False,
               fontsize=18)
    
    # Add title for the entire figure
    fig.suptitle("Model Performance Across Different Memory Utilizations", 
                y=1.2, fontsize=16)
    
    # Adjust the layout to make room for the title and legend
    plt.subplots_adjust(top=0.8)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the figure
    plt.show()

# Call the function with the dataframe
plot_model_performance(df, save_path='model_performance_comparison.png')