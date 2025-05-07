import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Set up the figure with 2x2 subplots
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Define the pastel color palette based on the second image
colors = ['#f8c1c1', '#f9f2c1', '#f0ecc0', '#b7c6d1', '#d1e8d0']

# Define the categories and groups for our data
datasets = ['Higgs', 'CD1']
pe_sizes = [1, 2, 4, 8, 16]
methods = ['Sequential (w/o WL)', 'K=0% (Baseline, HWL w/o LASR)', 'K=60%', 'K=80%', 'K=100%']

# Create some sample data
def generate_sample_data(panel, dataset_idx, pe_idx):
    base = (panel + 1) * (dataset_idx + 1) * (pe_idx + 0.5)
    return np.array([base * 0.5, base * 0.9, base * 0.8, base * 0.7, base * 0.6])

# Subplot titles
subplot_titles = ['(a) Cycles', '(b) Total Energy', 
                 '(c) Intra-NoC Accesses', '(d) Inter-NoC Accesses']

# Width of each bar
bar_width = 0.15

# Loop through the subplots
for panel_idx, panel_title in enumerate(subplot_titles):
    ax = fig.add_subplot(gs[panel_idx//2, panel_idx%2])
    
    # Set the y-axis limit based on the panel
    if panel_idx == 3:  # Panel d has a different scale
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 4)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='-', alpha=0.2)
    
    # Add the panel title
    ax.set_title(panel_title, fontsize=12, fontweight='bold')
    
    # Position for each group of bars
    positions = []
    tick_positions = []
    tick_labels = []
    
    # Add a vertical line between datasets
    dataset_separator = len(pe_sizes) - 0.5
    ax.axvline(x=dataset_separator, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Loop through each dataset
    for dataset_idx, dataset in enumerate(datasets):
        offset = dataset_idx * len(pe_sizes)
        
        # Loop through each PE size within the dataset
        for pe_idx, pe_size in enumerate(pe_sizes):
            pos = offset + pe_idx
            tick_positions.append(pos)
            tick_labels.append(str(pe_size))
            
            # Generate data for this specific combination
            data = generate_sample_data(panel_idx, dataset_idx, pe_idx)
            
            # Plot bars for each method
            for method_idx, method in enumerate(methods):
                x_pos = pos + (method_idx - 2) * bar_width
                bar = ax.bar(x_pos, data[method_idx], bar_width, 
                       color=colors[method_idx], edgecolor='black', linewidth=0.5)
                
                # Add value labels on top of each bar
                if panel_idx < 3:  # For panels a, b, c
                    value_text = f"{data[method_idx]:.2f}"
                else:  # For panel d which has larger values
                    value_text = f"{data[method_idx]:.1f}"
                
                ax.text(x_pos, data[method_idx] + 0.05, value_text, 
                        ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add dataset label
        middle_pos = offset + len(pe_sizes) / 2 - 0.5
        ax.text(middle_pos, -0.6, dataset, ha='center', fontsize=10, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Add PE Size label
    ax.text(len(pe_sizes) / 2 - 0.5, -0.3, 'PE Size', ha='center', fontsize=10)
    ax.text(len(pe_sizes) + len(pe_sizes) / 2 - 0.5, -0.3, 'PE Size', ha='center', fontsize=10)
    
    # Add Dataset label
    ax.text(len(pe_sizes) / 2 - 0.5, -0.9, 'Dataset', ha='center', fontsize=10)
    ax.text(len(pe_sizes) + len(pe_sizes) / 2 - 0.5, -0.9, 'Dataset', ha='center', fontsize=10)

# Create a custom legend
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=methods[i]) 
                  for i in range(len(methods))]
fig.legend(handles=legend_elements, loc='upper center', ncol=len(methods), 
          bbox_to_anchor=(0.5, 1.0), fontsize=10)

# Adjust the layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15)

# Show the plot
plt.savefig('multi_panel_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()