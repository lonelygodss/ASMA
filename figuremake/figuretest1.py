import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Set the style to be clean and professional
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

# Sample data - replace with your actual data
# Helper function to generate sample data (replace with your real data)
def generate_sample_data():
    # Create structure to hold all the data
    data = {}
    operations = ['filter', 'maxn', 'sort']
    sizes = ['small', 'medium', 'large']
    types = ['INT32', 'FP32']
    bank_counts = [16, 32, 64]
    parallelism = [32, 128, 512, 2048]
    
    # Generate random sample data with appropriate scales
    np.random.seed(42)  # for reproducibility
    
    for op in operations:
        data[op] = {}
        for size in sizes:
            data[op][size] = {}
            for dtype in types:
                data[op][size][dtype] = {}
                for bank in bank_counts:
                    # Scale factors to make data look realistic
                    scale = 1
                    if op == 'filter':
                        scale = np.random.uniform(4e2, 8e2) if size == 'small' else \
                                np.random.uniform(4e4, 8e4) if size == 'medium' else \
                                np.random.uniform(3e6, 5e6)
                    elif op == 'maxn':
                        scale = np.random.uniform(4e2, 8e2) if size == 'small' else \
                                np.random.uniform(4e4, 8e4) if size == 'medium' else \
                                np.random.uniform(3e6, 5e6)
                    elif op == 'sort':
                        scale = np.random.uniform(4e2, 8e2) if size == 'small' else \
                                np.random.uniform(4e4, 8e4) if size == 'medium' else \
                                np.random.uniform(1e7, 2e9) if size == 'large' else 0
                    
                    # Create values for each parallelism setting
                    values = []
                    for p in parallelism:
                        if p == 32:
                            values.append(scale * np.random.uniform(0.8, 1.0))
                        else:
                            # Lower parallelism values often show better performance
                            factor = 0.3 if p == 128 else 0.2 if p == 512 else 0.15
                            values.append(scale * factor * np.random.uniform(0.8, 1.2))
                    
                    data[op][size][dtype][bank] = values
    
    return data

# Generate or import your data
data = generate_sample_data()

# Create figure and grid layout
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# Define the operations, sizes, and types
operations = ['filter', 'maxn', 'sort']
sizes = ['small', 'medium', 'large']
types = ['INT32', 'FP32']
bank_counts = [16, 32, 64]
parallelism = [32, 128, 512, 2048]
colors = ['#B2B2D9', '#A8C0D9', '#E8DDB5', '#E8C5B5']  # Colors matching the image

# Add title with a label (a)
fig.text(0.02, 0.98, '(a)', fontsize=14, fontweight='bold')
fig.text(0.07, 0.98, 'runtime(ns)', fontsize=14)

# Add legend at the top
legend_ax = fig.add_axes([0.3, 0.94, 0.65, 0.05])
legend_ax.axis('off')
legend_ax.text(0, 0.5, 'Column Parallelism:', va='center', ha='right', fontsize=12)

for i, (p, c) in enumerate(zip(parallelism, colors)):
    legend_ax.add_patch(plt.Rectangle((0.05 + i*0.2, 0.3), 0.04, 0.4, facecolor=c))
    legend_ax.text(0.1 + i*0.2, 0.5, str(p), va='center', fontsize=12)

# Y-scale ranges for each subplot
y_scales = {
    ('INT32', 'small'): (0, 8e2, ['0E+0', '4E+2', '8E+2']),
    ('INT32', 'medium'): (0, 4e4, ['0E+0', '4E+4']),
    ('INT32', 'large'): (0, 3e6, ['0E+0', '3E+6']),
    ('FP32', 'small'): (0, 8e2, ['0E+0', '4E+2', '8E+2']),
    ('FP32', 'medium'): (0, 4e5, ['0E+0', '4E+5']),
    ('FP32', 'large'): (0, 2e9, ['0E+0', '1E+9', '2E+9'])
}

# Plot the data
for row, dtype in enumerate(types):
    for col, size in enumerate(sizes):
        # Create subplot for each size and type
        ax = fig.add_subplot(gs[row, col])
        
        # Set positions for the operations
        op_positions = [0, 1, 2]  # Positions for filter, maxn, sort
        width = 0.25  # Width of each group of bars for a bank count
        bar_width = width / len(parallelism)  # Width of individual bars
        
        # Get y-scale for this subplot
        y_min, y_max, y_labels = y_scales[(dtype, size)]
        ax.set_ylim(y_min, y_max)
        
        # Plot each operation group
        for op_idx, op in enumerate(operations):
            # Add operation labels at the top of each group
            ax.text(op_positions[op_idx], y_max * 1.05, op, 
                   ha='center', va='bottom', fontsize=10, color='navy')
            
            # Plot bars for each bank count within this operation
            for bank_idx, bank in enumerate(bank_counts):
                # Calculate position of this bank's group
                x_pos = op_positions[op_idx] - width + bank_idx * width
                
                # Get values for this specific configuration
                values = data[op][size][dtype][bank]
                
                # Plot each bar with appropriate color
                for p_idx, (val, color) in enumerate(zip(values, colors)):
                    ax.bar(x_pos + p_idx * bar_width, val, bar_width, 
                          color=color, edgecolor='none')
        
        # Set custom y-axis ticks and labels
        if len(y_labels) > 0:
            ax.set_yticks(np.linspace(y_min, y_max, len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        # Add bank count labels at the bottom x-axis
        for bank_idx, bank in enumerate(bank_counts):
            ax.text(1 + bank_idx * 0.3 - 0.3, -0.1, str(bank), 
                   transform=ax.transAxes, ha='center', fontsize=10)
        
        # Add subplot title
        ax.set_title(f'({size},{dtype})', fontsize=12, pad=10)
        
        # Remove x-axis ticks
        ax.set_xticks([])
        
        # Remove right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add bank count label at the bottom of the first subplot in second row
        if row == 1 and col == 0:
            ax.text(0.5, -0.3, 'bank count', transform=ax.transAxes,
                   ha='center', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to make room for the top legend
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()