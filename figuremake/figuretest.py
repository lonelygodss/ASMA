import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

# Set the figure size and create figure
plt.figure(figsize=(15, 10))

# Define column parallelism values and their colors
parallelism = [32, 128, 512, 2048]
colors = ['#AAAACC', '#AACCDD', '#E6DFB8', '#F2C1B6']

# Define operations
operations = ['filter', 'maxn', 'sort']

# Define data types and sizes
data_configs = [
    ('small', 'INT32'), 
    ('medium', 'INT32'), 
    ('large', 'INT32'),
    ('small', 'FP32'), 
    ('medium', 'FP32'), 
    ('large', 'FP32')
]

# Create a gridspec for better control over the layout
gs = gridspec.GridSpec(2, 3)

# Sample data - in real usage, replace with your actual data
# Structure: data[data_config_index][operation_index][bank_count_index][parallelism_index]
np.random.seed(42)  # For reproducibility

# Generate sample data with appropriate scales for each subplot
# This is just placeholder data - replace with your actual values
data = []
for i in range(6):
    ops_data = []
    for j in range(3):
        # Different scales for different operations and data sizes
        if i in [1, 4] and j in [0]:  # medium data, filter operation
            scale = 4e4
        elif i in [2, 5] and j in [2]:  # large data, sort operation
            scale = 1e7 if i == 2 else 2e9
        elif i in [2, 5] and j in [0]:  # large data, filter operation
            scale = 3e6
        else:
            scale = 8e2
        
        bank_data = []
        for _ in range(3):  # 3 bank counts
            par_data = np.random.rand(4) * scale  # 4 parallelism values
            if j == 1:  # maxn typically shows decreasing trend
                par_data.sort()
                par_data = par_data[::-1]
            bank_data.append(par_data)
        ops_data.append(bank_data)
    data.append(ops_data)

# Bank counts
bank_counts = [16, 32, 64]

# Create subplots
for idx, (size, dtype) in enumerate(data_configs):
    row, col = idx // 3, idx % 3
    
    # Create 3 subplots side by side for each operation
    for op_idx, operation in enumerate(operations):
        ax = plt.subplot(gs[row, col], position=[col/3 + op_idx*0.32, row/2 + 0.05, 0.25, 0.2])
        
        # Set different y-axis scales based on the data
        if (size == 'medium' and operation == 'filter') or (size == 'medium' and operation == 'maxn'):
            plt.ylim(0, 4e4)
            if operation == 'filter':
                ax.text(0.05, 0.9, "4E+4", transform=ax.transAxes, fontsize=9)
        elif size == 'large' and operation == 'sort' and dtype == 'INT32':
            plt.ylim(0, 1e7)
            ax.text(0.05, 0.9, "1E+7", transform=ax.transAxes, fontsize=9)
        elif size == 'large' and operation == 'sort' and dtype == 'FP32':
            plt.ylim(0, 2e9)
            ax.text(0.05, 0.9, "2E+9", transform=ax.transAxes, fontsize=9)
        elif size == 'large' and (operation == 'filter' or operation == 'maxn'):
            plt.ylim(0, 3e6)
            ax.text(0.05, 0.9, "3E+6", transform=ax.transAxes, fontsize=9)
        else:
            plt.ylim(0, 8e2)
            ax.text(0.05, 0.9, "8E+2", transform=ax.transAxes, fontsize=9)
        
        # Get the current subplot data
        subplot_data = data[idx][op_idx]
        
        # Position of each group of bars on the x-axis
        positions = np.arange(len(bank_counts))
        width = 0.2  # Width of each bar
        
        # Plot bars for each parallelism value
        for p_idx, p_val in enumerate(parallelism):
            offset = (p_idx - 1.5) * width
            plt.bar(positions + offset, [subplot_data[i][p_idx] for i in range(len(bank_counts))], 
                   width=width, color=colors[p_idx], label=str(p_val) if op_idx == 0 and idx == 0 else "")
        
        # Add title on top of each subplot
        if row == 0 and col == 0:
            plt.title(operation, y=1.1)
        else:
            plt.title(operation)
        
        # Add x-axis labels only on the bottom row
        if row == 1:
            plt.xticks(positions, bank_counts)
        else:
            plt.xticks(positions, [])
        
        # Add data config label under the middle subplot
        if op_idx == 1:
            plt.xlabel(f"({size},{dtype})")
        
        # Format y-axis with custom scientific notation
        plt.gca().yaxis.set_major_formatter(plt.NullFormatter())  # Remove default y ticks
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.text(-0.2, 0, "0E+0", transform=ax.transAxes, fontsize=9)
        
        # Remove most of the frame
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add y-axis label only on the leftmost subplots
        if op_idx == 0 and row == 0 and col == 0:
            plt.ylabel("runtime(ns)")
        
        # Add "bank count" label only on the bottom left subplot
        if row == 1 and op_idx == 0 and col == 0:
            plt.text(-0.7, -0.3, "bank count", transform=ax.transAxes)

# Add legend at the top
legend_ax = plt.axes([0.3, 0.95, 0.5, 0.03], frameon=False)
legend_ax.axis('off')
legend_ax.text(0, 0, "Column Parallelism:", ha='right', va='center')

for i, (p_val, color) in enumerate(zip(parallelism, colors)):
    rect = plt.Rectangle((0.25 + i*0.15, -0.2), 0.08, 0.4, facecolor=color)
    legend_ax.add_patch(rect)
    legend_ax.text(0.29 + i*0.15, 0, str(p_val), ha='left', va='center')

# Add the (a) label
plt.figtext(0.03, 0.95, "(a)", fontsize=12)

# Adjust the layout
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Save and show the figure
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()