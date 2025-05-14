from evaluation.visualize import visualize_multiple_heat_maps
import numpy as np

# Example usage
heatmap1 = np.random.rand(10, 10)
heatmap2 = np.random.rand(10, 10) * 2
heatmap3 = np.random.rand(10, 10) * 3

# Plot the heatmaps with default settings
fig = visualize_multiple_heat_maps(
    [heatmap1, heatmap2, heatmap3],
    titles=["First Map", "Second Map", "Third Map"],
    main_title="My Heatmap Comparison",
    show=True
)

# # Or with logarithmic scaling
# fig_log = visualize_multiple_heat_maps(
#     [heatmap1, heatmap2, heatmap3],
#     log_scale=True,
#     show=True
# )