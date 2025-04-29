import numpy as np
import matplotlib.pyplot as plt

def visualize_heat_map(heat_map: np.ndarray, title: str = "Dataflow Heat Map", show: bool = False, log_scale: bool = False):
    """
    Visualize the heat map using matplotlib
    
    Parameters
    ----------
    heat_map : np.ndarray
        The heat map data to visualize
    title : str
        The title of the heat map
    show : bool
        Whether to call plt.show() after creating the heat map
    log_scale : bool
        Whether to use logarithmic scaling for color mapping
    
    Returns
    -------
    fig:
        The figure object containing the heat map
    """
    fig = plt.figure()
    
    # Apply logarithmic transformation if requested
    if log_scale and np.any(heat_map > 0):
        # Handle zeros by replacing with minimum non-zero value divided by 10
        # to avoid log(0) which is undefined
        min_nonzero = np.min(heat_map[heat_map > 0]) / 10
        log_data = heat_map.copy()
        log_data[log_data == 0] = min_nonzero
        log_data = np.log10(log_data)
        plt.imshow(log_data, cmap='hot', interpolation='nearest')
        plt.colorbar(label="log10(value)")
        plt.title(f"{title} (Log Scale)")
    else:
        plt.imshow(heat_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label="value")
        plt.title(f"{title} (Linear Scale)")
    
    plt.xlabel("Module Index")
    plt.ylabel("Module Index")
    if show:
        plt.show()
    return fig
