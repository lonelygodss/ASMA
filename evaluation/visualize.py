import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import List, Optional, Tuple, Union

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



import copy

def visualize_multiple_heat_maps(heat_maps: List[np.ndarray], 
                              titles: Optional[List[str]] = None, 
                              main_title: str = "Dataflow Heat Maps",
                              show: bool = False, 
                              log_scale: bool = False,
                              figsize: Tuple[int, int] = (15, 5),
                              cmap: Union[str, matplotlib.colors.Colormap] = 'bwr',
                              zeros_black: bool = True):
    """
    Visualize multiple heat maps as subplots using matplotlib with a shared colorbar
    
    Parameters
    ----------
    heat_maps : List[np.ndarray]
        List of heat map data arrays to visualize (designed for 3 heatmaps)
    titles : List[str], optional
        List of titles for each heat map subplot
    main_title : str
        The main title for the entire figure
    show : bool
        Whether to call plt.show() after creating the heat maps
    log_scale : bool
        Whether to use logarithmic scaling for color mapping
    figsize : Tuple[int, int]
        Figure size as (width, height)
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for the heatmaps (e.g., 'hot', 'viridis', 'coolwarm')
    zeros_black : bool
        Whether to render zero values as black regardless of colormap
    
    Returns
    -------
    fig:
        The figure object containing the heat maps
    """
    num_maps = len(heat_maps)
    
    # Set default titles if not provided
    if titles is None:
        titles = [f"Heat Map {i+1}" for i in range(num_maps)]
    elif len(titles) < num_maps:
        # Extend titles if too few provided
        titles.extend([f"Heat Map {i+1}" for i in range(len(titles), num_maps)])
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, num_maps, figsize=figsize)
    
    # If only one heatmap, axes won't be an array
    if num_maps == 1:
        axes = [axes]
    
    # Find global min and max for consistent coloring
    vmin = min([np.min(hm) for hm in heat_maps])
    vmax = max([np.max(hm) for hm in heat_maps])
    
    # Create a normalization that will be used for all heatmaps
    if log_scale and vmax > 0:
        # Find minimum non-zero value across all heatmaps
        min_nonzero = float('inf')
        for hm in heat_maps:
            if np.any(hm > 0):
                min_nonzero = min(min_nonzero, np.min(hm[hm > 0]))
                
        if min_nonzero != float('inf'):
            # Use 1/10 of minimum non-zero value as the lower bound for log scale
            min_nonzero = min_nonzero / 10
            norm = matplotlib.colors.LogNorm(vmin=min_nonzero, vmax=vmax)
            cbar_label = "log10(value)"
            scale_type = "Log Scale"
        else:
            # If no positive values found, fall back to linear scale
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar_label = "value"
            scale_type = "Linear Scale (No positive values for log scale)"
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "value"
        scale_type = "Linear Scale"
    
    # Create a modified colormap if zeros_black is True
    if zeros_black:
        # Get the base colormap
        if isinstance(cmap, str):
            base_cmap = plt.cm.get_cmap(cmap)
        else:
            base_cmap = cmap
            
        # Create a new colormap with black for zeros
        colors = copy.copy(base_cmap(np.linspace(0, 1, 256)))
        
        # Create a masked array colormap
        cmap_with_black = matplotlib.colors.ListedColormap(colors)
        cmap_with_black.set_bad(color='black')
    
    # Create each subplot
    for i, (heatmap, title, ax) in enumerate(zip(heat_maps, titles, axes)):
        if zeros_black:
            # Create a masked array where zeros are masked
            masked_data = np.ma.masked_where(heatmap == 0, heatmap)
            
            # For log scale, replace zeros with minimum non-zero value / 10
            if log_scale and scale_type == "Log Scale":
                plot_data = masked_data
            else:
                plot_data = masked_data
                
            # Use the masked array and modified colormap
            im = ax.imshow(plot_data, cmap=cmap_with_black, interpolation='nearest', norm=norm)
        else:
            # Original code without black zeros
            if log_scale and scale_type == "Log Scale":
                plot_data = heatmap.copy()
                plot_data[plot_data == 0] = min_nonzero
            else:
                plot_data = heatmap
                
            im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest', norm=norm)
            
        ax.set_title(title)
        ax.set_xlabel("Module Index")
        ax.set_ylabel("Module Index")
    
    # Add a colorbar that applies to all subplots
    cbar = fig.colorbar(im, ax=axes.tolist() if num_maps > 1 else axes, label=cbar_label)
    
    # Set the main title
    fig.suptitle(f"{main_title} ({scale_type})")
    
    # # Adjust layout
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
    
    if show:
        plt.show()
    
    return fig