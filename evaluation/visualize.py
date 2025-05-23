import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Optional, Tuple, Union,List
import copy
import csv
from typing import List, Dict, Any


def save_heatmap_statistics_to_csv(
    heatmaps: List[np.ndarray],
    heatmap_names: List[str],
    csv_filepath: str
) -> bool:
    """
    Calculates various statistics for a list of heatmaps and saves them to a CSV file.
    Core statistics (average, max, min, std, sum, median) are calculated ONLY
    on non-zero, finite elements.

    The statistics include:
    - Average Value (Finite Non-Zero)
    - Maximum Value (Finite Non-Zero)
    - Minimum Value (Finite Non-Zero)
    - Standard Deviation (Finite Non-Zero)
    - Sum of Values (Finite Non-Zero)
    - Median Value (Finite Non-Zero)
    - Total Elements (in the original heatmap)
    - Zero Elements (count of elements exactly 0)
    - Non-Zero Finite Elements (count)
    - Percentage Zeros (%) (based on total elements)
    - NaN Count (in the original heatmap)
    - Inf Count (in the original heatmap)

    Parameters
    ----------
    heatmaps : List[np.ndarray]
        A list of heatmap data arrays.
    heatmap_names : List[str]
        A list of names corresponding to each heatmap. Must be the same length as heatmaps.
    csv_filepath : str
        The full path (including filename, e.g., "heatmap_stats.csv")
        where the CSV file will be saved.

    Returns
    -------
    bool
        True if the CSV was saved successfully, False otherwise.
    """
    if not heatmaps:
        print("Error: No heatmaps provided to calculate statistics.")
        return False

    if len(heatmaps) != len(heatmap_names):
        print("Error: The number of heatmaps and heatmap_names must be the same.")
        return False

    all_stats_data = []
    
    # Define fieldnames - updated for clarity on non-zero stats
    fieldnames = [
        "Heatmap Name",
        "Average Value (Finite Non-Zero)", "Maximum Value (Finite Non-Zero)", "Minimum Value (Finite Non-Zero)",
        "Standard Deviation (Finite Non-Zero)", "Sum of Values (Finite Non-Zero)", "Median Value (Finite Non-Zero)",
        "Count of Finite Non-Zero Elements", # Renamed for clarity, was "Non-Zero Finite Elements"
        "Total Elements", "Zero Elements (value == 0)",
        "Percentage Zeros (%)",
        "NaN Count", "Inf Count", "Error"
    ]

    for i, heatmap in enumerate(heatmaps):
        name = heatmap_names[i]
        stats_dict: Dict[str, Any] = {"Heatmap Name": name}

        if not isinstance(heatmap, np.ndarray):
            print(f"Warning: Item '{name}' is not a NumPy array. Skipping.")
            stats_dict.update({key: (0 if key in ["Total Elements", "Zero Elements (value == 0)", "Count of Finite Non-Zero Elements", "NaN Count", "Inf Count"] else np.nan) for key in fieldnames if key != "Heatmap Name"})
            stats_dict["Error"] = "Not a NumPy array"
            all_stats_data.append(stats_dict)
            continue
        
        if heatmap.size == 0:
            stats_dict.update({key: (0 if key in ["Total Elements", "Zero Elements (value == 0)", "Count of Finite Non-Zero Elements", "NaN Count", "Inf Count"] else np.nan) for key in fieldnames if key != "Heatmap Name"})
            stats_dict["Percentage Zeros (%)"] = 100.0 # or np.nan
        else:
            # General stats for the whole heatmap
            num_nans = np.sum(np.isnan(heatmap))
            num_infs = np.sum(np.isinf(heatmap))
            num_zeros = np.sum(heatmap == 0) # Includes cases where NaN might be numerically zero if not handled by np.isnan first

            stats_dict.update({
                "Total Elements": heatmap.size,
                "Zero Elements (value == 0)": num_zeros,
                "Percentage Zeros (%)": (num_zeros / heatmap.size) * 100.0 if heatmap.size > 0 else np.nan,
                "NaN Count": num_nans,
                "Inf Count": num_infs,
            })

            # Filter for finite, non-zero elements
            finite_non_zero_elements = heatmap[np.isfinite(heatmap) & (heatmap != 0)]
            count_finite_non_zero = finite_non_zero_elements.size
            stats_dict["Count of Finite Non-Zero Elements"] = count_finite_non_zero

            if count_finite_non_zero > 0:
                stats_dict.update({
                    "Average Value (Finite Non-Zero)": np.mean(finite_non_zero_elements), # np.nanmean not needed as NaNs are already filtered by isfinite
                    "Maximum Value (Finite Non-Zero)": np.max(finite_non_zero_elements),
                    "Minimum Value (Finite Non-Zero)": np.min(finite_non_zero_elements),
                    "Standard Deviation (Finite Non-Zero)": np.std(finite_non_zero_elements),
                    "Sum of Values (Finite Non-Zero)": np.sum(finite_non_zero_elements),
                    "Median Value (Finite Non-Zero)": np.median(finite_non_zero_elements),
                })
            else: # No finite non-zero elements
                stats_dict.update({
                    "Average Value (Finite Non-Zero)": np.nan,
                    "Maximum Value (Finite Non-Zero)": np.nan,
                    "Minimum Value (Finite Non-Zero)": np.nan,
                    "Standard Deviation (Finite Non-Zero)": np.nan,
                    "Sum of Values (Finite Non-Zero)": 0.0, # Sum of an empty set is 0
                    "Median Value (Finite Non-Zero)": np.nan,
                })
        
        # Ensure all fieldnames have a value, even if it's NaN or a default
        for field in fieldnames:
            if field not in stats_dict:
                # This might happen if an error occurred early or a specific stat wasn't applicable
                stats_dict[field] = np.nan # Or a more specific default if needed
        
        all_stats_data.append(stats_dict)

    if not all_stats_data:
        print("No valid statistics were generated.")
        return False
    
    # Ensure all keys from actual data are in fieldnames (dynamic update not really needed if fieldnames are predefined well)
    # Pass extrasaction='ignore' to DictWriter to handle any unexpected keys during development,
    # but for production, ideally, all_stats_data will strictly adhere to `fieldnames`.

    try:
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_stats_data)
        print(f"Heatmap statistics (non-zero focus) successfully saved to {csv_filepath}")
        return True
    except IOError as e:
        print(f"Error: Could not write to CSV file at {csv_filepath}. Reason: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        return False

def visualize_heat_map(heat_map: np.ndarray,
                       title: str = "Baseline",
                       show: bool = False,
                       log_scale: bool = False,
                       figsize: Tuple[int, int] = (5, 5),
                       cmap: Union[str, matplotlib.colors.Colormap] = 'coolwarm',
                       zeros_black: bool = True,
                       save_path = None):
    """
    Visualize a single heat map using matplotlib
    
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
    figsize : Tuple[int, int]
        Figure size as (width, height)
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for the heatmap (e.g., 'hot', 'viridis', 'coolwarm')
    zeros_black : bool
        Whether to render zero values as black regardless of colormap
    
    Returns
    -------
    fig:
        The figure object containing the heat map
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find min and max for coloring
    vmin = np.min(heat_map)
    vmax = np.max(heat_map)
    
    # Create normalization for coloring
    if log_scale and vmax > 0:
        if np.any(heat_map > 0):
            min_nonzero = np.min(heat_map[heat_map > 0])
            # Use 1/10 of minimum non-zero value as the lower bound for log scale
            min_nonzero = min_nonzero / 10
            norm = matplotlib.colors.LogNorm(vmin=min_nonzero, vmax=vmax)
            # cbar_label = "log10(value)"
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
        
        # Create a masked array where zeros are masked
        masked_data = np.ma.masked_where(heat_map == 0, heat_map)
        
        # For log scale, we need to be careful with zeros
        if log_scale and scale_type == "Log Scale":
            plot_data = masked_data
        else:
            plot_data = masked_data
            
        # Use the masked array and modified colormap
        im = ax.imshow(plot_data, cmap=cmap_with_black, interpolation='nearest', norm=norm)
    else:
        # Original approach without black zeros
        if log_scale and scale_type == "Log Scale":
            plot_data = heat_map.copy()
            plot_data[plot_data == 0] = min_nonzero
        else:
            plot_data = heat_map
            
        im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest', norm=norm)
        
    # Add labels and title
    ax.set_title(f"{title} ",fontsize = 16)
    ax.set_xlabel("src",fontsize = 16)
    ax.set_ylabel("dest",fontsize = 16)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    
    # Adjust layout
    # plt.tight_layout()
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
                              cmap: Union[str, matplotlib.colors.Colormap] = 'coolwarm',
                              zeros_black: bool = True,
                              save_path = None):
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
    
    titles = ["Standard-Parallel","Scatter-Parallel","Mutual-Parallel"]

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
            
        ax.set_title(title,fontsize = 18)
        ax.set_xlabel("src",fontsize = 18)
        if i==0 :ax.set_ylabel("dest",fontsize = 18)
    
    # Add a colorbar that applies to all subplots
    cbar = fig.colorbar(im, ax=axes.tolist() if num_maps > 1 else axes, label=cbar_label)
    
    # Set the main title
    fig.suptitle(f"{main_title} ")
    
    # # Adjust layout
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig

def visualize_combined_heat_maps(
    baseline_heatmap: np.ndarray,
    comparison_heatmaps: List[np.ndarray],
    baseline_title: str = "Baseline",
    comparison_titles: Optional[List[str]] = None,
    main_title: str = "Comparative Heatmap Analysis",
    show: bool = False,
    log_scale: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Union[str, matplotlib.colors.Colormap] = 'coolwarm',
    zeros_black: bool = True,
    save_path: Optional[str] = None,
    label_fontsize: int = 26,
    title_fontsize: int = 26
):
    """
    Visualize a baseline heatmap and multiple comparison heatmaps in a single figure.
    The baseline heatmap has its own colorbar.
    The comparison heatmaps share a single, common colorbar.
    All subplots are arranged in a single row.

    Parameters
    ----------
    baseline_heatmap : np.ndarray
        The heatmap data for the baseline plot.
    comparison_heatmaps : List[np.ndarray]
        List of heatmap data arrays for comparison plots.
    baseline_title : str
        Title for the baseline heatmap subplot.
    comparison_titles : List[str], optional
        List of titles for the comparison heatmap subplots.
    main_title : str
        The main title for the entire figure.
    show : bool
        Whether to call plt.show() after creating the heatmaps.
    log_scale : bool
        Whether to use logarithmic scaling for color mapping (applies to all heatmaps).
    figsize : Tuple[int, int], optional
        Figure size. If None, a default is calculated.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for all heatmaps.
    zeros_black : bool
        Whether to render zero values as black.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    label_fontsize : int
        Fontsize for x and y labels.
    title_fontsize : int
        Fontsize for subplot titles.
    """
    num_comparison_maps = len(comparison_heatmaps)
    total_subplots = 1 + num_comparison_maps

    if comparison_titles is None:
        comparison_titles = [f"Comparison {i+1}" for i in range(num_comparison_maps)]
    elif len(comparison_titles) < num_comparison_maps:
        comparison_titles.extend([f"Comparison {i+1}" for i in range(len(comparison_titles), num_comparison_maps)])
    
    # Override with user's specific titles if they called the old function
    if num_comparison_maps == 3 and comparison_titles == [f"Comparison {i+1}" for i in range(num_comparison_maps)]:
         # This is a heuristic, if titles were default and there are 3 maps, use the specific ones.
         # A better way would be to pass titles explicitly when calling this new function.
        user_comparison_titles = ["Standard-Parallel","Scatter-Parallel","Mutual-Parallel"]
        if len(user_comparison_titles) == num_comparison_maps:
            comparison_titles = user_comparison_titles

    if figsize is None:
        figsize = (6 * total_subplots, 5.5) # Adjust base width per subplot as needed

    fig, axes = plt.subplots(1, total_subplots, figsize=figsize, squeeze=False)
    axes = axes.flatten() # Ensure axes is a 1D array

    # --- 1. Baseline Heatmap (axes[0]) ---
    ax_base = axes[0]
    
    # Get base colormap object
    if isinstance(cmap, str):
        base_cmap_obj = plt.cm.get_cmap(cmap)
    else:
        base_cmap_obj = cmap

    # Normalization for baseline
    vmin_base = np.min(baseline_heatmap)
    vmax_base = np.max(baseline_heatmap)
    min_nonzero_base_for_norm = vmax_base # Default if no positive
    
    if log_scale and np.any(baseline_heatmap > 0):
        min_val = np.min(baseline_heatmap[baseline_heatmap > 0])
        min_nonzero_base_for_norm = min_val / 10 if min_val > 0 else 1e-9 # Avoid log(0)
        if min_nonzero_base_for_norm >= vmax_base: # Handle case where all positive values are tiny
             min_nonzero_base_for_norm = vmax_base / 100 if vmax_base > 0 else 1e-9
        norm_base = matplotlib.colors.LogNorm(vmin=max(1e-9, min_nonzero_base_for_norm), vmax=max(1e-8,vmax_base)) # ensure vmin < vmax
        cbar_label_base = "log10(value)"
        scale_type_base = "Log Scale"
    else:
        norm_base = matplotlib.colors.Normalize(vmin=vmin_base, vmax=vmax_base)
        cbar_label_base = "value"
        scale_type_base = "Linear Scale"
        if log_scale and not np.any(baseline_heatmap > 0) : # Log scale requested but no positive values
            scale_type_base += " (No positive values)"

    # Data and cmap for baseline plot
    plot_data_base = baseline_heatmap
    cmap_base_final = base_cmap_obj

    if zeros_black:
        cmap_colors_base = copy.copy(base_cmap_obj(np.linspace(0, 1, 256)))
        cmap_base_final = matplotlib.colors.ListedColormap(cmap_colors_base)
        cmap_base_final.set_bad(color='black')
        plot_data_base = np.ma.masked_where(baseline_heatmap == 0, baseline_heatmap)
    elif log_scale and scale_type_base == "Log Scale": # Not zeros_black, but log_scale
        plot_data_base = baseline_heatmap.copy()
        # min_nonzero_base_for_norm is already calculated for LogNorm's vmin
        plot_data_base[plot_data_base == 0] = min_nonzero_base_for_norm 

    im_base = ax_base.imshow(plot_data_base, cmap=cmap_base_final, norm=norm_base, interpolation='nearest')
    ax_base.set_title(f"{baseline_title}", fontsize=title_fontsize)
    ax_base.set_xlabel("src", fontsize=label_fontsize)
    ax_base.set_ylabel("dest", fontsize=label_fontsize)
    fig.colorbar(im_base, ax=ax_base, shrink=0.8, aspect=20*0.8)

    # --- 2. Comparison Heatmaps (axes[1:]) ---
    if num_comparison_maps > 0:
        all_comp_heatmaps_flat = np.concatenate([hm.flatten() for hm in comparison_heatmaps])
        vmin_comp = np.min(all_comp_heatmaps_flat)
        vmax_comp = np.max(all_comp_heatmaps_flat)
        min_nonzero_comp_for_norm = vmax_comp # Default if no positive

        if log_scale and np.any(all_comp_heatmaps_flat > 0):
            min_val_comp = np.min(all_comp_heatmaps_flat[all_comp_heatmaps_flat > 0])
            min_nonzero_comp_for_norm = min_val_comp / 10 if min_val_comp > 0 else 1e-9
            if min_nonzero_comp_for_norm >= vmax_comp:
                min_nonzero_comp_for_norm = vmax_comp / 100 if vmax_comp > 0 else 1e-9
            norm_comp = matplotlib.colors.LogNorm(vmin=max(1e-9, min_nonzero_comp_for_norm), vmax=max(1e-8,vmax_comp))
            cbar_label_comp = "log10(value)"
            scale_type_comp = "Log Scale"
        else:
            norm_comp = matplotlib.colors.Normalize(vmin=vmin_comp, vmax=vmax_comp)
            cbar_label_comp = "value"
            scale_type_comp = "Linear Scale"
            if log_scale and not np.any(all_comp_heatmaps_flat > 0) :
                 scale_type_comp += " (No positive values)"

        cmap_comp_final = base_cmap_obj
        if zeros_black:
            cmap_colors_comp = copy.copy(base_cmap_obj(np.linspace(0, 1, 256)))
            cmap_comp_final = matplotlib.colors.ListedColormap(cmap_colors_comp)
            cmap_comp_final.set_bad(color='black')

        im_last_comp = None
        for i in range(num_comparison_maps):
            ax_comp = axes[1 + i]
            current_heatmap = comparison_heatmaps[i]
            
            plot_data_comp = current_heatmap
            if zeros_black:
                plot_data_comp = np.ma.masked_where(current_heatmap == 0, current_heatmap)
            elif log_scale and scale_type_comp == "Log Scale":
                plot_data_comp = current_heatmap.copy()
                plot_data_comp[plot_data_comp == 0] = min_nonzero_comp_for_norm
            
            im_comp = ax_comp.imshow(plot_data_comp, cmap=cmap_comp_final, norm=norm_comp, interpolation='nearest')
            ax_comp.set_title(f"{comparison_titles[i]}", fontsize=title_fontsize)
            ax_comp.set_xlabel("src", fontsize=label_fontsize)
            if i == 0: # Only set y-label for the first comparison plot
                ax_comp.set_ylabel("dest", fontsize=label_fontsize)
            else:
                ax_comp.set_yticklabels([]) # Hide y-ticks for subsequent comparison plots
            im_last_comp = im_comp
        
        if im_last_comp is not None:
            # Add a shared colorbar for comparison plots
            # Position it carefully, e.g., to the right of the last comparison plot
            cbar_comp = fig.colorbar(im_last_comp, ax=axes[1:].tolist(), shrink=0.8, aspect=20*0.8) # Adjust aspect ratio for multiple axes

    # --- 3. Final Touches ---
    # Determine the overall scale type to mention in main title
    # If both used log, or both used linear, it's simple. If mixed, be more generic or specific.
    # For simplicity, we assume `log_scale` param dictates the attempt for all.
    main_title_suffix = f"({scale_type_base if num_comparison_maps == 0 else scale_type_comp if baseline_heatmap is None else ('Log Scale' if log_scale else 'Linear Scale')})"
    fig.suptitle(f"{main_title} {main_title_suffix}", fontsize=title_fontsize + 2)
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight') # Also save as PDF for quality
        print(f"Figure saved to {save_path}.png and {save_path}.pdf")

    if show:
        plt.show()
    
    return fig
