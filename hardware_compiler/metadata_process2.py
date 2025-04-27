import os
import graphviz
from hardware_compiler.utils import *
from typing import Dict, List, Set, Tuple

def visualize_hardware(hardware: Hardware, output_dir="hardware_visualizations", view=False):
    """
    Visualize the hardware hierarchy using nested clusters to reflect the actual hardware layout.
    
    Args:
        hardware: The hardware description
        output_dir: Directory to save visualization files
        view: Whether to open the visualization after creation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a hierarchical visualization that shows the nested structure
    hierarchy_path = create_nested_hierarchy_visualization(hardware, output_dir, view)
    
    # Create level-specific visualizations for detailed examination
    create_hierarchy_level_visualizations(hardware, output_dir, view)
    
    return hierarchy_path

def create_nested_hierarchy_visualization(hardware, output_dir, view):
    """Create a visualization with nested clusters to represent the hardware hierarchy"""
    # Create a new graph
    dot = graphviz.Digraph(name="Hardware_Hierarchy", 
                          comment="Hardware Hierarchy Visualization",
                          format="pdf")
    
    # Use dot engine for hierarchical layout
    dot.attr(engine="dot")
    
    # Set graph attributes
    dot.attr(rankdir="TB", ranksep="0.5")
    dot.attr("node", shape="box", style="filled", fontname="Arial", margin="0.1", height="0.3")
    dot.attr("edge", fontsize="10")
    
    # Get hierarchy levels
    hierarchy_levels = get_hierarchy_levels(hardware)
    
    # Start with the highest level (usually ACCELERATOR)
    highest_level = hierarchy_levels[0]
    top_modules = hardware.find_modules(hierarchy_type=highest_level)
    
    # For each top-level module, create a nested hierarchy
    for top_module in top_modules:
        create_nested_cluster(dot, hardware, top_module, hierarchy_levels, 0)
    
    # Add connections between modules
    add_connections(dot, hardware)
    
    # Save and render the visualization
    output_path = os.path.join(output_dir, "nested_hierarchy")
    
    try:
        dot.render(output_path, view=view)
        print(f"Nested hierarchy visualization saved to {output_path}.pdf")
        return output_path + ".pdf"
    except Exception as e:
        print(f"Error rendering nested hierarchy visualization: {e}")
        # Try with simpler settings
        try:
            # Create a simpler version
            create_simplified_nested_hierarchy(hardware, output_dir, view)
            return os.path.join(output_dir, "simplified_nested_hierarchy.pdf")
        except Exception as e2:
            print(f"Error rendering simplified nested hierarchy: {e2}")
            return None

def create_nested_cluster(parent_graph, hardware, module, hierarchy_levels, level_index):
    """
    Recursively create nested clusters for a module and its children.
    
    Args:
        parent_graph: The parent graph or cluster
        hardware: The hardware description
        module: The current module
        hierarchy_levels: List of hierarchy levels
        level_index: Current index in the hierarchy_levels list
    """
    # Create a cluster for this module
    cluster_name = f"cluster_{get_node_id(module)}"
    
    with parent_graph.subgraph(name=cluster_name) as cluster:
        # Set cluster attributes
        cluster.attr(label=get_cluster_label(module), 
                    style="filled,rounded", 
                    color=get_border_color_for_hierarchy(module.hierarchy_type),
                    fillcolor="white",
                    penwidth="1")
        
        # Add the module node itself
        node_id = get_node_id(module)
        cluster.node(node_id, 
                    label=get_node_label(module),
                    fillcolor=get_color_for_function(module.function_type),
                    tooltip=str(module.coords),
                    shape="box",
                    style="filled")
        
        # If there are more hierarchy levels below this one
        if level_index + 1 < len(hierarchy_levels):
            next_level = hierarchy_levels[level_index + 1]
            
            # Find child modules at the next level
            child_modules = find_direct_children(hardware, module, next_level)
            
            # If there are child modules, create clusters for them
            if child_modules:
                # Recursively create clusters for each child
                for child_module in child_modules:
                    create_nested_cluster(cluster, hardware, child_module, hierarchy_levels, level_index + 1)

def create_simplified_nested_hierarchy(hardware, output_dir, view):
    """Create a simplified nested hierarchy visualization for complex hardware"""
    # Create a new graph
    dot = graphviz.Digraph(name="Simplified_Hardware_Hierarchy", 
                          comment="Simplified Hardware Hierarchy Visualization",
                          format="pdf")
    
    # Use dot engine for hierarchical layout
    dot.attr(engine="dot")
    
    # Set graph attributes
    dot.attr(rankdir="TB", ranksep="0.5")
    dot.attr("node", shape="box", style="filled", fontname="Arial", margin="0.1")
    
    # Get hierarchy levels
    hierarchy_levels = get_hierarchy_levels(hardware)
    
    # For each level, create a sample representation
    for i, level in enumerate(hierarchy_levels):
        # Create a cluster for this level
        with dot.subgraph(name=f"cluster_{level}") as cluster:
            cluster.attr(label=level, style="filled,rounded", color="black", fillcolor="white")
            
            # Add a sample of modules at this level (limit to avoid overloading)
            modules = hardware.find_modules(hierarchy_type=level)
            sample_size = min(5, len(modules))
            
            for j in range(sample_size):
                module = modules[j]
                node_id = f"{level}_{j}"
                label = f"{level} {j}"
                
                cluster.node(node_id, 
                           label=label,
                           fillcolor=get_color_for_hierarchy(level))
            
            # If there are more modules, add a placeholder
            if len(modules) > sample_size:
                cluster.node(f"{level}_more", 
                           label=f"... {len(modules) - sample_size} more",
                           fillcolor="lightgrey",
                           style="filled,dashed")
    
    # Add representative connections between levels
    for i in range(1, len(hierarchy_levels)):
        parent_level = hierarchy_levels[i-1]
        child_level = hierarchy_levels[i]
        
        # Connect the first sample of each level
        dot.edge(f"{parent_level}_0", f"{child_level}_0", 
                style="dashed", 
                label="contains")
    
    # Save and render the visualization
    output_path = os.path.join(output_dir, "simplified_nested_hierarchy")
    dot.render(output_path, view=view)
    print(f"Simplified nested hierarchy visualization saved to {output_path}.pdf")

def add_connections(dot, hardware):
    """Add connections between modules based on dataflow"""
    added_edges = set()
    
    for module in hardware.modules:
        for receiver, dataflow in module.send.items():
            # Create a unique edge identifier
            edge_id = (get_node_id(module), get_node_id(receiver))
            reverse_edge_id = (edge_id[1], edge_id[0])
            
            # Only add if we haven't added this edge or its reverse
            if edge_id not in added_edges and reverse_edge_id not in added_edges:
                bandwidth = dataflow.get('bandwidth', 0)
                
                # Use different colors for connections between different hierarchy levels
                if module.hierarchy_type != receiver.hierarchy_type:
                    color = "red"
                    style = "dashed"
                else:
                    color = "blue"
                    style = "solid"
                
                # Use headlabel instead of label/xlabel for better placement
                dot.edge(edge_id[0], edge_id[1], 
                        headlabel=f"{bandwidth}",
                        penwidth=str(min(1 + bandwidth/50, 2)),  # Scale line width with bandwidth
                        color=color,
                        style=style,
                        minlen="1",
                        weight="0.1")
                added_edges.add(edge_id)

def create_hierarchy_level_visualizations(hardware, output_dir, view):
    """Create separate visualizations for each hierarchy level and their connections"""
    hierarchy_levels = get_hierarchy_levels(hardware)
    
    for level in hierarchy_levels:
        # Create a new graph for this level
        level_dot = graphviz.Digraph(name=f"Hardware_{level}", 
                                    comment=f"{level} Level Visualization",
                                    format="pdf",
                                    engine="neato")
        
        level_dot.attr(overlap="false", splines="spline")
        level_dot.attr("node", shape="box", style="filled", fontname="Arial")
        
        # Get all modules at this level
        modules_at_level = hardware.find_modules(hierarchy_type=level)
        
        # Create nodes for each module
        for module in modules_at_level:
            node_id = get_node_id(module)
            level_dot.node(node_id, 
                          label=get_node_label(module),
                          fillcolor=get_color_for_function(module.function_type),
                          tooltip=str(module.coords))
        
        # Add connections between modules at this level
        for module in modules_at_level:
            for receiver, dataflow in module.send.items():
                # Only add connections to modules at the same level
                if receiver.hierarchy_type == level:
                    bandwidth = dataflow.get('bandwidth', 0)
                    level_dot.edge(get_node_id(module), get_node_id(receiver), 
                                 headlabel=f"{bandwidth}",
                                 penwidth=str(min(1 + bandwidth/50, 2)),
                                 color="blue")
        
        # Save and render the visualization
        output_path = os.path.join(output_dir, f"{level}_level")
        try:
            level_dot.render(output_path, view=False)  # Don't open all views
        except Exception as e:
            print(f"Error rendering {level} level visualization: {e}")

def find_direct_children(hardware, parent_module, child_hierarchy_type):
    """Find all modules that are direct children of the given parent module at the specified hierarchy level"""
    parent_coords = parent_module.coords
    
    # Find all modules at the child level that have the same parent coordinates
    child_modules = []
    for module in hardware.modules:
        if module.hierarchy_type == child_hierarchy_type:
            # Check if all parent coordinates match
            is_child = True
            for key, value in parent_coords.items():
                if key not in module.coords or module.coords[key] != value:
                    is_child = False
                    break
            
            if is_child:
                child_modules.append(module)
    
    return child_modules

def get_hierarchy_levels(hardware):
    """Get all hierarchy levels present in the hardware, ordered from highest to lowest"""
    levels = set()
    for module in hardware.modules:
        levels.add(module.hierarchy_type)
    
    # Define the hierarchy order
    hierarchy_order = [
        HierarchyType.ACCELERATOR.value,
        HierarchyType.BANK.value,
        HierarchyType.TILE.value,
        HierarchyType.SUBTILE.value,
        HierarchyType.PE.value,
        HierarchyType.CHAIN.value,
        HierarchyType.BLOCK.value
    ]
    
    # Return the levels in the correct order
    return [level for level in hierarchy_order if level in levels]

def get_node_id(module):
    """Generate a unique ID for a module node"""
    coords_str = "_".join(f"{k}_{v}" for k, v in sorted(module.coords.items()))
    return f"{module.hierarchy_type}_{coords_str}"

def get_node_label(module):
    """Generate a label for a module node"""
    # For lower levels, use a more compact label
    if module.hierarchy_type in [HierarchyType.PE.value, HierarchyType.CHAIN.value, HierarchyType.BLOCK.value]:
        specific_coord = module.coords.get(module.hierarchy_type, "")
        return f"{module.function_type}\n{specific_coord}"
    
    # For higher levels, include more information
    specific_coord = module.coords.get(module.hierarchy_type, "")
    return f"{module.hierarchy_type} {specific_coord}\n{module.function_type}"

def get_cluster_label(module):
    """Generate a label for a module cluster"""
    # Just show the hierarchy type and its specific coordinate
    specific_coord = module.coords.get(module.hierarchy_type, "")
    return f"{module.hierarchy_type} {specific_coord}"

def get_color_for_function(function_type):
    """Return a color based on the function type"""
    color_map = {
        FunctionType.MVM.value: "#ffcccc",  # Light red
        FunctionType.ACTIVATION.value: "#ccffcc",  # Light green
        FunctionType.GLU.value: "#ccccff",  # Light blue
        FunctionType.ADD.value: "#ffffcc",  # Light yellow
        FunctionType.DATAFOWARD.value: "#ffccff",  # Light purple
    }
    return color_map.get(function_type, "#ffffff")  # Default to white

def get_color_for_hierarchy(hierarchy_type):
    """Return a color based on the hierarchy type"""
    color_map = {
        HierarchyType.ACCELERATOR.value: "#ffe6e6",  # Very light red
        HierarchyType.BANK.value: "#e6ffe6",  # Very light green
        HierarchyType.TILE.value: "#e6e6ff",  # Very light blue
        HierarchyType.SUBTILE.value: "#ffe6ff",  # Very light purple
        HierarchyType.PE.value: "#e6ffff",  # Very light cyan
        HierarchyType.CHAIN.value: "#ffffe6",  # Very light yellow
        HierarchyType.BLOCK.value: "#ffe6cc",  # Very light orange
    }
    return color_map.get(hierarchy_type, "#ffffff")  # Default to white

def get_border_color_for_hierarchy(hierarchy_type):
    """Return a border color based on the hierarchy type"""
    color_map = {
        HierarchyType.ACCELERATOR.value: "#ff0000",  # Red
        HierarchyType.BANK.value: "#00ff00",  # Green
        HierarchyType.TILE.value: "#0000ff",  # Blue
        HierarchyType.SUBTILE.value: "#ff00ff",  # Purple
        HierarchyType.PE.value: "#00ffff",  # Cyan
        HierarchyType.CHAIN.value: "#ffff00",  # Yellow
        HierarchyType.BLOCK.value: "#ff8800",  # Orange
    }
    return color_map.get(hierarchy_type, "#000000")  # Default to black