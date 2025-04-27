import os
import graphviz
from hardware_compiler.utils import *
from typing import Dict, List, Set, Tuple

def visualize_hardware(hardware: Hardware, output_dir="hardware_visualizations", view=False):
    """
    Visualize the hardware hierarchy using graphviz, showing all levels in a single graph.
    
    Args:
        hardware: The hardware description
        output_dir: Directory to save visualization files
        view: Whether to open the visualization after creation
    """
    # Create main graph
    dot = graphviz.Digraph(name="Hardware_Hierarchy", 
                          comment="Hardware Hierarchy Visualization",
                          format="pdf")
    
    # Use fdp engine which handles large graphs better
    dot.attr(engine="fdp")
    
    # Set graph attributes
    dot.attr(overlap="false", splines="true", fontname="Arial")
    dot.attr("node", shape="box", style="filled", fontname="Arial", margin="0.2")
    
    # Create a simplified visualization
    create_simplified_hierarchy(dot, hardware)
    
    # Save and render the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hardware_hierarchy")
    
    try:
        dot.render(output_path, view=view)
        print(f"Visualization saved to {output_path}.pdf")
    except Exception as e:
        print(f"Error rendering visualization: {e}")
        # Try with even simpler settings
        try:
            dot.attr(engine="neato")
            dot.attr(overlap="scale", splines="line")
            dot.render(output_path + "_simple", view=view)
            print(f"Simplified visualization saved to {output_path}_simple.pdf")
        except Exception as e:
            print(f"Error rendering simplified visualization: {e}")
    
    # Create additional visualizations for specific hierarchy groupings
    create_hierarchy_level_visualizations(hardware, output_dir, view)
    
    return output_path + ".pdf"

def create_simplified_hierarchy(dot, hardware):
    """Create a simplified visualization of the hardware hierarchy"""
    # Add all modules as nodes
    for module in hardware.modules:
        node_id = get_node_id(module)
        label = get_node_label(module)
        
        dot.node(node_id, 
                label=label,
                fillcolor=get_color_for_function(module.function_type),
                tooltip=str(module.coords))
    
    # Add connections between modules
    add_connections(dot, hardware)
    
    # Add invisible edges to represent hierarchy
    add_hierarchy_edges(dot, hardware)

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
                
                # Use xlabel instead of label for orthogonal edges
                dot.edge(edge_id[0], edge_id[1], 
                        xlabel=f"{bandwidth}",  # Use xlabel instead of label
                        penwidth=str(min(1 + bandwidth/20, 3)),  # Scale line width with bandwidth
                        color=color,
                        style=style)
                added_edges.add(edge_id)

def add_hierarchy_edges(dot, hardware):
    """Add invisible edges to represent the hierarchy"""
    hierarchy_order = get_hierarchy_levels(hardware)
    
    # For each module, connect it to its parent with an invisible edge
    for i in range(1, len(hierarchy_order)):
        child_level = hierarchy_order[i]
        parent_level = hierarchy_order[i-1]
        
        child_modules = hardware.find_modules(hierarchy_type=child_level)
        
        for child in child_modules:
            # Find the parent of this child
            parent_coords = {k: v for k, v in child.coords.items() if k != child.hierarchy_type}
            parent = hardware.find_module(hierarchy_type=parent_level, **parent_coords)
            
            if parent:
                dot.edge(get_node_id(parent), get_node_id(child), 
                        style="dotted", 
                        color="gray", 
                        constraint="true",
                        weight="0.1")

def create_hierarchy_level_visualizations(hardware, output_dir, view):
    """Create separate visualizations for each hierarchy level and their connections"""
    hierarchy_levels = get_hierarchy_levels(hardware)
    
    for level in hierarchy_levels:
        # Create a new graph for this level
        level_dot = graphviz.Digraph(name=f"Hardware_{level}", 
                                    comment=f"{level} Level Visualization",
                                    format="pdf",
                                    engine="neato")
        
        level_dot.attr(overlap="false", splines="true")
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
                                 xlabel=f"{bandwidth}",
                                 penwidth=str(min(1 + bandwidth/20, 3)),
                                 color="blue")
        
        # Save and render the visualization
        output_path = os.path.join(output_dir, f"{level}_level")
        try:
            level_dot.render(output_path, view=False)  # Don't open all views
        except Exception as e:
            print(f"Error rendering {level} level visualization: {e}")

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
    # Include function type and the lowest level coordinate
    if not module.coords:
        return f"{module.hierarchy_type}\n{module.function_type}"
    
    # Get the most specific coordinate (the one for this hierarchy level)
    specific_coord = module.coords.get(module.hierarchy_type, "")
    return f"{module.hierarchy_type} {specific_coord}\n{module.function_type}"

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