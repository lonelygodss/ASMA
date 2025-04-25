import graphviz
from hardware_compiler.utils import *
from typing import Dict, List, Optional, Set, Tuple

class HardwareVisualizer:
    """Class for visualizing hardware descriptions using graphviz"""
    
    def __init__(self, hardware: Hardware):
        self.hardware = hardware
        # Color scheme for hierarchy types
        self.color_map = {
            HierarchyType.ACCELERATOR.value: "#E8F8F5",  # Light Teal
            HierarchyType.BANK.value: "#EAFAF1",         # Light Green
            HierarchyType.TILE.value: "#FEF9E7",         # Light Yellow
            HierarchyType.SUBTILE.value: "#F9EBEA",      # Light Red
            HierarchyType.PE.value: "#EBF5FB",           # Light Blue
        }
        
        # Color scheme for function types
        self.function_color_map = {
            FunctionType.MVM.value: "#AED6F1",          # Blue
            FunctionType.ACTIVATION.value: "#F5B7B1",   # Red
            FunctionType.GLU.value: "#F9E79F",          # Yellow
            FunctionType.ADD.value: "#A9DFBF",          # Green
            FunctionType.DATAFOWARD.value: "#D7BDE2",   # Purple
        }
        
        # Define hierarchy order for proper nesting
        self.hierarchy_levels = [
            HierarchyType.ACCELERATOR.value,
            HierarchyType.BANK.value,
            HierarchyType.TILE.value,
            HierarchyType.SUBTILE.value,
            HierarchyType.PE.value
        ]
    
    def generate_node_id(self, module: Module) -> str:
        """Generate a unique ID for a module based on its coordinates"""
        coords = [f"{k}_{v}" for k, v in sorted(module.coords.items())]
        return "_".join(coords)
    
    def generate_node_label(self, module: Module, include_coords: bool = True) -> str:
        """Generate a human-readable label for a module"""
        if include_coords:
            coords = [f"{k}={v}" for k, v in sorted(module.coords.items())]
            return f"{module.hierarchy_type}\n({', '.join(coords)})\n{module.function_type}"
        else:
            return f"{module.hierarchy_type}\n{module.function_type}"
    
    def visualize_hierarchy(self, output_path: str = "hardware_hierarchy", view: bool = True):
        """Visualize the hardware hierarchy with proper nesting of modules"""
        dot = graphviz.Digraph('Hardware Hierarchy', 
                              filename=f"{output_path}.gv",
                              engine='dot',
                              format='pdf')
        dot.attr(rankdir='TB', size='14,14', dpi='300', newrank='true')
        
        # Group modules by their hierarchical position
        hierarchy_dict = self._build_hierarchy_dict()
        
        # Recursively create nested clusters to represent containment
        self._add_hierarchy_clusters(dot, hierarchy_dict)
        
        dot.render(view=view)
        return dot
    
    def _build_hierarchy_dict(self) -> dict:
        """Build a nested dictionary representing the hardware hierarchy"""
        hierarchy_dict = {}
        
        # Start with Accelerator level
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.ACCELERATOR.value:
                acc_id = module.coords[HierarchyType.ACCELERATOR.value]
                if acc_id not in hierarchy_dict:
                    hierarchy_dict[acc_id] = {
                        'module': module,
                        'banks': {}
                    }
        
        # Add Banks within each Accelerator
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.BANK.value:
                acc_id = module.coords[HierarchyType.ACCELERATOR.value]
                bank_id = module.coords[HierarchyType.BANK.value]
                if acc_id in hierarchy_dict:
                    hierarchy_dict[acc_id]['banks'][bank_id] = {
                        'module': module,
                        'tiles': {}
                    }
        
        # Add Tiles within each Bank
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.TILE.value:
                acc_id = module.coords[HierarchyType.ACCELERATOR.value]
                bank_id = module.coords[HierarchyType.BANK.value]
                tile_id = module.coords[HierarchyType.TILE.value]
                if acc_id in hierarchy_dict and bank_id in hierarchy_dict[acc_id]['banks']:
                    hierarchy_dict[acc_id]['banks'][bank_id]['tiles'][tile_id] = {
                        'module': module,
                        'subtiles': {}
                    }
        
        # Add SubTiles within each Tile
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.SUBTILE.value:
                acc_id = module.coords[HierarchyType.ACCELERATOR.value]
                bank_id = module.coords[HierarchyType.BANK.value]
                tile_id = module.coords[HierarchyType.TILE.value]
                subtile_id = module.coords[HierarchyType.SUBTILE.value]
                if (acc_id in hierarchy_dict and 
                    bank_id in hierarchy_dict[acc_id]['banks'] and 
                    tile_id in hierarchy_dict[acc_id]['banks'][bank_id]['tiles']):
                    hierarchy_dict[acc_id]['banks'][bank_id]['tiles'][tile_id]['subtiles'][subtile_id] = {
                        'module': module,
                        'pes': {}
                    }
        
        # Add PEs within each SubTile
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.PE.value:
                acc_id = module.coords[HierarchyType.ACCELERATOR.value]
                bank_id = module.coords[HierarchyType.BANK.value]
                tile_id = module.coords[HierarchyType.TILE.value]
                subtile_id = module.coords[HierarchyType.SUBTILE.value]
                pe_id = module.coords[HierarchyType.PE.value]
                path = hierarchy_dict.get(acc_id, {}).get('banks', {}).get(bank_id, {}).get('tiles', {}).get(tile_id, {}).get('subtiles', {})
                if subtile_id in path:
                    path[subtile_id]['pes'][pe_id] = {
                        'module': module
                    }
        
        return hierarchy_dict
    
    def _add_hierarchy_clusters(self, dot, hierarchy_dict, prefix=""):
        """Recursively add clusters to represent nested hierarchy"""
        # Add Accelerator level
        for acc_id, acc_data in hierarchy_dict.items():
            acc_module = acc_data['module']
            acc_node_id = self.generate_node_id(acc_module)
            
            with dot.subgraph(name=f"cluster_acc_{acc_id}") as acc_cluster:
                acc_cluster.attr(label=f"Accelerator {acc_id}", style='filled', fillcolor=self.color_map[HierarchyType.ACCELERATOR.value])
                acc_cluster.node(acc_node_id, self.generate_node_label(acc_module, include_coords=False))
                
                # Add Banks within this Accelerator
                for bank_id, bank_data in acc_data['banks'].items():
                    bank_module = bank_data['module']
                    bank_node_id = self.generate_node_id(bank_module)
                    
                    with acc_cluster.subgraph(name=f"cluster_bank_{acc_id}_{bank_id}") as bank_cluster:
                        bank_cluster.attr(label=f"Bank {bank_id}", style='filled', fillcolor=self.color_map[HierarchyType.BANK.value])
                        bank_cluster.node(bank_node_id, self.generate_node_label(bank_module, include_coords=False))
                        
                        # Add Tiles within this Bank
                        for tile_id, tile_data in bank_data['tiles'].items():
                            tile_module = tile_data['module']
                            tile_node_id = self.generate_node_id(tile_module)
                            
                            with bank_cluster.subgraph(name=f"cluster_tile_{acc_id}_{bank_id}_{tile_id}") as tile_cluster:
                                tile_cluster.attr(label=f"Tile {tile_id}", style='filled', fillcolor=self.color_map[HierarchyType.TILE.value])
                                tile_cluster.node(tile_node_id, self.generate_node_label(tile_module, include_coords=False))
                                
                                # Add SubTiles within this Tile
                                for subtile_id, subtile_data in tile_data['subtiles'].items():
                                    subtile_module = subtile_data['module']
                                    subtile_node_id = self.generate_node_id(subtile_module)
                                    
                                    with tile_cluster.subgraph(name=f"cluster_subtile_{acc_id}_{bank_id}_{tile_id}_{subtile_id}") as subtile_cluster:
                                        subtile_cluster.attr(label=f"SubTile {subtile_id}", style='filled', fillcolor=self.color_map[HierarchyType.SUBTILE.value])
                                        subtile_cluster.node(subtile_node_id, self.generate_node_label(subtile_module, include_coords=False))
                                        
                                        # Add PEs within this SubTile
                                        for pe_id, pe_data in subtile_data['pes'].items():
                                            pe_module = pe_data['module']
                                            pe_node_id = self.generate_node_id(pe_module)
                                            
                                            # Add PE with function type coloring
                                            subtile_cluster.node(pe_node_id, 
                                                               f"PE {pe_id}\n{pe_module.function_type}", 
                                                               style='filled', 
                                                               fillcolor=self.function_color_map[pe_module.function_type])
    
    def visualize_connections(self, output_path: str = "hardware_connections", view: bool = True):
        """Visualize the hardware connections with hierarchical nesting"""
        dot = graphviz.Digraph('Hardware Connections', 
                              filename=f"{output_path}.gv",
                              engine='dot',
                              format='pdf')
        dot.attr(rankdir='TB', size='14,14', dpi='300', newrank='true')
        
        # Build the hierarchy dictionary
        hierarchy_dict = self._build_hierarchy_dict()
        
        # Create the nested structure
        self._add_hierarchy_clusters(dot, hierarchy_dict)
        
        # Add connection edges
        added_edges = set()
        
        # Process connections within the same hierarchy level
        for module in self.hardware.modules:
            src_id = self.generate_node_id(module)
            
            for target, dataflow in module.send.items():
                dst_id = self.generate_node_id(target)
                edge_key = (src_id, dst_id)
                
                if edge_key not in added_edges:
                    bandwidth = dataflow.get('bandwidth', 1)
                    thickness = min(5, max(1, 0.5 + bandwidth / 20))  # Scale thickness by bandwidth
                    
                    # Different edge styles based on the relationship
                    if module.hierarchy_type == target.hierarchy_type:
                        # Same-level connections
                        dot.edge(src_id, dst_id, label=f"BW={bandwidth}", 
                                penwidth=str(thickness), color="blue")
                    else:
                        # Different level connections
                        dot.edge(src_id, dst_id, label=f"BW={bandwidth}", 
                                penwidth=str(thickness), color="green",
                                style="dashed" if module.hierarchy_type > target.hierarchy_type else "solid")
                    
                    added_edges.add(edge_key)
        
        dot.render(view=view)
        return dot
    
    def visualize_function_view(self, output_path: str = "hardware_functions", view: bool = True):
        """Visualize modules grouped by function types with hierarchical structure"""
        dot = graphviz.Digraph('Hardware Functions', 
                              filename=f"{output_path}.gv",
                              engine='dot',
                              format='pdf')
        dot.attr(rankdir='TB', size='14,14', dpi='300', newrank='true')
        
        # First group by function type
        function_groups = {}
        for module in self.hardware.modules:
            if module.function_type not in function_groups:
                function_groups[module.function_type] = []
            function_groups[module.function_type].append(module)
        
        # Create main clusters by function type
        for function_type, modules in function_groups.items():
            with dot.subgraph(name=f"cluster_{function_type}") as func_cluster:
                func_cluster.attr(label=f"{function_type}", style='filled', 
                                fillcolor=self.function_color_map[function_type])
                
                # Group modules by hierarchy type within function cluster
                hierarchy_modules = {}
                for module in modules:
                    hierarchy_type = module.hierarchy_type
                    if hierarchy_type not in hierarchy_modules:
                        hierarchy_modules[hierarchy_type] = []
                    hierarchy_modules[hierarchy_type].append(module)
                
                # Create nested clusters for each hierarchy level
                for hierarchy in self.hierarchy_levels:
                    if hierarchy in hierarchy_modules:
                        # Further group by parent coordinates
                        parent_groups = {}
                        for module in hierarchy_modules[hierarchy]:
                            # Create parent key based on hierarchy
                            parent_key = "_".join([f"{k}_{v}" for k, v in sorted(module.coords.items()) 
                                                if k != hierarchy])
                            if parent_key not in parent_groups:
                                parent_groups[parent_key] = []
                            parent_groups[parent_key].append(module)
                        
                        # Create clusters for each parent group
                        for parent_key, parent_modules in parent_groups.items():
                            cluster_name = f"cluster_{function_type}_{hierarchy}_{parent_key}"
                            with func_cluster.subgraph(name=cluster_name) as hier_cluster:
                                if parent_modules:
                                    coords_str = ", ".join([f"{k}={v}" for k, v in sorted(parent_modules[0].coords.items()) 
                                                        if k != hierarchy])
                                    hier_cluster.attr(label=f"{hierarchy} ({coords_str})", 
                                                    style='filled', 
                                                    fillcolor=self.color_map[hierarchy])
                                    
                                    # Add nodes to this cluster
                                    for module in parent_modules:
                                        node_id = self.generate_node_id(module)
                                        if hierarchy == HierarchyType.PE.value:
                                            hier_cluster.node(node_id, 
                                                            f"PE {module.coords[HierarchyType.PE.value]}", 
                                                            shape='box')
                                        else:
                                            hier_cluster.node(node_id, 
                                                            f"{hierarchy} {module.coords[hierarchy]}", 
                                                            shape='box')
        
        # Add edges for dataflow connections between different function types
        # We only show inter-function connections to reduce visual clutter
        added_edges = set()
        for module in self.hardware.modules:
            src_id = self.generate_node_id(module)
            
            for target, dataflow in module.send.items():
                if module.function_type != target.function_type:
                    dst_id = self.generate_node_id(target)
                    edge_key = (src_id, dst_id)
                    
                    if edge_key not in added_edges:
                        bandwidth = dataflow.get('bandwidth', 1)
                        dot.edge(src_id, dst_id, label=f"BW={bandwidth}", 
                                color="darkgreen", penwidth=str(min(3, max(1, bandwidth/30))))
                        added_edges.add(edge_key)
        
        dot.render(view=view)
        return dot

def visualize_hardware(hardware: Hardware, output_dir: str = "", view: bool = True):
    """Convenience function to generate all hardware visualizations"""
    visualizer = HardwareVisualizer(hardware)
    
    prefix = output_dir + "/" if output_dir else ""
    
    # Generate three different visualization views
    visualizer.visualize_hierarchy(output_path=f"{prefix}hardware_hierarchy", view=view)
    visualizer.visualize_connections(output_path=f"{prefix}hardware_connections", view=view)
    visualizer.visualize_function_view(output_path=f"{prefix}hardware_functions", view=view)
    
    print(f"Generated hardware visualizations with proper hierarchical nesting")
