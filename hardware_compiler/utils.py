from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable

class HierarchyType(Enum):
    """Enum for different hierarchy types"""
    ACCELERATOR = "ACCELERATOR"
    BANK = "BANK"
    PE = "PE"
    TILE = "TILE"
    SUBTILE = "SUBTILE"
    CHAIN = "CHAIN"
    BLOCK = "BLOCK"

class FunctionType(Enum):
    """Enum for different function types"""
    MVM = "MVM"                 # Matrix-vector multiplication
    ACTIVATION = "ACTIVATION"   # Activation function (e.g., SiLU)
    GLU = "GLU"                 # Gated Linear Unit
    ADD = "ADD"                 # Addition operation (for aggregation)
    DATAFOWARD = "DATAFOWARD"   # Data forwarding operation    

class ModuleCoords:
    """Flexible identifier for modules in the model"""
    def __init__(self, **kwargs):
        self.coords = kwargs  # Store all coordinates as a dictionary
    
    @classmethod
    def create(cls, **kwargs):
        """Factory method to create a ModuleCoords with specific coordinates"""
        return cls(**kwargs)
    
    def with_coords(self, **kwargs):
        """Create a new ModuleCoords with updated coordinates"""
        new_coords = self.coords.copy()
        new_coords.update(kwargs)
        return ModuleCoords(**new_coords)
    
    def get(self, key, default=None):
        """Get a coordinate value by key"""
        return self.coords.get(key, default)
    
    def __str__(self):
        coords_str = ", ".join(f"{k}={v}" for k, v in sorted(self.coords.items()))
        return f"Module({coords_str})"
    
    def __eq__(self, other):
        if not isinstance(other, ModuleCoords):
            return False
        return self.coords == other.coords
    
    def __hash__(self):
        return hash(tuple(sorted(self.coords.items())))
    
class Dataflow():
    """manage bandwith and accumulated dataflow"""
    def __init__(self, **kwargs):
        self.dataflow = kwargs  # Store all dataflow information as a dictionary

    def __str__(self):
        dataflow_str = ", ".join(f"{k}={v}" for k, v in sorted(self.dataflow.items()))
        return f"Dataflow({dataflow_str})"
    
    def __eq__(self, other):
        if not isinstance(other, Dataflow):
            return False
        return self.dataflow == other.dataflow
    
    def get(self, key, default=None):
        """Get a dataflow value by key"""
        return self.dataflow.get(key, default)
    
    def update(self, **kwargs):
        """Update the dataflow with new values"""
        self.dataflow.update(kwargs)

class Module():
    """Base class for all modules"""
    def __init__(self, hierarchy_type: HierarchyType, function_type: FunctionType, **coords):
        self.coords = coords
        self.hierarchy_type = hierarchy_type
        self.function_type = function_type
        self.receive = {}
        self.send = {}
        self.energy = 0
        self.area = 0
        self.latency = 0
        self.visit_count = 0

    def regist_receive(self, module_coords: ModuleCoords, dataflow: Dataflow):
        """Add an receive module and its dataflow"""
        self.receive[module_coords] = dataflow

    def get_receive(self) -> Dict[ModuleCoords, Dataflow]:
        """Get the receive modules and their dataflow"""
        return self.receive
    
    def update_receive(self, module_coords: ModuleCoords, dataflow: Dataflow):
        """Update the dataflow for an receive module"""
        if module_coords in self.receive:
            self.receive[module_coords].update(**dataflow.dataflow)
            self.visit_count += 1
        else:
            raise KeyError(f"Module {module_coords} not found in receive modules.")

    def regist_send(self, module_coords: ModuleCoords, dataflow: Dataflow):
        """Add a send module and its dataflow"""
        self.send[module_coords] = dataflow
    
    def get_send(self) -> Dict[ModuleCoords, Dataflow]:
        """Get the send modules and their dataflow"""
        return self.send
    
    def update_send(self, module_coords: ModuleCoords, dataflow: Dataflow):
        """Update the dataflow for a send module"""
        if module_coords in self.send:
            self.send[module_coords].update(**dataflow.dataflow)
        else:
            raise KeyError(f"Module {module_coords} not found in send modules.")
        
class Hardware():
    """Class for hardware description"""
    def __init__(self):
       self.modules = List[Module]
       self.array_h = 0
       self.array_v = 0

    def add_module(self, module: Module):
        """Add a module to the hardware description"""
        self.modules.append(module)
    
    def find_module(self, **criteria) -> Optional[Module]:
        """Find a module by its ID"""
        for module in self.modules:
            match = True
            for key, value in criteria.items():
                if key == 'hierarchy_type':
                    if module.hierarchy_type != value:
                        match = False
                        break
                elif key == 'function_type':
                    if module.function_type != value:
                        match = False
                        break
                elif hasattr(module, key):
                    if getattr(module, key) != value:
                        match = False
                        break
                elif key in module.coords:
                    if module.coords[key] != value:
                        match = False
                        break
            if match:
                return module
        return None
    
    def find_modules(self, **criteria) -> List[Module]:
        """Find modules by specific coordinates"""
        results = []
        for module in self.modules:
            match = True
            for key, value in criteria.items():
                if key == 'hierarchy_type':
                    if module.hierarchy_type != value:
                        match = False
                        break
                elif key == 'function_type':
                    if module.function_type != value:
                        match = False
                        break
                elif hasattr(module, key):
                    if getattr(module, key) != value:
                        match = False
                        break
                elif key in module.coords:
                    if module.coords[key] != value:
                        match = False
                        break

            if match:
                results.append(module)
        return results
    
class HardwareCreator():
    """Base class for creating hardware modules"""

    #kwargs in hardwarecreator is numbers of elements in each hierarchy
    # e.g. {ACCELERATOR: 1, BANK: 4, PE: 16, TILE: 64, SUBTILE: 256}
    def __init__(self, array_h: int, array_v: int, **kwargs):
        self.array_h = array_h
        self.array_v = array_v
        self.hierarchy = kwargs

    def create_hardware() -> Hardware:
        """Create hardware discription. Abstract method to be implemented by subclasses"""
        return Hardware()