from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable

class HierarchyType(Enum):
    """Enum for different hierarchy types"""
    ACCELERATOR = "ACCELERATOR"
    BANK = "BANK"
    PE = "PE"
    TILE = "TILE"
    SUBTILE = "SUBTILE"

class FunctionType(Enum):
    """Enum for different function types"""
    MVM = "MVM"                 # Matrix-vector multiplication
    ACTIVATION = "ACTIVATION"   # Activation function (e.g., SiLU)
    GLU = "GLU"                 # Gated Linear Unit
    ADD = "ADD"                 # Addition operation (for aggregation)
    DATAFOWARD = "DATAFOWARD"   # Data forwarding operation    

class ModuleId:
    """Flexible identifier for modules in the model"""
    def __init__(self, **kwargs):
        self.coords = kwargs  # Store all coordinates as a dictionary
    
    @classmethod
    def create(cls, **kwargs):
        """Factory method to create a ModuleId with specific coordinates"""
        return cls(**kwargs)
    
    def with_coords(self, **kwargs):
        """Create a new ModuleId with updated coordinates"""
        new_coords = self.coords.copy()
        new_coords.update(kwargs)
        return ModuleId(**new_coords)
    
    def get(self, key, default=None):
        """Get a coordinate value by key"""
        return self.coords.get(key, default)
    
    def __str__(self):
        coords_str = ", ".join(f"{k}={v}" for k, v in sorted(self.coords.items()))
        return f"Module({coords_str})"
    
    def __eq__(self, other):
        if not isinstance(other, ModuleId):
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
        self.receive = Dict[ModuleId, Dataflow]
        self.send = Dict[ModuleId, Dataflow]
        self.energy = 0
        self.area = 0
        self.latency = 0
        self.visit_count = 0

    def regist_receive(self, module_id: ModuleId, dataflow: Dataflow):
        """Add an receive module and its dataflow"""
        self.receive[module_id] = dataflow

    def get_receive(self) -> Dict[ModuleId, Dataflow]:
        """Get the receive modules and their dataflow"""
        return self.receive
    
    def update_receive(self, module_id: ModuleId, dataflow: Dataflow):
        """Update the dataflow for an receive module"""
        if module_id in self.receive:
            self.receive[module_id].update(**dataflow.dataflow)
            self.visit_count += 1
        else:
            raise KeyError(f"Module {module_id} not found in receive modules.")

    def regist_send(self, module_id: ModuleId, dataflow: Dataflow):
        """Add a send module and its dataflow"""
        self.send[module_id] = dataflow
    
    def get_send(self) -> Dict[ModuleId, Dataflow]:
        """Get the send modules and their dataflow"""
        return self.send
    
    def update_send(self, module_id: ModuleId, dataflow: Dataflow):
        """Update the dataflow for a send module"""
        if module_id in self.send:
            self.send[module_id].update(**dataflow.dataflow)
        else:
            raise KeyError(f"Module {module_id} not found in send modules.")
        
class HardwareBase():
    """Base class for hardware description"""
    def __init__(self):
       self.modules = List[Module]
       self.array_h = 0
       self.array_v = 0

    def add_module(self, module: Module):
        """Add a module to the hardware description"""
        self.modules.append(module)
    
    def find_module(self, module_id: ModuleId) -> Optional[Module]:
        """Find a module by its ID"""
        for module in self.modules:
            if module.coords == module_id.coords:
                return module
        return None
    

    