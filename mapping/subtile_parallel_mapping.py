from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable, Sequence
from mapping.utils import *

class TrivialMapping(Map_Compiledmodel_to_Hardware):
    """Trivial mapping of compiled model to hardware"""
    def __init__(self, compiled_model: CompiledModel, hardware: Hardware):
        super().__init__(compiled_model, hardware)
        self.tile_entry = 0
        self.subtile_entry = 0
        
    
    def map(self):
        """Map the compiled model to hardware"""
        # Implement the trivial mapping logic here
        self.initialize_module_available_map()
        self.initialize_sequences()

        for subfunction in self.compiled_model.subfunctions:
            # For each subfunction, find a suitable hardware module
            # and create a mapping entry
            hardware_module = self.map_available_module(subfunction)
            # if hardware_module:
            #     print('successfully mapped subfunction:', subfunction.coords, 'to hardware module:', hardware_module.coords)
            # else:
            #     print('failed to map subfunction:', subfunction.coords, 'function type',subfunction.op_type.value)
            
    def map_available_module(self, subfunction:SubFunction)-> Optional[Module]:
        """Find an available hardware module for the given subfunction"""
        # This is a placeholder implementation. You should implement the logic
        # to find an available hardware module based on your specific requirements.
        module = None
        for module in self.hardware.modules:
            occupy = self.is_available(module, subfunction)
            if occupy:
                if module.available_map[occupy] and module.hierarchy_type == HierarchyType.PE.value:
                    self.mapping[subfunction] = module
                    self.reverse_mapping[module] = subfunction
                    module.available_map[occupy] = False
                    self.subtile_entry = module.coords['SUBTILE']
                    self.tile_entry = module.coords['TILE']
                    return module
                elif module.hierarchy_type == HierarchyType.SUBTILE.value:
                    if module.coords['SUBTILE'] == self.subtile_entry and module.coords['TILE'] == self.tile_entry:
                        self.mapping[subfunction] = module
                        self.reverse_mapping[module] = subfunction
                        return module
                elif module.hierarchy_type == HierarchyType.TILE.value:
                    if module.coords['TILE'] == self.tile_entry:
                        self.mapping[subfunction] = module
                        self.reverse_mapping[module] = subfunction
                        return module
        return None
    
    def is_available(self, module:Module, subfunction:SubFunction) -> Optional[str]:
        """Check if the module is available for the subfunction"""
        # This is a placeholder implementation. You should implement the logic
        # to check if the module is available based on your specific requirements.
        occupy = None
        if subfunction.op_type == OperationType.MVM:
            if module.hierarchy_type == HierarchyType.PE.value and module.function_type == FunctionType.MVM.value:
                if subfunction.parent_function.name == 'up_proj':
                    if module.coords['PE'] == 0:
                        occupy = 'mvm_calculate'
                elif subfunction.parent_function.name == 'gate_proj':
                    if module.coords['PE'] == 1:
                        occupy = 'mvm_calculate'
                elif subfunction.parent_function.name == 'down_proj':
                    if module.coords['PE'] == 2:
                        occupy = 'mvm_calculate'
        elif subfunction.op_type == OperationType.GLU:
            if module.hierarchy_type == HierarchyType.PE.value and module.function_type == FunctionType.GLU.value:
                if module.coords['PE'] == 4:
                    occupy = 'glu_calculate'
        elif subfunction.op_type == OperationType.ACTIVATION:
            if module.hierarchy_type == HierarchyType.PE.value and module.function_type == FunctionType.ACTIVATION.value:
                if module.coords['PE'] == 3:
                    occupy = 'activation_calculate'
        elif subfunction.op_type == OperationType.ADD:
            if module.hierarchy_type == HierarchyType.SUBTILE.value:
                occupy = 'addition'
        elif subfunction.op_type == OperationType.DISTRIBUTE:
            if module.hierarchy_type == HierarchyType.TILE.value:
                occupy = 'distribution'
        elif subfunction.op_type == OperationType.CONCAT:
            if module.hierarchy_type == HierarchyType.TILE.value:
                occupy = 'concatenation'
        return occupy
            
    
    def initialize_module_available_map(self):
        """Initialize the available map for each module"""
        for module in self.hardware.modules:
            if module.hierarchy_type == HierarchyType.PE.value:
                if module.function_type == FunctionType.GLU.value:
                    module.available_map = {
                        'glu_calculate': True
                    }
                elif module.function_type == FunctionType.MVM.value:
                    module.available_map = {
                        'mvm_calculate': True
                    }
                elif module.function_type == FunctionType.ACTIVATION.value:
                    module.available_map = {
                        'activation_calculate': True
                    }
            elif module.hierarchy_type == HierarchyType.SUBTILE.value:
                module.available_map = {
                    'addition': True
                }
            elif module.hierarchy_type == HierarchyType.TILE.value:
                module.available_map = {
                    'distribution': True,
                    'concatenation': True
                }

    def initialize_sequences(self):
        subfunction_sequence = ['n','m','k','j','i']
        module_sequence = ['ACCELERATOR','BANK','TILE','SUBTILE','PE']

        self.compiled_model.subfunctions.sort(key = lambda o: tuple(o.coords[k] for k in subfunction_sequence))

        self.hardware.modules.sort(key = lambda o: tuple(o.coords.get(k,0) for k in module_sequence))
        # print('========after sorting========')
        # for module in self.hardware.modules:
        #     print('module:', module.coords)

    def sort_by_hierarchy(objs: List[Any], key_order: Sequence[str]) -> List[Any]:
        """
        根据 key_order 定义的层级顺序，对 objs 进行 **整体一次性** 排序（高位在前，低位在后）。
        
        参数
        ----
        objs : 待排序对象列表，对象需有 .coords(dict)
        key_order : 从“最高位”到“最低位”的 key 顺序，如 ["bank", "row", "col"]

        返回
        ----
        新的已排序列表（不会修改原列表）
        """
        # 若缺失 key 会抛 KeyError；如需容错，可用 obj.coords.get(k, 默认值)
        return sorted(objs, key=lambda o: tuple(o.coords[k] for k in key_order))