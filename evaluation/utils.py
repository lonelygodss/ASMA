from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable, Iterable, TypeVar

from hardware_compiler.utils import *
from model_compiler.utils import *
from mapping.utils import *
import model_compiler.metadata_proess as dataproc
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from evaluation.visualize import visualize_heat_map

# 用 TypeVar 而非具体类名,方便在任意 Module 类型上复用
T = TypeVar("T")

def shortest_paths(start: T, target: T, neighbors: Dict[T, Iterable[T]]) -> List[Tuple[T, ...]]:
    """
    BFS 寻找 start &rarr; target 的所有最短路径。

    Parameters
    ----------
    start   : 起点 module
    target  : 终点 module
    neighbors : dict,键为 module,值为可迭代对象,列出与键直接相连的 module

    Returns
    -------
    List[Tuple[T, ...]]
        每条最短路径一个 tuple;若不可达则返回空列表。
    """
    if start == target:                          # 起终点重合
        return [(start,)]

    q: deque[Tuple[T, ...]] = deque([(start,)])  # 队列里存「整条路径」
    seen: Dict[T, int] = {start: 0}              # 已访问节点及其 BFS 深度
    shortest: List[Tuple[T, ...]] = []
    best_len = None                              # 目前已知最短路径长度

    while q:
        path = q.popleft()
        cur = path[-1]

        # 找到终点：记录、更新 best_len
        if cur == target:
            if best_len is None:
                best_len = len(path)
            if len(path) == best_len:
                shortest.append(tuple(path))
            continue                             # 不必扩展终点

        # 若已超过当前最短长度,无需再延伸
        if best_len is not None and len(path) >= best_len:
            continue

        for nxt in neighbors.get(cur, ()):
            # 只在以下两种情况下入队：
            # 1）没见过；2）见过但同样深度（支持多条等长路径）
            nxt_depth = len(path)
            if nxt not in seen or seen[nxt] == nxt_depth:
                seen[nxt] = nxt_depth
                q.append(path + (nxt,))

    return shortest



class Dataflow_parser():
    """Parse the dataflow information"""
    def __init__(self, model: CompiledModel, hardware: Hardware, mapping: dict, dataflow_path: list):
        self.model = model
        self.hardware = hardware
        self.mapping = mapping
        self.dataflow_path = dataflow_path

    def parse_dataflow(self, flag: bool = False):
        """Parse the dataflow information"""
        # Implement the parsing logic here
        heat_map_base = self.initialize_heat_map(flag)
        if flag: visualize_heat_map(heat_map_base, "Base Heat Map (Bandwidth)",False,log_scale= True)
        for model_path in self.dataflow_path:
            transfers = model_path.get("transfers", [])
            for transfer in transfers:
                # Extract the source and destination coordinates
                source_id = transfer.get('from')
                dest_id = transfer.get('to')
                source_coords = dataproc.id_to_coords(source_id)
                dest_coords = dataproc.id_to_coords(dest_id)
                source_sf = self.model.get_subfunctions_by_coords(**source_coords)
                dest_sf = self.model.get_subfunctions_by_coords(**dest_coords)
                package = transfer.get('size_h')*transfer.get('size_v')
                # Get initial and target modules
                initial_module = self.mapping.get(source_sf[0])
                target_module = self.mapping.get(dest_sf[0])
                # Update the dataflow
                if initial_module and target_module:
                    if flag: print('find path for: ',source_id,' to ',dest_id)
                    self.update_dataflow(initial_module, target_module, package)
        if flag : print('parsing finished!')
        heat_map_with_data = self.dataflow_heat_map(flag)
        latency_map = self.latency_heat_map(flag)
        if flag: 
            visualize_heat_map(heat_map_with_data, "Dataflow Heat Map (Accumulated Data)", False, log_scale= True)
            visualize_heat_map(latency_map, "Latency Heat Map", True, log_scale= True)
        return [heat_map_with_data, latency_map]

    def update_dataflow(self, module_init: Module, module_target: Module, package: int, flag: bool = False):
        """Update the dataflow information"""
        # Find hardware pathes connecting the two modules
        hardware_pathes = self.find_hardware_pathes(module_init, module_target,flag)
        if not hardware_pathes:
            print('error')
        path = hardware_pathes[0]
        # Update the dataflow for each module in the path
        for module_index in range(len(path)-1):
            if flag : print('updating dataflow from: ',path[module_index].coords,'to:',path[module_index+1].coords,' with packge size:',package)
            data_send = {
                'data_accumulated':                package
            }
            # Because in module1.send and module2.receive, they bind the same dataflow object,only one direction need to be updated and they work for both
            # Update the dataflow
            path[module_index].add_send_package(path[module_index+1], Dataflow(**data_send))
            

    def find_hardware_pathes(self, module_init: Module, module_target: Module, flag: bool = False) -> List[Tuple[Module]]:
        """Find hardware pathes connecting the two modules"""

        hardware_pathes = shortest_paths(module_init, module_target, self.hardware.graph)
        if flag : print('path found:')
        if flag : print(" -> ".join(f"{m.coords}" for m in hardware_pathes[0]))
        if flag : print('=====================')
        return        hardware_pathes
        # Implement the logic to find hardware pathes
    
    def generate_heat_map(self, flag: bool = False):
        """Generate a heat map for the dataflow"""
        # Implement the logic to generate a heat map
        heat_map = self.initialize_heat_map(flag)

    def initialize_heat_map(self,flag: bool = False):
        """Initialize the heat map"""
        # Implement the logic to initialize the heat map
        heat_map = np.zeros((len(self.hardware.modules), len(self.hardware.modules)))
        for i in range(len(self.hardware.modules)):
            module_i = self.hardware.modules[i]
            if module_i.get_send():
                for j in range(len(self.hardware.modules)):
                    module_j = self.hardware.modules[j]
                    if module_j in module_i.get_send():
                        data_bandwidth = module_i.send[module_j].dataflow['bandwidth']
                        heat_map[i][j] = data_bandwidth
                        if flag: print(f"Heat map [{i}][{j}]: {data_bandwidth}")
        if flag: print('heat map initialized!')
        return heat_map
    
    def dataflow_heat_map(self,flag: bool = False):
        """generate the heat map with dataflow information"""
        heat_map = np.zeros((len(self.hardware.modules), len(self.hardware.modules)))
        for i in range(len(self.hardware.modules)):
            module_i = self.hardware.modules[i]
            if module_i.get_send():
                for j in range(len(self.hardware.modules)):
                    module_j = self.hardware.modules[j]
                    if module_j in module_i.get_send():
                        data_accum = module_i.send[module_j].dataflow['data_accumulated']
                        heat_map[i][j] = data_accum/1024
                        if flag: print(f"Heat map [{i}][{j}]: {data_accum}")
        if flag: print('dataflow heat map generated!')
        return heat_map   

    def latency_heat_map(self,flag: bool = False):
        """generate the heat map with dataflow information"""
        heat_map = np.zeros((len(self.hardware.modules), len(self.hardware.modules)))
        for i in range(len(self.hardware.modules)):
            module_i = self.hardware.modules[i]
            if module_i.get_send():
                for j in range(len(self.hardware.modules)):
                    module_j = self.hardware.modules[j]
                    if module_j in module_i.get_send():
                        data_accum = module_i.send[module_j].dataflow['data_accumulated']
                        bandwidth = module_i.send[module_j].dataflow['bandwidth']
                        latency = data_accum/bandwidth
                        heat_map[i][j] = latency
                        if flag: print(f"Heat map [{i}][{j}]: {latency}")
        if flag: print('dataflow heat map generated!')
        return heat_map        
