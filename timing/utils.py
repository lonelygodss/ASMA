from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable, Iterable, TypeVar

from hardware_compiler.utils import *
from model_compiler.utils import *
from mapping.utils import *
from evaluation.utils import *
import model_compiler.metadata_proess as dataproc
import ast

class SimpleTimedSimulation(Dataflow_parser):
    def __init__(self,model: CompiledModel, hardware: Hardware, mapping: dict,reverse_mapping:dict, dataflow_path: list, connection_info: dict, timelimit:float,flag:bool = False):
        super().__init__(model, hardware, mapping, dataflow_path)
        self.execution_time = 0
        self.transfer_time = 0
        self.reverse_mapping = reverse_mapping
        self.available_tensors = {}
        self.all_tensors = []
        self.input_tensor = ''
        self.output_tensor = ''
        self.connection_info = connection_info
        self.timelimit = timelimit
        self.flag = flag
        self.current_transfer = {}
        self.current_node_transfer = {}
        self.energy = 0
        self.energy_step = 0
        self.tempflag = False
    def run(self):
        """Run the simulation and calculate execution time"""
        # Sort subfunctions
        subfunction_sequence = ['n','m','k','j','i']
        self.model.subfunctions.sort(key = lambda o: tuple(o.coords[k] for k in subfunction_sequence))

        # Transfer-Excecute iteration
        self.initialize()
        while not self.is_finished(self.timelimit):
            # Execute phase
            if self.flag: self.tempflag = True
            worst_excecute_time = 0
            self.energy_step = 0
            self.current_transfer = {}
            self.current_node_transfer = {}
            for subfunction in self.model.subfunctions:
                # Check if the subfunction is ready to execute
                if self.is_calculating(subfunction):
                    # excecute all active subfunctions
                    excecute_time = self.execute_one_step(subfunction)
                    if excecute_time > worst_excecute_time:
                        worst_excecute_time = excecute_time
            # Update execution time
            self.execution_time += worst_excecute_time
            if self.flag: 
                print('execution time:', worst_excecute_time)
                print('execution energy:', self.energy_step)
            # Data Transfer phase
            worst_transfer_time = 0
            for subfunction in self.model.subfunctions:
                # Check if the subfunction is ready to transfer
                if subfunction.is_ready:
                    # Transfer data
                    transfer_time = self.transfer_one_step(subfunction)
            # Update transfer time
            if self.current_transfer.values():
                worst_transfer_time = max(self.current_node_transfer.values())
            if self.flag: 
                print('transfer time:', worst_transfer_time)
                print('transfer energy:', self.energy_step)
            self.transfer_time += worst_transfer_time
        print('Simulation finished at time:', self.execution_time+self.transfer_time,'ns')
        print('Total energy consumption:', self.energy/1000,'uJ')
        return {'time': self.execution_time+self.transfer_time, 'energy': self.energy/1000}



                
            

    def execute_one_step(self, subfunction: SubFunction)->float:
        """Execute one step of the simulation"""
        # Get the module for the subfunction
        module = self.mapping[subfunction]
        time = module.latency
        # Set subfunction ready for transfer phase
        subfunction.is_ready = True
        # Add available tensors
        output_tensors = subfunction.output_tensors
        for tensor in output_tensors:
            tensor_str = self.coords_to_str(**tensor.tensor_id.coords)
            self.available_tensors[tensor_str]='will be available'
        # Silence calculated subfuntion
        subfunction.has_calculated = True
        self.energy += module.energy
        self.energy_step += module.energy
        if self.tempflag: 
            print('excecute subfunction:', subfunction.op_type.value)
            self.tempflag = False
        return time

    def transfer_one_step(self, subfunction: SubFunction)->float:
        """Transfer one step of the simulation"""
        # Get the module for the subfunction
        init_module = self.mapping[subfunction]
        output_tensors = subfunction.output_tensors
        time_cost = 0
        targets = {}
        for tensor in output_tensors:
            # Check if output tensor is available to finish
            if self.coords_to_str(**tensor.tensor_id.coords) == self.output_tensor:
                self.available_tensors[self.output_tensor] = 'available'
                #print('output tensor:', self.output_tensor)
                return time_cost
            
            tensor_tuple = dataproc.get_coord_tuple(tensor.tensor_id)
            target_ids = self.connection_info['tensor_consumers'][tensor_tuple]
            target_coords_sfs = []
            for target_id in target_ids:
                target_coords_sfs.append(dataproc.id_to_coords(target_id))
            for target_coords_sf in target_coords_sfs:
                target_sf = self.model.get_subfunctions_by_coords(**target_coords_sf)[0]
                data_size = 0
                for input_tensor in target_sf.input_tensors:
                    if tensor.tensor_id.coords == input_tensor.tensor_id.coords:
                        data_size = input_tensor.size_params['size_h'] * input_tensor.size_params['size_v']
                        break
                target = self.mapping[target_sf]
                targets[target] = data_size
            self.available_tensors[self.coords_to_str(**tensor.tensor_id.coords)] = 'available'
        for target in targets.keys():
            paths = self.find_hardware_pathes(init_module, target,False)
            if not paths:
                print('error')
            path = paths[0]
            data_size = targets[target]
            data_flow = {
                'data_transfer': data_size
            }
            for module_index in range(len(path)-1):
                module1 = path[module_index]
                module2 = path[module_index+1]

                time=module1.perform_transfer(module2, Dataflow(**data_flow))
                #print('transfer from:',module1.coords,'to:',module2.coords,'with data size:',data_size,'in time:',time)
                energy = module1.transfer_energy(module2, Dataflow(**data_flow))
                self.energy += energy
                self.energy_step += energy
                transfer_path = self.coords_to_str(**module1.coords) + '->' + self.coords_to_str(**module2.coords)
                if transfer_path not in self.current_transfer.keys():
                    self.current_transfer[transfer_path] = time
                else:
                    self.current_transfer[transfer_path] += time
                if module1 not in self.current_node_transfer.keys():
                    self.current_node_transfer[module1] = time
                else:
                    self.current_node_transfer[module1] += time
                if module2 not in self.current_node_transfer.keys():
                    self.current_node_transfer[module2] = time
                else:
                    self.current_node_transfer[module2] += time

        # Reset subfunction state to indicate finished
        subfunction.is_ready = False
        if self.current_transfer.values():

            time_cost = max(self.current_transfer.values())
        return time_cost



        

        
    def is_calculating(self, subfunction: SubFunction)->bool:
        """Check if the subfunction is calculating"""
        # Check if the subfunction is calculating
        all_inputs = subfunction.input_tensors
        if subfunction.has_calculated:
            return False
        for input_tensor in all_inputs:
            input = self.coords_to_str(**input_tensor.tensor_id.coords)
            if not self.available_tensors[input] == 'available':
                return False
        return True
    

    def initialize(self):
        """Feed the first step of the simulation"""
        # Initialize the first step of the simulation
        tensor_with_consumers = self.dict_keys_to_str_list(self.connection_info['tensor_consumers'])
        tensor_with_producers = self.dict_keys_to_str_list(self.connection_info['tensor_producers'])
        for tensor in tensor_with_consumers:
            if tensor not in tensor_with_producers:
                self.input_tensor = tensor
                #print('input tensor:', tensor)
            self.all_tensors.append(tensor)
            self.available_tensors[tensor] = 'not_available'

        for tensor in tensor_with_producers:
            if tensor not in tensor_with_consumers:
                self.output_tensor = tensor
                #print('output tensor:', tensor)
                self.available_tensors[tensor] = 'not_available'
            self.all_tensors.append(tensor)
        # Initialize the available tensors
        self.available_tensors[self.input_tensor] =  'available'
        self.available_tensors[self.output_tensor] = 'not_available'

        # Deactivate all subfunctions
        for subfunction in self.model.subfunctions:
            subfunction.is_ready = False
            

    def is_finished(self, sim_time:float)->bool:
        """Check if the simulation is finished"""
        # Check if the simulation is finished
        finished = False
        if self.execution_time >= sim_time:
            finished = True
            return finished
        elif self.available_tensors[self.output_tensor] == 'available':
            finished = True
            return finished
        else:
            return finished

    def coords_to_str(self, **coords)->str:
        """Convert coordinates to string"""
        coords_str = ",".join(str((k,v)) for k, v in sorted(coords.items()))
        return coords_str
    
    def dict_keys_to_str_list(self, dict:Dict)->List[str]:
        """Convert dictionary keys to string"""
        str_list = []
        dict_keys = dict.keys()
        for tupels in dict_keys:
            str_list.append(','.join(str(i)for i in tupels))
        return str_list
