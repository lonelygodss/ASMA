from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
from model_compiler.scatter_parallel_compiler import ScatterParallelCompiler
import model_compiler.metadata_proess as dataproc
from hardware_compiler.utils import *
from hardware_compiler.baseline_hardware import *
from hardware_compiler.basic_hardware import *
from mapping.baseline_baseline_mapping import BaselineMapping
from mapping.subtile_scatter_mapping import ScatterMapping
from mapping.subtile_parallel_mapping import TrivialMapping

from timing.utils import SimpleTimedSimulation
import csv
import os
from datetime import datetime

# This main compile with BaselineCompiler, create basic_hardware, and map with baseline mapping

def main():
    # Example usage
    # Define model parameters
    hidden_dim = 1024  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 16384    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 2048      # Horizontal size of CIM array
    array_v = 2048      # Vertical size of CIM array
    
    # Create a list to store all simulation results
    results = []
    # Add header row
    results.append(["Compiler", "Array Size", "Hidden Dimension", "Time", "Energy"])
    
    logflag = False
    for n in range(3):
        array_h = 2**(n+8)
        array_v = 2**(n+8)
        print("array_size:", array_h)
        for i in range(4):
            hidden_dim = 2**(i+10)
            print("hidden_dim:", hidden_dim)
            # Create model
            model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)

            # Baseline compiler
            compiler = BaselineCompiler(array_h, array_v)
            compiled_model = compiler.divide_model(model)

            hierarchy = {
                HierarchyType.ACCELERATOR.value: 1,
                HierarchyType.BANK.value: 16,
                HierarchyType.TILE.value: 64,
                HierarchyType.PE.value: 16
            }
            creator = BaselineHardwareCreator(array_h, array_v, **hierarchy)
            hardware = creator.create_hardware(logflag)
            hardware.generate_hardware_graph()
            
            
            mapping = BaselineMapping(compiled_model, hardware)
            mapping.map()
            print("Mapping complete!")

            connection_info = dataproc.parse_compute_graph(compiled_model,False,extract_paths=False)

            simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping, connection_info['data_flow_paths'],connection_info,1000000,False)
            result = simulator.run() 
            time = result['time']
            energy = result['energy']
            print("Simulation complete!")
            
            # Store baseline results
            results.append(["b_b", array_h, hidden_dim, time, energy])
            
    #==============================================
            hierarchy = {
                HierarchyType.ACCELERATOR.value: 1,
                HierarchyType.BANK.value: 1,
                HierarchyType.TILE.value: 32,
                HierarchyType.SUBTILE.value: 64,
                HierarchyType.PE.value: 7
            }
            compiler = ParallelCompiler(array_h, array_v)
            compiled_model = compiler.divide_model(model)
            
            creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
            hardware = creator.create_hardware(logflag)
            hardware.generate_hardware_graph()
                    
            mapping = TrivialMapping(compiled_model, hardware)
            mapping.map()
            print("Mapping complete!")

            connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=False)

            simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping, connection_info['data_flow_paths'],connection_info,100000,False)
            result = simulator.run() 
            time = result['time']
            energy = result['energy']
            print("Simulation complete!")
            
            # Store ParallelCompiler results
            results.append(["s_p", array_h, hidden_dim, time, energy])
    #==============================================
            model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
            compiler = ScatterCompiler(array_h, array_v)
            compiled_model = compiler.divide_model(model)

            creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
            hardware = creator.create_hardware(logflag)
            hardware.generate_hardware_graph()
            
            
            mapping = ScatterMapping(compiled_model, hardware)
            mapping.map()
            print("Mapping complete!")
            connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=False)


            simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping,connection_info['data_flow_paths'],connection_info,100000,False)
            result = simulator.run() 
            time = result['time']
            energy = result['energy']
            print("Simulation complete!")
            
            # Store 3rd results
            results.append(["s_s", array_h, hidden_dim, time, energy])
    #==============================================
            compiler = ScatterParallelCompiler(array_h, array_v)                        
            compiled_model = compiler.divide_model(model)

            creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
            hardware = creator.create_hardware(logflag)
            hardware.generate_hardware_graph()

            mapping = ScatterMapping(compiled_model, hardware)
            mapping.map()
            print("Mapping complete!")
            connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=False)

            simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping,connection_info['data_flow_paths'],connection_info,100000,False)
            result = simulator.run() 
            time = result['time']
            energy = result['energy']
            print("Simulation complete!")
            
            # Store 4th results
            results.append(["s_sp", array_h, hidden_dim, time, energy])

    # Export results to CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
    
    print(f"Results exported to {filename}")

if __name__ == "__main__":
    main()