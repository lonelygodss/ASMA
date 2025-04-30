from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
import model_compiler.metadata_proess as dataproc
from hardware_compiler.utils import *
from hardware_compiler.baseline_hardware import *
from hardware_compiler.basic_hardware import *
from mapping.baseline_baseline_mapping import BaselineMapping
from timing.utils import SimpleTimedSimulation


# This main compile with BaselineCompiler, create basic_hardware, and map with baseline mapping

def main():
    # Example usage
    # Define model parameters
    hidden_dim = 4096  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 11008    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 2048      # Horizontal size of CIM array
    array_v = 2048      # Vertical size of CIM array
    
        # version control

    logflag = False

    # Create model
    model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    # print("Original Model:")
    # print(model)
    # print("\n" + "="*80 + "\n")
    
    # Compile model
    compiler_baseline = BaselineCompiler(array_h, array_v)
    compiler_parallel = ParallelCompiler(array_h, array_v)
    compiled_model_baseline = compiler_baseline.divide_model(model)
    compiled_model_parallel = compiler_parallel.divide_model(model)
    
    print("Compiled baseline Model:")
    print(f"Total subfunctions: {len(compiled_model_baseline.subfunctions)}")
    print("Compiled parallel Model:")
    print(f"Total subfunctions: {len(compiled_model_parallel.subfunctions)}")

    hierarchy_baseline = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 5,
        HierarchyType.PE.value: 16
    }
    hierarchy_subtile = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 4,
        HierarchyType.SUBTILE.value: 16,
        HierarchyType.PE.value: 5
    }
    creator_baseline = BaselineHardwareCreator(array_h, array_v, **hierarchy_baseline)
    creator_subtile = BasicHardwareCreator(array_h, array_v, **hierarchy_subtile)
    hardware_baseline = creator_baseline.create_hardware(logflag)
    hardware_subtile = creator_subtile.create_hardware(logflag)
    hardware_baseline.generate_hardware_graph()
    hardware_subtile.generate_hardware_graph()
    
    print("Hardware creation complete!")
    
    mapping_baseline_baseline = BaselineMapping(compiled_model_baseline, hardware_baseline)
    mapping_baseline_baseline.map()
    print("baseline_baseline Mapping complete!")
    mapping_subtile_baseline = BaselineMapping(compiled_model_baseline, hardware_subtile)
    mapping_subtile_baseline.map()
    print("subtile_baseline Mapping complete!")
    mapping_subtile_parallel = BaselineMapping(compiled_model_parallel, hardware_subtile)
    mapping_subtile_parallel.map()
    print("subtile_parallel Mapping complete!")

    connection_info_baseline = dataproc.parse_compute_graph(compiled_model_baseline)
    print("Compute graph parsing baseline complete!")
    connection_info_parallel = dataproc.parse_compute_graph(compiled_model_parallel)
    print("Compute graph parsing parallel complete!")


    simulator = SimpleTimedSimulation(compiled_model_baseline, hardware_baseline, mapping_baseline_baseline.mapping,mapping_baseline_baseline.reverse_mapping, connection_info_baseline['data_flow_paths'],connection_info_baseline,1000000,False)
    simulator.run() 
    print("baseline_baseline Simulation complete!")
    simulator = SimpleTimedSimulation(compiled_model_baseline, hardware_subtile, mapping_subtile_baseline.mapping,mapping_subtile_baseline.reverse_mapping, connection_info_baseline['data_flow_paths'],connection_info_baseline,1000000,False)
    simulator.run()
    print("subtile_baseline Simulation complete!")
    simulator = SimpleTimedSimulation(compiled_model_parallel, hardware_subtile, mapping_subtile_parallel.mapping,mapping_subtile_parallel.reverse_mapping, connection_info_parallel['data_flow_paths'],connection_info_parallel,1000000,False)
    simulator.run()
    print("subtile_parallel Simulation complete!")
    



if __name__ == "__main__":
    main()