from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
import model_compiler.metadata_proess as dataproc
from hardware_compiler.utils import *
from hardware_compiler.basic_hardware import *
from mapping.subtile_parallel_mapping import TrivialMapping
from timing.utils import SimpleTimedSimulation
from evaluation.utils import Dataflow_parser


# This main compile with ParrallelCompiler, create basic_hardware, and map with trivil mapping

def main():
    # Example usage
    # Define model parameters
    hidden_dim = 4096  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 11008    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 512      # Horizontal size of CIM array
    array_v = 512      # Vertical size of CIM array
    
    logflag = False

    # Create model
    model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    # print("Original Model:")
    # print(model)
    # print("\n" + "="*80 + "\n")
    
    # Compile model
    compiler = ParallelCompiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)
    
    print("Compiled Model:")
    print(f"Total subfunctions: {len(compiled_model.subfunctions)}")
    


    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 3,
        HierarchyType.SUBTILE.value: 64,
        HierarchyType.PE.value: 5
    }
    creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
    hardware = creator.create_hardware(logflag)
    hardware.generate_hardware_graph()
    
    print("Hardware creation and visualization complete!")
    
    mapping = TrivialMapping(compiled_model, hardware)
    mapping.map()
    print("Mapping complete!")

    connection_info = dataproc.parse_compute_graph(compiled_model)
    print("Compute graph parsing complete!")

    simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping, connection_info['data_flow_paths'],connection_info,100000,False)
    simulator.run() 
    print("Simulation complete!")

    parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
    parser.parse_dataflow(logflag)


if __name__ == "__main__":
    main()