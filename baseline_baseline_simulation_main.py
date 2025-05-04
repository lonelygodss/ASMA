from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
import model_compiler.metadata_proess as dataproc
from hardware_compiler.utils import *
from hardware_compiler.baseline_hardware import *
from mapping.baseline_baseline_mapping import BaselineMapping
from evaluation.utils import Dataflow_parser

from timing.utils import SimpleTimedSimulation


# This main compile with BaselineCompiler, create basic_hardware, and map with baseline mapping

def main():
    # Example usage
    # Define model parameters
    hidden_dim = 1024  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 8192    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 512      # Horizontal size of CIM array
    array_v = 512      # Vertical size of CIM array
    
        # version control
    filename = "baseline/"

    # file separation from git
    filedir = "compiled_model/"+filename

    logflag = False
    for i in range(7):
        ffn_dim = 2**(i+9)
        print("ffn_dim:", ffn_dim)
        # Create model
        model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
        # print("Original Model:")
        # print(model)
        # print("\n" + "="*80 + "\n")
        
        # Compile model
        compiler = BaselineCompiler(array_h, array_v)
        compiled_model = compiler.divide_model(model)
        
        print("Compiled Model:")
        print(f"Total subfunctions: {len(compiled_model.subfunctions)}")


        hierarchy = {
            HierarchyType.ACCELERATOR.value: 1,
            HierarchyType.BANK.value: 4,
            HierarchyType.TILE.value: 64,
            HierarchyType.PE.value: 16
        }
        creator = BaselineHardwareCreator(array_h, array_v, **hierarchy)
        hardware = creator.create_hardware(logflag)
        hardware.generate_hardware_graph()
        
        print("Hardware creation and visualization complete!")
        
        mapping = BaselineMapping(compiled_model, hardware)
        mapping.map()
        print("Mapping complete!")

        connection_info = dataproc.parse_compute_graph(compiled_model,False,extract_paths=False)
        print("Compute graph parsing complete!")

        simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping, connection_info['data_flow_paths'],connection_info,1000000,False)
        simulator.run() 
        print("Simulation complete!")
        
        # parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
        # parser.parse_dataflow(logflag)


if __name__ == "__main__":
    main()