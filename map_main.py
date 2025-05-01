from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
import model_compiler.metadata_proess as dataproc
from hardware_compiler.utils import *
from hardware_compiler.basic_hardware import *
from mapping.subtile_parallel_mapping import TrivialMapping


def main():
    # Example usage
    # Define model parameters
    hidden_dim = 4096  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 11008    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 2048      # Horizontal size of CIM array
    array_v = 2048      # Vertical size of CIM array
    
    logflag = True

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
    
    # Print some statistics
    op_counts = {}
    for subfunc in compiled_model.subfunctions:
        op_type = subfunc.op_type.value
        if op_type not in op_counts:
            op_counts[op_type] = 0
        op_counts[op_type] += 1

        hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 5,
        HierarchyType.SUBTILE.value: 16,
        HierarchyType.PE.value: 5
    }
    creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
    hardware = creator.create_hardware(logflag)
    
    print("Hardware creation and visualization complete!")
    
    mapping = TrivialMapping(compiled_model, hardware)
    mapping.map()


if __name__ == "__main__":
    main()