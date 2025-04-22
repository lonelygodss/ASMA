import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.advanced_compiler import AdvancedCompiler
from model_compiler.utils import TensorId

def main():
    # Create a GLU FFN model
    hidden_dim = 4096
    ffn_dim = 11008
    layer_idx = 1
    
    # Create the high-level model
    model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    
    # Initialize compiler with hardware constraints
    array_h = 2048
    array_v = 2048
    compiler = AdvancedCompiler(array_h, array_v)
    
    # Method 1: Default compilation
    print("=== Default Compilation ===")
    compiled_model = compiler.compile_glu_ffn(model)
    print(f"Default compilation generated {len(compiled_model.subfunctions)} subfunctions")
    
    # Method 2: Custom dataflow - function-by-function compilation
    print("\n=== Custom Function-by-Function Compilation ===")
    custom_model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    custom_compiled = compiler.current_compiled_model = type(compiled_model)()
    
    # Define custom tensor IDs for connections
    up_proj_output = TensorId(k=1, m=1, n=layer_idx, custom='up_proj_output')
    gate_proj_output = TensorId(k=2, m=1, n=layer_idx, custom='gate_proj_output')
    up_copy_output = TensorId(k=1, m=2, n=layer_idx, custom='up_copy_output')
    gate_act_output = TensorId(k=2, m=2, n=layer_idx, custom='gate_act_output')
    glu_output = TensorId(k=1, m=3, n=layer_idx, custom='glu_output')
    
    # Find functions by position
    up_proj = model.get_functions_by_coords(k=1, m=1, n=layer_idx)[0]
    gate_proj = model.get_functions_by_coords(k=2, m=1, n=layer_idx)[0]
    up_copy = model.get_functions_by_coords(k=1, m=2, n=layer_idx)[0]
    activation = model.get_functions_by_coords(k=2, m=2, n=layer_idx)[0]
    glu = model.get_functions_by_coords(k=1, m=3, n=layer_idx)[0]
    down_proj = model.get_functions_by_coords(k=1, m=4, n=layer_idx)[0]
    
    # Manual compilation with custom connections
    input_tensor = TensorId(k=0, m=0, n=layer_idx-1)
    
    # 1. Compile up_proj with custom output
    compiler.compile_mvm_function(
        up_proj, 
        custom_compiled,
        custom_input_id=input_tensor,
        custom_output_id=up_proj_output
    )
    
    # 2. Compile gate_proj with custom output
    compiler.compile_mvm_function(
        gate_proj, 
        custom_compiled,
        custom_input_id=input_tensor,
        custom_output_id=gate_proj_output
    )
    
    # 3. Compile up_copy with custom connections
    compiler.compile_elementwise_function(
        up_copy, 
        custom_compiled,
        custom_input_id=up_proj_output,
        custom_output_id=up_copy_output
    )
    
    # 4. Compile activation with custom connections
    compiler.compile_elementwise_function(
        activation, 
        custom_compiled,
        custom_input_id=gate_proj_output,
        custom_output_id=gate_act_output
    )
    
    # 5. Compile GLU with custom connections
    compiler.compile_glu_function(
        glu, 
        custom_compiled,
        custom_input1_id=up_copy_output,
        custom_input2_id=gate_act_output,
        custom_output_id=glu_output
    )
    
    # 6. Compile down_proj with custom connections
    compiler.compile_mvm_function(
        down_proj, 
        custom_compiled,
        custom_input_id=glu_output
    )
    
    print(f"Custom compilation generated {len(custom_compiled.subfunctions)} subfunctions")
    
    # Method 3: Using the configuration API for custom dataflow
    print("\n=== Configuration-based Custom Dataflow ===")
    dataflow_config = {
        'functions': [
            {
                'id': {'k': 1, 'm': 1, 'n': layer_idx},  # up_proj
                'inputs': {0: input_tensor},
                'output': up_proj_output
            },
            {
                'id': {'k': 2, 'm': 1, 'n': layer_idx},  # gate_proj
                'inputs': {0: input_tensor},
                'output': gate_proj_output
            },
            {
                'id': {'k': 1, 'm': 2, 'n': layer_idx},  # up_copy
                'inputs': {0: up_proj_output},
                'output': up_copy_output
            },
            {
                'id': {'k': 2, 'm': 2, 'n': layer_idx},  # activation
                'inputs': {0: gate_proj_output},
                'output': gate_act_output
            },
            {
                'id': {'k': 1, 'm': 3, 'n': layer_idx},  # GLU
                'inputs': {1: up_copy_output, 2: gate_act_output},
                'output': glu_output
            },
            {
                'id': {'k': 1, 'm': 4, 'n': layer_idx},  # down_proj
                'inputs': {0: glu_output},
                'output': None  # Use default
            }
        ]
    }
    
    config_compiled = compiler.create_custom_dataflow(model, dataflow_config)
    print(f"Configuration-based compilation generated {len(config_compiled.subfunctions)} subfunctions")


if __name__ == "__main__":
    main()
