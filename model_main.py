from model_compiler.GLU_ffn import create_glu_ffn_model
from model_compiler.parallel_compiler import ParallelCompiler
from model_compiler.function_wise_compiler import FunctionWiseCompiler
from model_compiler.basline_compiler import BaselineCompiler
from model_compiler.scatter_compiler import ScatterCompiler
from model_compiler.scatter_parallel_compiler import ScatterParallelCompiler
import model_compiler.metadata_proess as dataproc


def main():
    # Example usage
    # Define model parameters
    hidden_dim = 4096  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 11008    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 1024      # Horizontal size of CIM array
    array_v = 1024      # Vertical size of CIM array
    
    # version control
    filename = "scatter/"

    # file separation from git
    filedir = "compiled_model/"+filename

    # Generate outputfile or not
    data_flag = True

    # Create model
    model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    # print("Original Model:")
    # print(model)
    # print("\n" + "="*80 + "\n")
    
    # Compile model
    compiler = ScatterCompiler(array_h, array_v)
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
    
    # print("\nOperation counts:")
    # for op_type, count in op_counts.items():
    #     print(f"  {op_type}: {count}")
    
    # Print a sample of subfunctions
    # print("\nSample subfunctions:")
    # for i, subfunc in enumerate(compiled_model.subfunctions[:]):
    #      print(f"  {i+1}. {subfunc}")
    
    # if len(compiled_model.subfunctions) > 50:
    #     print(f"  ... and {len(compiled_model.subfunctions) - 50} more")

    # Parse and analyze the compute graph
    connection_info = dataproc.parse_compute_graph(compiled_model)

    if data_flag:
        # Visualize the compiled model with shorter labels
        dataproc.visualize_compiled_model(compiled_model, filedir+ "ffn_compiled_model")
        # Alternative layered visualization with shorter labels
        dataproc.visualize_compiled_model_layered(compiled_model, filedir+ "ffn_compiled_model_layered")
    
        # Simplified visualization focusing on dataflow
        dataproc.visualize_compiled_model_simple(compiled_model, filedir+ "ffn_compiled_model_simple")

        # Save the compute graph
        dataproc.save_compute_graph(connection_info, filedir+ "ffn_compute_graph.json")
        # Visualize the compute graph
        dataproc.visualize_compute_graph_graphviz(connection_info, filedir+ "ffn_compute_graph_graphviz")
    
    #Analyze the compute graph
    analysis = dataproc.analyze_compute_graph(connection_info)
    print("\nCompute Graph Analysis:")
    for key, value in analysis.items():  # Fixed: using analysis instead of dataproc.analysis
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()