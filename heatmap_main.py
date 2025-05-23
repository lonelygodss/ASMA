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
from mapping.subtile_parallel_mapping import TrivialMapping
from mapping.subtile_scatter_mapping import ScatterMapping
from mapping.baseline_baseline_mapping import BaselineMapping
from evaluation.utils import Dataflow_parser
from timing.utils import SimpleTimedSimulation
from evaluation.visualize import *


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
    print("ffn_dim:", ffn_dim)
    # Create model
    model = create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    # print("Original Model:")
    # print(model)
    # print("\n" + "="*80 + "\n")
#==================bb==================    
    # Compile model
    compiler = BaselineCompiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)


    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 3,
        HierarchyType.PE.value: 12
    }
    creator = BaselineHardwareCreator(array_h, array_v, **hierarchy)
    hardware = creator.create_hardware(logflag)
    hardware.generate_hardware_graph()
    
    print("Hardware creation and visualization complete!")
    
    mapping = BaselineMapping(compiled_model, hardware)
    mapping.map()
    print("Mapping complete!")

    connection_info = dataproc.parse_compute_graph(compiled_model,False,extract_paths=True)
    print("Compute graph parsing complete!")
    
    parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
    [baseline_data_heatmap,baseline_latency_heatmap] = parser.parse_dataflow(logflag)

#====================sp=====================
    compiler = ParallelCompiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)
    


    subtile_hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 12,
        HierarchyType.PE.value: 7
    }
    creator = BasicHardwareCreator(array_h, array_v, **subtile_hierarchy)
    hardware = creator.create_hardware(logflag)
    hardware.generate_hardware_graph()
    
    print("Hardware creation and visualization complete!")
    
    mapping = TrivialMapping(compiled_model, hardware)
    mapping.map()
    print("Mapping complete!")

    connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=True)
    print("Compute graph parsing complete!")

    # simulator = SimpleTimedSimulation(compiled_model, hardware, mapping.mapping,mapping.reverse_mapping, connection_info['data_flow_paths'],connection_info,100000,True)
    # simulator.run() 
    # print("Simulation complete!")

    parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
    [parallel_data_heatmap, parallel_latency_heatmap] = parser.parse_dataflow(logflag)

#========================ss======================

    compiler = ScatterCompiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)

    

    creator = BasicHardwareCreator(array_h, array_v, **subtile_hierarchy)
    hardware = creator.create_hardware(logflag)
    hardware.generate_hardware_graph()
    
    print("Hardware creation and visualization complete!")
    
    mapping = ScatterMapping(compiled_model, hardware)
    mapping.map()
    print("Mapping complete!")
    connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=True)
    print("Compute graph parsing complete!")

    parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
    [scatter_data_heatmap, scatter_latency_heatmap] = parser.parse_dataflow(logflag)

#====================ssp======================
    # Compile model
    compiler = ScatterParallelCompiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)
    creator = BasicHardwareCreator(array_h, array_v, **subtile_hierarchy)
    hardware = creator.create_hardware(logflag)
    hardware.generate_hardware_graph()
    
    print("Hardware creation and visualization complete!")
    
    mapping = ScatterMapping(compiled_model, hardware)
    mapping.map()
    print("Mapping complete!")
    connection_info = dataproc.parse_compute_graph(compiled_model,extract_paths=True)
    print("Compute graph parsing complete!")

    parser = Dataflow_parser(compiled_model, hardware, mapping.mapping, connection_info['data_flow_paths'])
    [scatter_parallel_data_heatmap,scatter_parallel_latency_map] = parser.parse_dataflow(logflag)

    heatmaplist1 = [parallel_data_heatmap, scatter_data_heatmap, scatter_parallel_data_heatmap]
    # heatmaplist2 = [parallel_latency_heatmap,scatter_latency_heatmap,scatter_parallel_latency_map]
    # visualize_multiple_heat_maps(heatmaplist1,log_scale= True,show=True, save_path = "heatmap-3")
    # visualize_heat_map(baseline_data_heatmap,log_scale=True,show=True,save_path="heatmap-1")
    fig_combined = visualize_combined_heat_maps(
    baseline_heatmap=baseline_data_heatmap,
    comparison_heatmaps=heatmaplist1,
    baseline_title="Baseline", # Custom title for baseline
    comparison_titles=None, # Custom titles for comparison plots
    main_title="Full Dataflow Analysis",
    log_scale=True,
    show=True,
    zeros_black=True,
    cmap='coolwarm', # Example: using 'viridis' colormap
    save_path="heatmap_combined_analysis" # Will save as .png and .pdf
)
    list_of_all_heatmaps = [
        baseline_data_heatmap,
        parallel_data_heatmap,
        scatter_data_heatmap,
        scatter_parallel_data_heatmap
    ]

    # Create a corresponding list of names for each heatmap
    list_of_heatmap_names = [
        "Baseline",
        "Standard-Parallel",
        "Scatter-Parallel",
        "Mutual-Parallel"
    ]

    # Define the desired output path for your CSV file
    output_statistics_csv = "heatmap_analysis_statistics.csv"

    # Call the function to calculate and save statistics
    # success = save_heatmap_statistics_to_csv(
    #     list_of_all_heatmaps,
    #     list_of_heatmap_names,
    #     output_statistics_csv
    # )

    # if success:
    #     print("Statistics CSV generated successfully.")
    # else:
    #     print("Failed to generate statistics CSV.")


if __name__ == "__main__":
    main()