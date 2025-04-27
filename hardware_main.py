from hardware_compiler.utils import *
from hardware_compiler.basic_hardware import *
from hardware_compiler.metadata_process2 import visualize_hardware
import os

def main():
    logflag = True
    array_h = 1024      # Horizontal size of CIM array
    array_v = 1024      # Vertical size of CIM array
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 5,
        HierarchyType.TILE.value: 5,
        HierarchyType.SUBTILE.value: 16,
        HierarchyType.PE.value: 5
    }
    creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
    hardware = creator.create_hardware(logflag)
    
    # Create output directory if it doesn't exist
    output_dir = "hardware_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate hardware visualizations
    visualize_hardware(hardware, output_dir=output_dir, view=True)
    
    print("Hardware creation and visualization complete!")


if __name__ == "__main__":
    main()