from hardware_compiler.utils import *
from hardware_compiler.basic_hardware import *

def main():
    logflag = True
    array_h = 1024      # Horizontal size of CIM array
    array_v = 1024      # Vertical size of CIM array
    hierarchy = {
        HierarchyType.ACCELERATOR.value: 1,
        HierarchyType.BANK.value: 1,
        HierarchyType.TILE.value: 1,
        HierarchyType.SUBTILE.value: 16,
        HierarchyType.PE.value: 5
    }
    creator = BasicHardwareCreator(array_h, array_v, **hierarchy)
    hardware = creator.create_hardware(logflag)


if __name__ == "__main__":
    main()