from hardware_compiler.utils import *
from typing import List, Dict, Tuple, Optional, Any, Set

class BasicHardwareCreator(HardwareCreator):
    """Class for creating basic hardware description"""
    def create_hardware(self, logflag = False) -> Hardware:
        """Create hardware description"""
        hardware = Hardware()
        hardware.array_h = self.array_h
        hardware.array_v = self.array_v
        hardware.modules = []
        self.logflag = logflag
        self.n_Accelerator = self.hierarchy[HierarchyType.ACCELERATOR.value]
        self.n_Bank = self.hierarchy[HierarchyType.BANK.value]
        self.n_Tile = self.hierarchy[HierarchyType.TILE.value]
        self.n_SubTile = self.hierarchy[HierarchyType.SUBTILE.value]
        self.n_PE = self.hierarchy[HierarchyType.PE.value]

        if self.logflag: 
            print('=============basic hardware info====================')
            print('h = ', self.array_h, ', v =', self.array_v)
            for hierarchy, number in self.hierarchy.items(): 
                print('hierarchy:', hierarchy, '-> ', number)

        # Create all modules instance based on the hierarchy

        self.instance_modules(hardware)

        # Connect all modules with dataflow with bandwidth
        self.connect_modules(hardware)
        

        return hardware
        
    def instance_modules(self, hardware: Hardware):
        """Create all modules instance based on the hierarchy"""

        # Iteratively register the modules based on hierarchy info
        for i_Accelerator in range(self.n_Accelerator):
            module = Module(
                HierarchyType.ACCELERATOR.value, 
                FunctionType.DATAFOWARD.value,
                **{HierarchyType.ACCELERATOR.value: i_Accelerator}
            )
            hardware.add_module(module)

            for i_Bank in range(self.n_Bank):
                module = Module(
                    HierarchyType.BANK.value, 
                    FunctionType.DATAFOWARD.value,
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank}
                )
                hardware.add_module(module)

                for i_Tile in range(self.n_Tile):
                    module = Module(
                        HierarchyType.TILE.value, 
                        FunctionType.DATAFOWARD.value,
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile}
                    )
                    hardware.add_module(module)

                    for i_SubTile in range(self.n_SubTile):
                        module = Module(
                            HierarchyType.SUBTILE.value, 
                            FunctionType.DATAFOWARD.value,
                            **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile}
                        )
                        hardware.add_module(module)

                        for i_PE in range(self.n_PE):
                            if i_PE < 3:
                                module = Module(
                                    HierarchyType.PE.value, 
                                    FunctionType.MVM.value,
                                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile, HierarchyType.PE.value: i_PE}
                                )
                                hardware.add_module(module)
                            elif i_PE == 3:
                                module = Module(
                                    HierarchyType.PE.value, 
                                    FunctionType.ACTIVATION.value,
                                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile, HierarchyType.PE.value: i_PE}
                                )
                                hardware.add_module(module)
                            elif i_PE == 4:
                                module = Module(
                                    HierarchyType.PE.value, 
                                    FunctionType.GLU.value,
                                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile, HierarchyType.PE.value: i_PE}
                                )
                                hardware.add_module(module)



    def connect_modules(self,hardware: Hardware):
        for i_Accelerator in range(self.n_Accelerator):
            

            for i_Bank in range(self.n_Bank):
               

                for i_Tile in range(self.n_Tile):
                    

                    for i_SubTile in range(self.n_SubTile):
                        # connect PEs within subtile
                        PE_coords = []
                        PEs = hardware.find_modules(
                            **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile,'hierarchy_type': HierarchyType.PE.value}
                        )
                        for PE in PEs:
                            PE_coords.append(ModuleCoords(**PE.coords))
                        
                        self.connect_wtih_bandwidth(hardware,PE_coords[0], PE_coords[1], 1)
        

                            

    def connect_wtih_bandwidth(self, hardware: Hardware, module_coords1: ModuleCoords, module_coords2: ModuleCoords, bandwidth: int):
        """Connect two modules with a bandwidth"""
        print(module_coords1, module_coords2)