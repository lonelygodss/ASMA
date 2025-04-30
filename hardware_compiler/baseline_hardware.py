from typing import List, Dict, Tuple, Optional, Any, Set
from hardware_compiler.utils import *

class BaselineHardwareCreator(HardwareCreator):
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
        self.n_PE = self.hierarchy[HierarchyType.PE.value]
        self.bandwidth = {
            'Accelerator to Bank': 5,
            'Bank to Tile': 5,
            'Tile to PE': 2,
        }
        self.latency = {
            FunctionType.MVM.value: 0.5,
            FunctionType.ACTIVATION.value: 1.1,
            FunctionType.GLU.value: 0.7,
            FunctionType.DATAFOWARD.value: 0.03,
            FunctionType.ADD.value: 0.6,
        }

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
                self.latency[FunctionType.DATAFOWARD.value],
                **{HierarchyType.ACCELERATOR.value: i_Accelerator}
            )
            hardware.add_module(module)

            for i_Bank in range(self.n_Bank):
                module = Module(
                    HierarchyType.BANK.value, 
                    FunctionType.DATAFOWARD.value,
                    self.latency[FunctionType.DATAFOWARD.value],
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank}
                )
                hardware.add_module(module)

                for i_Tile in range(self.n_Tile):
                    module = Module(
                        HierarchyType.TILE.value, 
                        FunctionType.DATAFOWARD.value,
                        self.latency[FunctionType.DATAFOWARD.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile}
                    )
                    hardware.add_module(module)

                    for i_PE in range(self.n_PE):
                        module = Module(
                            HierarchyType.PE.value, 
                            FunctionType.MVM.value,
                            self.latency[FunctionType.MVM.value],
                            **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: i_PE}
                        )
                        hardware.add_module(module)

                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.ACTIVATION.value,
                        self.latency[FunctionType.ACTIVATION.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: self.n_PE}
                    )
                    hardware.add_module(module)
                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.GLU.value,
                        self.latency[FunctionType.GLU.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: self.n_PE+1}
                    )
                    hardware.add_module(module)
                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.ADD.value,
                        self.latency[FunctionType.ADD.value]
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: self.n_PE+2}
                    )
                    hardware.add_module(module)

    def connect_modules(self,hardware: Hardware):
        Accelerator = hardware.find_module(
            **{'hierarchy_type':HierarchyType.ACCELERATOR.value}
        )
        for i_Accelerator in range(self.n_Accelerator):
            # find all banks
            Banks = hardware.find_modules(
                **{HierarchyType.ACCELERATOR.value: i_Accelerator, 'hierarchy_type':HierarchyType.BANK.value}
            )
            
            for i_Bank in range(self.n_Bank):
                # connect Bank to Accelerator
                self.connect_with_bandwidth_bothsides(Banks[i_Bank], Accelerator,self.bandwidth['Accelerator to Bank'])        
                # find all tiles
                Tiles = hardware.find_modules(
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, 'hierarchy_type':HierarchyType.TILE.value}
                )
                for i_Tile in range(self.n_Tile):
                    # connect Tiles to Bank
                    self.connect_with_bandwidth_bothsides(Banks[i_Bank], Tiles[i_Tile], self.bandwidth['Bank to Tile'])
                    # connect Tiles to PE
                    PEs = hardware.find_modules(
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, 'hierarchy_type':HierarchyType.PE.value}
                    )
                    for i_PE in range(self.n_PE+3):
                        self.connect_with_bandwidth_bothsides(Tiles[i_Tile], PEs[i_PE], self.bandwidth['Tile to PE'])

                            

