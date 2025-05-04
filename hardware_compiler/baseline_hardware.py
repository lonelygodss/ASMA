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
            'Accelerator to Bank': 50*2,
            'Bank to Tile': 100*2,
            'Tile to PE': 500*2,
        }

        self.latency = { #ns
            FunctionType.MVM.value: 40*self.array_v/512,
            FunctionType.ACTIVATION.value: 0.5*self.array_h/512,
            FunctionType.GLU.value:  0.5*self.array_h/512,
            FunctionType.DATAFOWARD.value:  0.5*self.array_h/512,
            FunctionType.ADD.value:  0.5*self.array_h/512,
        }
        self.energy = { #nJ
            FunctionType.MVM.value: 5.2*pow(self.array_v/512,2),
            FunctionType.ACTIVATION.value: 0.02*self.array_h*4/1000,
            FunctionType.GLU.value: 0.02*self.array_h*4/1000,
            FunctionType.DATAFOWARD.value: 0.02*self.array_h*4/1000,
            FunctionType.ADD.value: 0.02*self.array_h*4/1000,
        }
        self.energy_cost = {
            'Accelerator to Bank': 50/2,
            'Bank to Tile': 10/2,
            'Tile to PE': 3/2,
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
                self.energy[FunctionType.DATAFOWARD.value],
                **{HierarchyType.ACCELERATOR.value: i_Accelerator}
            )
            hardware.add_module(module)

            for i_Bank in range(self.n_Bank):
                module = Module(
                    HierarchyType.BANK.value, 
                    FunctionType.DATAFOWARD.value,
                    self.latency[FunctionType.DATAFOWARD.value],
                    self.energy[FunctionType.DATAFOWARD.value],
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank}
                )
                hardware.add_module(module)

                for i_Tile in range(self.n_Tile):
                    module = Module(
                        HierarchyType.TILE.value, 
                        FunctionType.DATAFOWARD.value,
                        self.latency[FunctionType.DATAFOWARD.value],
                        self.energy[FunctionType.DATAFOWARD.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile}
                    )
                    hardware.add_module(module)

                    for i_PE in range(self.n_PE):
                        module = Module(
                            HierarchyType.PE.value, 
                            FunctionType.MVM.value,
                            self.latency[FunctionType.MVM.value],
                            self.energy[FunctionType.MVM.value],
                            **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: i_PE}
                        )
                        hardware.add_module(module)

                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.ACTIVATION.value,
                        self.latency[FunctionType.ACTIVATION.value],
                        self.energy[FunctionType.ACTIVATION.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: self.n_PE}
                    )
                    hardware.add_module(module)
                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.GLU.value,
                        self.latency[FunctionType.GLU.value],
                        self.energy[FunctionType.GLU.value],
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.PE.value: self.n_PE+1}
                    )
                    hardware.add_module(module)
                    module = Module(
                        HierarchyType.PE.value,
                        FunctionType.ADD.value,
                        self.latency[FunctionType.ADD.value],
                        self.energy[FunctionType.ADD.value],
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
                self.connect_with_bandwidth_bothsides(Banks[i_Bank], Accelerator,self.bandwidth['Accelerator to Bank'],self.energy_cost['Accelerator to Bank'])        
                # find all tiles
                Tiles = hardware.find_modules(
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, 'hierarchy_type':HierarchyType.TILE.value}
                )
                for i_Tile in range(self.n_Tile):
                    # connect Tiles to Bank
                    self.connect_with_bandwidth_bothsides(Banks[i_Bank], Tiles[i_Tile], self.bandwidth['Bank to Tile'],self.energy_cost['Bank to Tile'])
                    # connect Tiles to PE
                    PEs = hardware.find_modules(
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, 'hierarchy_type':HierarchyType.PE.value}
                    )
                    for i_PE in range(self.n_PE+3):
                        self.connect_with_bandwidth_bothsides(Tiles[i_Tile], PEs[i_PE], self.bandwidth['Tile to PE'],self.energy_cost['Tile to PE'])

                            

