from hardware_compiler.utils import *
from typing import List, Dict, Tuple, Optional, Any, Set

class ScatterHardwareCreator(HardwareCreator):
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
        self.bandwidth = {
            'Accelerator to Bank': 100,
            'Bank to Tile': 100,
            'Tile to Subtile': 50,
            'Subtile to Subtile': 10,
            'Subtile to PE': 1,
            'PE to PE': 1,
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
            # find all banks
            Banks = hardware.find_modules(
                **{HierarchyType.ACCELERATOR.value: i_Accelerator, 'hierarchy_type':HierarchyType.BANK.value}
            )
            
            for i_Bank in range(self.n_Bank):
                # find all tiles
                Tiles = hardware.find_modules(
                    **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, 'hierarchy_type':HierarchyType.TILE.value}
                )
                for i_Tile in range(self.n_Tile):
                    # connect Tiles to Bank
                    self.connect_with_bandwidth_bothsides(Banks[i_Bank], Tiles[i_Tile], self.bandwidth['Bank to Tile'])

                    # find all subtiles
                    SubTiles = hardware.find_modules(
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, 'hierarchy_type':HierarchyType.SUBTILE.value}
                    )
                    
                    # connect all GLUs with each other
                    GLUs = hardware.find_modules(
                        **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, 'hierarchy_type':HierarchyType.PE.value, 'function_type': FunctionType.GLU.value}
                    )

                    for i_GLUs in range(len(GLUs)):
                        self.connect_with_bandwidth_bothsides(GLUs[i_GLUs], GLUs[(i_GLUs+1)%len(GLUs)], self.bandwidth['Subtile to Subtile'])

                    for i_SubTile in range(self.n_SubTile):
                        #connect subtiles to tiles
                        self.connect_with_bandwidth_bothsides(Tiles[i_Tile], SubTiles[i_SubTile], self.bandwidth['Tile to Subtile'])
                        #find all PEs
                        PEs = hardware.find_modules(
                            **{HierarchyType.ACCELERATOR.value: i_Accelerator, HierarchyType.BANK.value: i_Bank, HierarchyType.TILE.value: i_Tile, HierarchyType.SUBTILE.value: i_SubTile,'hierarchy_type': HierarchyType.PE.value}
                        )
                        
                        # connect PEs within subtile
                        self.connect_with_bandwidth_single(SubTiles[i_SubTile], PEs[0], self.bandwidth['Subtile to PE'])
                        self.connect_with_bandwidth_single(SubTiles[i_SubTile], PEs[1], self.bandwidth['Subtile to PE'])
                        self.connect_with_bandwidth_single(PEs[2], SubTiles[i_SubTile], self.bandwidth['Subtile to PE'])
                        
                        # connect PEs with each other
                        self.connect_with_bandwidth_single(PEs[0], PEs[3], self.bandwidth['PE to PE'])
                        self.connect_with_bandwidth_single(PEs[1], PEs[4], self.bandwidth['PE to PE'])
                        self.connect_with_bandwidth_single(PEs[3], PEs[4], self.bandwidth['PE to PE'])
                        self.connect_with_bandwidth_single(PEs[4], PEs[2], self.bandwidth['PE to PE'])
        

                            

