from hardware_compiler.utils import *
from typing import List, Dict, Tuple, Optional, Any, Set

class BasicHardwareCreator(HardwareCreator):
    """Class for creating basic hardware description"""
    def create_hardware(self) -> Hardware:
        """Create hardware description"""
        hardware = Hardware()
        hardware.array_h = self.array_h
        hardware.array_v = self.array_v
        hardware.modules = []
        self.logflag = False

        if self.logflag: 
            print('=============basic hardware info====================')
            print('h = ', self.array_h, ', v =', self.array_v)
            for hierarchy, number in self.hierarchy.items: 
                print('hierarchy:', hierarchy, '-> number', number)

        # Create all modules instance based on the hierarchy

        self.instance_modules()

        # Connect all modules with dataflow with bandwidth
        self.connect_modules()
        

        return hardware
        
    def instance_modules(self):
        pass
    def connect_modules(self):
        pass
    