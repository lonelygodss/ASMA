from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable

from hardware_compiler.utils import *

from model_compiler.utils import *


class Map_Compiledmodel_to_Hardware:
    """Map the compiled model to hardware"""
    def __init__(self, compiled_model: CompiledModel, hardware: Hardware):
        self.compiled_model = compiled_model
        self.hardware = hardware
        self.mapping = {}  # Mapping from compiled model to hardware

    def map(self):
        """Map the compiled model to hardware"""
        # Implement the mapping logic here
        pass
