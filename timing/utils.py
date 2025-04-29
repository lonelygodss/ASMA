from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable, Iterable, TypeVar

from hardware_compiler.utils import *
from model_compiler.utils import *
from mapping.utils import *
from evaluation.utils import *
import model_compiler.metadata_proess as dataproc

class TimedSimulation(Dataflow_parser):
    def __init__(self):
        self.execution_time = 0

    def run(self):
        """Run the simulation and calculate execution time"""
        # Sort subfunctions
        subfunction_sequence = ['n','m','k','j','i']
        self.model.subfunctions.sort(key = lambda o: tuple(o.coords[k] for k in subfunction_sequence))

        for subfunction in self.model.subfunctions:
            # excecute in order
            self.execute_one_step(subfunction)

    def execute_one_step(self, subfunction: SubFunction):
        """Execute one step of the simulation"""
        # Get the module for the subfunction
        pass