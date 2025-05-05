# Base compiler class with essential methods for subfunction creation

from model_compiler.utils import OperationType, TensorId, TensorWithSize, Function, SubFunction, Model, CompiledModel
from typing import List, Dict, Tuple, Optional, Any, Set


class CompilerBase:
    """Base compiler class that contains essential methods for dividing models"""
    
    def __init__(self, array_h: int, array_v: int):
        """
        Initialize compiler with array dimensions
        
        Args:
            array_h: Horizontal size of CIM array
            array_v: Vertical size of CIM array
        """
        self.array_h = array_h
        self.array_v = array_v

    
    def divide_model(self, model: Model) -> CompiledModel:
        """
        Divide the model according to hardware constraints
        
        Args:
            model: High-level model description
            
        Returns:
            Compiled model with subfunctions
        """
        compiled_model = CompiledModel()
        
        # This is an abstract method that should be implemented by child classes
        # Process the functions in model and add subfunctions to compiled_model
        
        # Build the dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model
    
    def _create_tensor_id(self, base_coords, **override_coords):
        """Helper to create a tensor ID with consistent naming"""
        coords = base_coords.copy()
        coords.update(override_coords)
        return TensorId(**coords)
    
    def _create_subfunction(self, base_coords, op_type, **override_coords):
        """Helper to create a subfunction with consistent naming"""
        coords = base_coords.copy()
        coords.update(override_coords)
        return SubFunction(op_type=op_type, **coords)
    
    def _add_pass_function(self, function, compiled_model, output_size, output_tensor_id=None):
        """Helper to create a pass function to the next step"""
        base_coords = function.coords.copy()
        
        # Create pass function
        pass_func = self._create_subfunction(base_coords, OperationType.PASS, i=-1, j=0)
        
        # Set up input tensor - from concat result
        input_tensor_id = self._create_tensor_id(base_coords, i=0, j=0)            
        pass_func.add_input_tensor(input_tensor_id, size_h=output_size, size_v=1)
        
        # Set up output tensor - to next step
        if output_tensor_id is None:
            # Default to next step in sequence
            m_key = 'm'  # Assuming 'm' is used for sequence step
            next_m = base_coords.get(m_key, 0) + 1
            output_tensor_id = self._create_tensor_id({k: v for k, v in base_coords.items() if k != m_key}, **{m_key: next_m})
            
        pass_func.add_output_tensor(output_tensor_id, size_h=output_size, size_v=1)
        compiled_model.add_subfunction(pass_func)

    def _create_distribution_function(self, base_coords: Dict, input_tensor_id: TensorId, 
                                     output_tensor_ids: List[Tuple[TensorId, Dict]], 
                                     compiled_model: CompiledModel) -> SubFunction:
        """
        Create a distribution function to distribute input to multiple destinations
        
        Args:
            base_coords: Base coordinates for this function context
            input_tensor_id: Input tensor ID for distribution
            output_tensor_ids: List of (tensor_id, size_params) tuples for outputs
            compiled_model: Compiled model to add the function to
            
        Returns:
            Created distribution function
        """
        # Create distribution function
        distribution_func = self._create_subfunction(
            base_coords, 
            OperationType.DISTRIBUTE, 
            i=0, 
            j=-1
        )
        
        # Add input tensor
        input_size_h = sum(size_params.get('size_h', 0) for _, size_params in output_tensor_ids)
        distribution_func.add_input_tensor(input_tensor_id, size_h=input_size_h, size_v=1)
        
        # Add output tensors
        for tensor_id, size_params in output_tensor_ids:
            distribution_func.add_output_tensor(tensor_id, **size_params)
        
        # Add to compiled model
        compiled_model.add_subfunction(distribution_func)
        
        return distribution_func

    def _create_concat_function(self, base_coords: Dict, input_tensor_ids: List[Tuple[TensorId, Dict]],
                               output_tensor_id: TensorId, output_size: Dict, 
                               compiled_model: CompiledModel) -> SubFunction:
        """
        Create a concatenation function to collect multiple inputs
        
        Args:
            base_coords: Base coordinates for this function context
            input_tensor_ids: List of (tensor_id, size_params) tuples for inputs
            output_tensor_id: Output tensor ID for concatenated result
            output_size: Size parameters for output tensor
            compiled_model: Compiled model to add the function to
            
        Returns:
            Created concat function
        """
        # Create concat function
        concat_func = self._create_subfunction(base_coords, OperationType.CONCAT, i=0, j=0)
        
        # Add input tensors
        for tensor_id, size_params in input_tensor_ids:
            concat_func.add_input_tensor(tensor_id, **size_params)
        
        # Add output tensor
        concat_func.add_output_tensor(output_tensor_id, **output_size)
        
        # Add to compiled model
        compiled_model.add_subfunction(concat_func)
        
        return concat_func

    def _create_add_function(self, base_coords: Dict, input_tensor_ids: List[Tuple[TensorId, Dict]],
                            output_tensor_id: TensorId, output_size: Dict, 
                            compiled_model: CompiledModel, add_idx: List) -> SubFunction:
        """
        Create an addition function to combine multiple inputs
        
        Args:
            base_coords: Base coordinates for this function context
            input_tensor_ids: List of (tensor_id, size_params) tuples for inputs
            output_tensor_id: Output tensor ID for addition result
            output_size: Size parameters for output tensor
            compiled_model: Compiled model to add the function to
            add_idx: Index for this add function (for positioning)
            
        Returns:
            Created add function
        """
        # Create add function
        add_func = self._create_subfunction(base_coords, OperationType.ADD, i=add_idx[0], j=add_idx[1])
        
        # Set shape if size_h and size_v are available
        if 'size_h' in output_size and 'size_v' in output_size:
            add_func.set_shape((output_size['size_v'], output_size['size_h']))
        
        # Add input tensors
        for tensor_id, size_params in input_tensor_ids:
            add_func.add_input_tensor(tensor_id, **size_params)
        
        # Add output tensor
        add_func.add_output_tensor(output_tensor_id, **output_size)
        
        # Add to compiled model
        compiled_model.add_subfunction(add_func)
        
        return add_func
    
    def _create_add_concat_function(self, base_coords: Dict, input_tensor_ids: List[Tuple[TensorId, Dict]],
                            output_tensor_id: TensorId, output_size: Dict, 
                            compiled_model: CompiledModel, add_idx: List) -> SubFunction:
        """
        Create an addition function to combine multiple inputs
        
        Args:
            base_coords: Base coordinates for this function context
            input_tensor_ids: List of (tensor_id, size_params) tuples for inputs
            output_tensor_id: Output tensor ID for addition result
            output_size: Size parameters for output tensor
            compiled_model: Compiled model to add the function to
            add_idx: Index for this add function (for positioning)
            
        Returns:
            Created add function
        """
        # Create add function
        add_func = self._create_subfunction(base_coords, OperationType.CONCAT, i=add_idx[0], j=add_idx[1])
        
        # Set shape if size_h and size_v are available
        if 'size_h' in output_size and 'size_v' in output_size:
            add_func.set_shape((output_size['size_v'], output_size['size_h']))
        
        # Add input tensors
        for tensor_id, size_params in input_tensor_ids:
            add_func.add_input_tensor(tensor_id, **size_params)
        
        # Add output tensor
        add_func.add_output_tensor(output_tensor_id, **output_size)
        
        # Add to compiled model
        compiled_model.add_subfunction(add_func)
        
        return add_func
    def find_subfunction(self, compiled_model: CompiledModel, **criteria) -> Optional[SubFunction]:
        """
        Find a specific subfunction in the compiled model based on search criteria.
        
        Args:
            compiled_model: The compiled model to search in
            criteria: Key-value pairs for filter conditions (e.g., i=1, j=2, op_type=OperationType.MVM)
            
        Returns:
            Matching subfunction or None if not found
        """
        for subfunc in compiled_model.subfunctions:
            match = True
            for key, value in criteria.items():
                if key == 'op_type':
                    if subfunc.op_type != value:
                        match = False
                        break
                elif hasattr(subfunc, key):
                    if getattr(subfunc, key) != value:
                        match = False
                        break
                elif key in subfunc.coords:
                    if subfunc.coords[key] != value:
                        match = False
                        break
                else:
                    # If attribute doesn't exist, it's not a match
                    match = False
                    break
            
            if match:
                return subfunc
        
        return None
    
    def find_subfunctions(self, compiled_model: CompiledModel, **criteria) -> List[SubFunction]:
        """
        Find all subfunctions in the compiled model that match given criteria.
        
        Args:
            compiled_model: The compiled model to search in
            criteria: Key-value pairs for filter conditions (e.g., i=1, j=2, op_type=OperationType.MVM)
            
        Returns:
            List of matching subfunctions
        """
        results = []
        
        for subfunc in compiled_model.subfunctions:
            match = True
            for key, value in criteria.items():
                if key == 'op_type':
                    if subfunc.op_type != value:
                        match = False
                        break
                elif hasattr(subfunc, key):
                    if getattr(subfunc, key) != value:
                        match = False
                        break
                elif key in subfunc.coords:
                    if subfunc.coords[key] != value:
                        match = False
                        break
                else:
                    # If attribute doesn't exist, it's not a match
                    match = False
                    break
            
            if match:
                results.append(subfunc)
        
        return results
    
    def add_input_to_subfunction(self, subfunction: SubFunction, 
                                tensor_id: TensorId, 
                                size_h: int, 
                                size_v: int = 1) -> None:
        """
        Add an input tensor to an existing subfunction
        
        Args:
            subfunction: The subfunction to add the input tensor to
            tensor_id: ID of the input tensor
            size_h: Horizontal size of the tensor
            size_v: Vertical size of the tensor (default 1)
        """
        subfunction.add_input_tensor(tensor_id, size_h=size_h, size_v=size_v)
    
    def add_output_to_subfunction(self, subfunction: SubFunction, 
                                 tensor_id: TensorId, 
                                 size_h: int, 
                                 size_v: int = 1) -> None:
        """
        Add an output tensor to an existing subfunction
        
        Args:
            subfunction: The subfunction to add the output tensor to
            tensor_id: ID of the output tensor
            size_h: Horizontal size of the tensor
            size_v: Vertical size of the tensor (default 1)
        """
        subfunction.add_output_tensor(tensor_id, size_h=size_h, size_v=size_v)
    
    def connect_subfunctions(self, 
                           source_subfunction: SubFunction, 
                           target_subfunction: SubFunction, 
                           tensor_size_h: int,
                           tensor_size_v: int = 1) -> TensorId:
        """
        Connect two subfunctions by creating a tensor that serves as output
        for the source and input for the target.
        
        Args:
            source_subfunction: Source subfunction
            target_subfunction: Target subfunction 
            tensor_size_h: Horizontal size of connecting tensor
            tensor_size_v: Vertical size of connecting tensor
            
        Returns:
            The created tensor ID
        """
        # Create a tensor ID that combines coordinates from both subfunctions
        base_coords = source_subfunction.coords.copy()
        
        # Use source i and j values but increment to make the tensor ID unique
        i_val = base_coords.get('i', 0)
        j_val = base_coords.get('j', 0)
        
        # Create a unique tensor ID
        tensor_id = self._create_tensor_id(base_coords, i=i_val*10, j=j_val*10)
        
        # Add as output to source
        self.add_output_to_subfunction(source_subfunction, tensor_id, tensor_size_h, tensor_size_v)
        
        # Add as input to target
        self.add_input_to_subfunction(target_subfunction, tensor_id, tensor_size_h, tensor_size_v)
        
        return tensor_id
            
    def _divide_mvm(self, function: Function, compiled_model: CompiledModel):
        """
        Base method for dividing an MVM function
        Should be implemented by child classes
        """
        raise NotImplementedError("_divide_mvm must be implemented by child classes")

    def _divide_elementwise(self, function: Function, compiled_model: CompiledModel):
        """
        Base method for dividing elementwise operations
        Should be implemented by child classes
        """
        raise NotImplementedError("_divide_elementwise must be implemented by child classes")
