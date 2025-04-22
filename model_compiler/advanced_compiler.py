from model_compiler.basic_compiler import Compiler
from model_compiler.utils import OperationType, TensorId, TensorWithSize, Function, SubFunction, Model, CompiledModel
from typing import List, Dict, Tuple, Optional, Any, Set, Union


class AdvancedCompiler(Compiler):
    """
    Advanced compiler with fine-grained control over GLU-FFN dataflow.
    Allows for compiling functions one by one with customizable interconnections.
    """
    
    def __init__(self, array_h: int, array_v: int):
        """
        Initialize compiler with array dimensions
        
        Args:
            array_h: Horizontal size of CIM array
            array_v: Vertical size of CIM array
        """
        super().__init__(array_h, array_v)
        self.current_compiled_model = None
        
    def compile_glu_ffn(self, model: Model) -> CompiledModel:
        """
        Compile a GLU-FFN model with fine-grained control over dataflow
        
        Args:
            model: High-level model description for GLU-FFN
            
        Returns:
            Compiled model with subfunctions
        """
        # Initialize compiled model
        compiled_model = CompiledModel()
        self.current_compiled_model = compiled_model
        
        # Extract dimensions from model metadata or first function
        hidden_dim, ffn_dim = self._extract_model_dimensions(model)
        
        # Find functions by their operation type and coordinates
        up_proj = None
        gate_proj = None
        up_copy = None
        activation = None
        glu = None
        down_proj = None
        
        for func in model.functions:
            # Identify functions by their k, m coordinates
            k = func.coords.get('k', 0)
            m = func.coords.get('m', 0)
            
            if func.op_type == OperationType.MVM:
                if k == 1 and m == 1:
                    up_proj = func
                elif k == 2 and m == 1:
                    gate_proj = func
                elif k == 1 and m == 4:
                    down_proj = func
            elif func.op_type == OperationType.TRIVIAL_COPY:
                if k == 1 and m == 2:
                    up_copy = func
            elif func.op_type == OperationType.ACTIVATION:
                if k == 2 and m == 2:
                    activation = func
            elif func.op_type == OperationType.GLU:
                if k == 1 and m == 3:
                    glu = func
        
        # Compile functions one by one with explicit dataflow
        
        # 1. Up and Gate projections (can be compiled in parallel)
        if up_proj:
            self.compile_mvm_function(up_proj, compiled_model)
        if gate_proj:
            self.compile_mvm_function(gate_proj, compiled_model)
        
        # 2. Up copy and Gate activation
        if up_copy:
            self.compile_elementwise_function(up_copy, compiled_model)
        if activation:
            self.compile_elementwise_function(activation, compiled_model)
        
        # 3. GLU operation
        if glu:
            self.compile_glu_function(glu, compiled_model)
        
        # 4. Down projection
        if down_proj:
            self.compile_mvm_function(down_proj, compiled_model)
        
        # Build dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model
    
    def _extract_model_dimensions(self, model: Model) -> Tuple[int, int]:
        """Extract hidden_dim and ffn_dim from model"""
        # Try to get dimensions from model metadata
        hidden_dim = model.get_metadata('hidden_dim', None)
        ffn_dim = model.get_metadata('ffn_dim', None)
        
        # If not found, try to infer from function shapes
        if hidden_dim is None or ffn_dim is None:
            for func in model.functions:
                if func.op_type == OperationType.MVM and func.shape:
                    if func.coords.get('m') == 1:  # Up projection
                        hidden_dim = func.shape[0]
                        ffn_dim = func.shape[1]
                    elif func.coords.get('m') == 4:  # Down projection
                        ffn_dim = func.shape[0]
                        hidden_dim = func.shape[1]
        
        # Default values if still not found
        if hidden_dim is None:
            hidden_dim = 1024  # Default value
        if ffn_dim is None:
            ffn_dim = 4096  # Default value
        
        return hidden_dim, ffn_dim
    
    def compile_mvm_function(self, function: Function, compiled_model: CompiledModel, 
                             custom_input_id: TensorId = None, custom_output_id: TensorId = None):
        """
        Compile an MVM function with customizable input and output tensor IDs
        
        Args:
            function: MVM function to compile
            compiled_model: Target compiled model
            custom_input_id: Custom input tensor ID (optional)
            custom_output_id: Custom output tensor ID (optional)
        """
        if not function.shape:
            raise ValueError(f"Function {function} has no shape defined, required for MVM division")
            
        input_dim, output_dim = function.shape
        base_coords = function.coords.copy()
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_h - 1) // self.array_h
        v_divisions = (input_dim + self.array_v - 1) // self.array_v
        
        # Step 1: Create compute subfunctions for each division
        compute_subfuncs = []
        compute_output_tensors = []
        
        for i in range(v_divisions):
            row_compute_subfuncs = []
            row_output_tensors = []
            
            for j in range(h_divisions):
                # Calculate the actual dimensions of this submatrix
                start_h = j * self.array_h
                end_h = min((j + 1) * self.array_h, output_dim)
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                
                slice_h = end_h - start_h
                slice_v = end_v - start_v
                
                # Create subfunction
                compute_i = i + 1  # Base index for MVM computation
                compute_j = j + 1  # Base index for MVM computation
                subfunc = self._create_subfunction(
                    base_coords, 
                    OperationType.MVM, 
                    i=compute_i, 
                    j=compute_j
                )
                subfunc.set_shape((slice_v, slice_h))
                subfunc.set_parent(function)
                
                # Input tensor ID for this slice
                input_tensor_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                subfunc.add_input_tensor(input_tensor_id, size_h=slice_v, size_v=1)
                
                # Output tensor ID for compute result
                output_tensor_id = self._create_tensor_id(base_coords, i=compute_i, j=compute_j)
                subfunc.add_output_tensor(output_tensor_id, size_h=slice_h, size_v=1)
                
                # Store for later reference
                row_compute_subfuncs.append(subfunc)
                row_output_tensors.append((output_tensor_id, {'size_h': slice_h, 'size_v': 1}))
                
                # Add to compiled model
                compiled_model.add_subfunction(subfunc)
            
            compute_subfuncs.append(row_compute_subfuncs)
            compute_output_tensors.append(row_output_tensors)
        
        # Step 2: Create distribution function and connect to compute inputs
        
        # Prepare output tensors for distribution
        dist_output_tensors = []
        for i in range(v_divisions):
            for j in range(h_divisions):
                compute_i = i + 1
                compute_j = j + 1
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                slice_v = end_v - start_v
                
                # Create tensor ID for distribution output
                dist_output_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                dist_output_tensors.append((dist_output_id, {'size_h': slice_v, 'size_v': 1}))
        
        # Input tensor for distribution (use custom if provided)
        input_tensor_id = custom_input_id if custom_input_id else self._create_default_input_id(base_coords)
        
        # Create distribution function
        self._create_distribution_function(
            base_coords,
            input_tensor_id,
            dist_output_tensors,
            compiled_model
        )
        
        # Step 3: Create addition functions for each column to combine vertical slices
        add_output_tensors = []
        
        for j in range(h_divisions):
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_h = end_h - start_h
            add_j = j + 1  # Base index for addition operations
            
            # Collect input tensors for this column's addition
            add_input_tensors = []
            for i in range(v_divisions):
                compute_i = i + 1
                compute_j = j + 1
                output_tensor_id = self._create_tensor_id(base_coords, i=compute_i, j=compute_j)
                add_input_tensors.append((output_tensor_id, {'size_h': slice_h, 'size_v': 1}))
            
            # Create output tensor ID for addition
            add_output_tensor_id = self._create_tensor_id(base_coords, i=0, j=add_j)
            output_size = {'size_h': slice_h, 'size_v': 1}
            
            # Create addition function
            self._create_add_function(
                base_coords,
                add_input_tensors,
                add_output_tensor_id,
                output_size,
                compiled_model,
                add_j
            )
            
            # Store for concat
            add_output_tensors.append((add_output_tensor_id, output_size))
        
        # Step 4: Create concatenation function to combine horizontal slices
        concat_output_tensor_id = custom_output_id if custom_output_id else self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            add_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
    
    def compile_elementwise_function(self, function: Function, compiled_model: CompiledModel, 
                                    custom_input_id: TensorId = None, custom_output_id: TensorId = None):
        """
        Compile an elementwise function (activation, trivial copy) with customizable I/O
        
        Args:
            function: Elementwise function to compile
            compiled_model: Target compiled model
            custom_input_id: Custom input tensor ID (optional)
            custom_output_id: Custom output tensor ID (optional)
        """
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 4096  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_h - 1) // self.array_h

        # Step 1: Create compute subfunctions for each division
        compute_subfuncs = []
        compute_output_tensors = []
        element_idx = 1  # Index for elementwise operations
        
        for j in range(h_divisions):
            # Calculate slice dimensions
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_size = end_h - start_h
            
            # Create compute subfunction
            subfunc = self._create_subfunction(
                base_coords,
                function.op_type,
                i=element_idx,
                j=j+1
            )
            subfunc.set_shape((1, slice_size))
            subfunc.set_parent(function)
            
            # Set output tensor for compute function
            output_tensor_id = self._create_tensor_id(base_coords, i=element_idx, j=j+1)
            subfunc.add_output_tensor(output_tensor_id, size_h=slice_size, size_v=1)
            
            compute_subfuncs.append(subfunc)
            compute_output_tensors.append((output_tensor_id, {'size_h': slice_size, 'size_v': 1}))
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)
        
        # Step 2: Create distribution function
        # Input tensor for distribution (use custom if provided)
        input_tensor_id = custom_input_id if custom_input_id else self._create_default_input_id(base_coords)
        
        # Prepare output tensors for distribution
        dist_output_tensors = []
        for j in range(h_divisions):
            # Calculate slice dimensions
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_size = end_h - start_h
            
            # Create tensor ID for distribution output
            dist_output_id = self._create_tensor_id(base_coords, i=element_idx, j=-(j+1))
            dist_output_tensors.append((dist_output_id, {'size_h': slice_size, 'size_v': 1}))
            
            # Add input to corresponding compute function
            compute_subfuncs[j].add_input_tensor(dist_output_id, size_h=slice_size, size_v=1)
        
        # Create distribution function
        self._create_distribution_function(
            base_coords,
            input_tensor_id,
            dist_output_tensors,
            compiled_model
        )
        
        # Step 3: Create concatenation function
        concat_output_tensor_id = custom_output_id if custom_output_id else self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            compute_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
    
    def compile_glu_function(self, function: Function, compiled_model: CompiledModel,
                            custom_input1_id: TensorId = None, custom_input2_id: TensorId = None,
                            custom_output_id: TensorId = None):
        """
        Compile a GLU function with customizable inputs and output
        
        Args:
            function: GLU function to compile
            compiled_model: Target compiled model
            custom_input1_id: Custom first input tensor ID (optional)
            custom_input2_id: Custom second input tensor ID (optional)
            custom_output_id: Custom output tensor ID (optional)
        """
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer or use default
            output_dim = 4096  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_h - 1) // self.array_h

        # Step 1: Create compute subfunctions for each division
        compute_subfuncs = []
        compute_output_tensors = []
        element_idx = 1  # Index for elementwise operations
        
        for j in range(h_divisions):
            # Calculate slice dimensions
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_size = end_h - start_h
            
            # Create compute subfunction
            subfunc = self._create_subfunction(
                base_coords,
                function.op_type,
                i=element_idx,
                j=j+1
            )
            subfunc.set_shape((1, slice_size))
            subfunc.set_parent(function)
            
            # Set output tensor for compute function
            output_tensor_id = self._create_tensor_id(base_coords, i=element_idx, j=j+1)
            subfunc.add_output_tensor(output_tensor_id, size_h=slice_size, size_v=1)
            
            compute_subfuncs.append(subfunc)
            compute_output_tensors.append((output_tensor_id, {'size_h': slice_size, 'size_v': 1}))
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)
        
        # Step 2: Create distribution functions for both inputs
        # For each of the two inputs to GLU
        for k_idx, custom_input in [(1, custom_input1_id), (2, custom_input2_id)]:
            # Set up coordinates for this input stream
            dist_coords = base_coords.copy()
            dist_coords['k'] = k_idx
            
            # Input tensor for distribution (use custom if provided)
            input_tensor_id = custom_input if custom_input else self._create_tensor_id(dist_coords)
            
            # Prepare output tensors for distribution
            dist_output_tensors = []
            for j in range(h_divisions):
                # Calculate slice dimensions
                start_h = j * self.array_h
                end_h = min((j + 1) * self.array_h, output_dim)
                slice_size = end_h - start_h
                
                # Create tensor ID for distribution output
                output_coords = base_coords.copy()
                output_coords['k'] = k_idx
                output_coords['i'] = element_idx
                output_coords['j'] = -(j+1)
                
                dist_output_id = self._create_tensor_id(output_coords)
                dist_output_tensors.append((dist_output_id, {'size_h': slice_size, 'size_v': 1}))
                
                # Add input to corresponding compute function
                compute_subfuncs[j].add_input_tensor(dist_output_id, size_h=slice_size, size_v=1)
            
            # Create distribution function
            self._create_distribution_function(
                dist_coords,
                input_tensor_id,
                dist_output_tensors,
                compiled_model
            )
        
        # Step 3: Create concatenation function
        concat_output_tensor_id = custom_output_id if custom_output_id else self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            compute_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
    
    def _create_default_input_id(self, base_coords: Dict) -> TensorId:
        """Create a default input tensor ID based on base coordinates"""
        # For most functions, the default input tensor is from the previous step
        m_key = 'm'  # Assuming 'm' is the sequence step
        prev_m = base_coords.get(m_key, 0) - 1
        if prev_m < 0:
            # If at the beginning of the sequence, use previous layer
            n_key = 'n'  # Layer coordinate
            prev_n = base_coords.get(n_key, 0) - 1
            input_coords = {k: v for k, v in base_coords.items() if k != m_key and k != n_key}
            input_coords[m_key] = 0
            input_coords[n_key] = prev_n
        else:
            input_coords = {k: v for k, v in base_coords.items() if k != m_key}
            input_coords[m_key] = prev_m
        
        return TensorId(**input_coords)
    
    def create_custom_dataflow(self, model: Model, dataflow_config: Dict) -> CompiledModel:
        """
        Create a custom dataflow pattern for a GLU-FFN model
        
        Args:
            model: High-level model description
            dataflow_config: Configuration for custom dataflow
            
        Returns:
            Compiled model with custom dataflow pattern
        """
        compiled_model = CompiledModel()
        self.current_compiled_model = compiled_model
        
        # Process functions according to custom dataflow configuration
        for function_config in dataflow_config.get('functions', []):
            function_id = function_config.get('id')
            custom_inputs = function_config.get('inputs', {})
            custom_output = function_config.get('output')
            
            # Find the function in the model
            target_function = None
            for func in model.functions:
                if self._match_function_id(func, function_id):
                    target_function = func
                    break
            
            if target_function:
                self._compile_function_with_custom_io(
                    target_function, 
                    compiled_model,
                    custom_inputs, 
                    custom_output
                )
        
        # Build dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model
    
    def _match_function_id(self, function: Function, function_id: Dict) -> bool:
        """Check if a function matches the given identifier"""
        for key, value in function_id.items():
            if key in function.coords and function.coords[key] != value:
                return False
            if key == 'op_type' and function.op_type != value:
                return False
        return True
    
    def _compile_function_with_custom_io(self, function: Function, compiled_model: CompiledModel,
                                         custom_inputs: Dict, custom_output: TensorId = None):
        """Compile a function with custom inputs and output"""
        if function.op_type == OperationType.MVM:
            self.compile_mvm_function(function, compiled_model, 
                                    custom_input_id=custom_inputs.get(0),
                                    custom_output_id=custom_output)
        elif function.op_type == OperationType.GLU:
            self.compile_glu_function(function, compiled_model,
                                    custom_input1_id=custom_inputs.get(1),
                                    custom_input2_id=custom_inputs.get(2),
                                    custom_output_id=custom_output)
        else:
            self.compile_elementwise_function(function, compiled_model,
                                           custom_input_id=custom_inputs.get(0),
                                           custom_output_id=custom_output)
