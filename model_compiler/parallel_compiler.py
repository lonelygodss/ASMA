# Function-wise compiler that handles each function in the GLU-FFN model with specialized methods

from model_compiler.utils import OperationType, TensorId, Function, Model, CompiledModel
from model_compiler.compiler_base import CompilerBase
from typing import List, Dict, Tuple, Optional


class FunctionWiseCompiler(CompilerBase):
    """
    Compiler that handles each function in the GLU-FFN model with specialized methods.
    This gives more granular control over the compilation process for different function types.
    """
    
    def divide_model(self, model: Model) -> CompiledModel:
        """
        Process each function in the model with a specialized method based on function type.
        
        Args:
            model: High-level model description
            
        Returns:
            Compiled model with subfunctions
        """
        compiled_model = CompiledModel()
        
        # Dictionary mapping function name patterns to handler methods
        handlers = {
            # MVM functions in a GLU-FFN architecture
            "gate_proj": self._divide_gate_proj,
            "up_proj": self._divide_up_proj,
            "down_proj": self._divide_down_proj,
            
            # Element-wise operations
            "glu": self._divide_glu,
            "activation": self._divide_activation,
            "trivial_copy": self._divide_trivial_copy,
            "dot_product": self._divide_dot_product,
        }
        
        # Process each function with its specific handler if available
        for function in model.functions:
            # Try to find a specific handler for this function based on name or pattern
            handler = None
            for pattern, method in handlers.items():
                if pattern in function.name.lower():
                    handler = method
                    break
            
            # Fall back to operation type if no name-based handler is found
            if handler is None:
                if function.op_type == OperationType.MVM:
                    handler = self._divide_generic_mvm
                elif function.op_type == OperationType.GLU:
                    handler = self._divide_glu
                elif function.op_type == OperationType.ACTIVATION:
                    handler = self._divide_activation
                elif function.op_type == OperationType.TRIVIAL_COPY:
                    handler = self._divide_trivial_copy
                elif function.op_type == OperationType.DOT_PRODUCT:
                    handler = self._divide_dot_product
                else:
                    # Generic handling for any other function type
                    handler = self._divide_generic_function
            
            # Apply the selected handler
            handler(function, compiled_model)
        
        # Build the dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model
    
    def _divide_gate_proj(self, function: Function, compiled_model: CompiledModel):
        """
        Handle gate projection MVM specifically.
        This produces the gating part of the GLU mechanism.
        
        Args:
            function: Gate projection function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing gate projection: {function.name}")
        if not function.shape:
            raise ValueError(f"Function {function.name} has no shape defined, required for MVM division")
            
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
        dist_output_tensors1 = []
        dist_output_tensors2 = []
        for i in range(v_divisions):
            for j in range(h_divisions):
                compute_i = i + 1
                compute_j = j + 1
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                slice_v = end_v - start_v
                
                # Create tensor ID for distribution output
                dist_output_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                if i == 0:
                    dist_output_tensors1.append((dist_output_id, {'size_h': slice_v, 'size_v': 1}))
                else:
                    dist_output_tensors2.append((dist_output_id, {'size_h': slice_v, 'size_v': 1}))
        
        # Get standard input key from metadata or use default (assuming 'k' for parallel paths)
        k_key = 'k'
        distri_coords = base_coords.copy()
        distri_coords['i'] = 0
        distri_coords['j'] = -1

        distri_func_coords_1 = distri_coords.copy()
        distri_func_coords_2 = distri_coords.copy()
        distri_func_coords_1[k_key] = 1
        distri_func_coords_2[k_key] = 2
        
        # Find distribution function
        distri_fun1 = self.find_subfunction(
            compiled_model,
            coords = distri_func_coords_1,
            op_type = OperationType.DISTRIBUTE
        )

        distri_fun2 = self.find_subfunction(
            compiled_model,
            coords = distri_func_coords_2,
            op_type = OperationType.DISTRIBUTE
        )
        
        # Add new output tensors to distribution function
        for tensor_id, size_params in dist_output_tensors1:
            if distri_fun1:
                self.add_output_to_subfunction(distri_fun1, tensor_id, size_params)
            else:
                raise ValueError(f"Distribution function not found for coordinates {distri_func_coords_1}")
        for tensor_id, size_params in dist_output_tensors2:
            if distri_fun2:
                self.add_output_to_subfunction(distri_fun2, tensor_id, size_params)
            else:
                raise ValueError(f"Distribution function not found for coordinates {distri_func_coords_2}")

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
        
    
    def _divide_up_proj(self, function: Function, compiled_model: CompiledModel):
        """
        Handle up projection MVM specifically.
        This produces the input for the GLU element-wise multiplication.
        
        Args:
            function: Up projection function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing up projection: {function.name}")
        if not function.shape:
            raise ValueError(f"Function {function.name} has no shape defined, required for MVM division")
            
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
        dist_output_tensors1 = []
        dist_output_tensors2 = []
        for i in range(v_divisions):
            for j in range(h_divisions):
                compute_i = i + 1
                compute_j = j + 1
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                slice_v = end_v - start_v
                
                # Create tensor ID for distribution output
                dist_output_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                if i == 0:
                    dist_output_tensors1.append((dist_output_id, {'size_h': slice_v, 'size_v': 1}))
                else:
                    dist_output_tensors2.append((dist_output_id, {'size_h': slice_v, 'size_v': 1}))
        
        # Get standard input key from metadata or use default (assuming 'k' for parallel paths)
        k_key = 'k'
        default_input_k = base_coords.get(k_key, 1)
        
        # Input tensor for distribution (from previous layer/step)
        input_coords = {k: v for k, v in base_coords.items()}
        input_coords[k_key] = default_input_k
        input_tensor_id = self._create_tensor_id(input_coords)

        distri_func_coords_1 = base_coords.copy()
        distri_func_coords_2 = base_coords.copy()
        distri_func_coords_1[k_key] = 1
        distri_func_coords_2[k_key] = 2
        
        # Create distribution function
        self._create_distribution_function(
            distri_func_coords_1,
            input_tensor_id,
            dist_output_tensors1,
            compiled_model
        )

        self._create_distribution_function(
            distri_func_coords_2,
            input_tensor_id,
            dist_output_tensors2,
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
        

    
    def _divide_down_proj(self, function: Function, compiled_model: CompiledModel):
        """
        Handle down projection MVM specifically.
        This processes the result of the GLU operation back to the model dimension.
        Note that for down projection, the array dimensions are swapped.
        
        Args:
            function: Down projection function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing down projection: {function.name}")
        # Use base MVM division logic but with specific metadata for down projection
        if not function.shape:
            raise ValueError(f"Function {function.name} has no shape defined, required for MVM division")
            
        input_dim, output_dim = function.shape
        base_coords = function.coords.copy()
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_v - 1) // self.array_v
        v_divisions = (input_dim + self.array_h - 1) // self.array_h
        
        # Step 1: Create compute subfunctions for each division
        compute_subfuncs = []
        compute_output_tensors = []
        
        for i in range(v_divisions):
            row_compute_subfuncs = []
            row_output_tensors = []
            
            for j in range(h_divisions):
                # Calculate the actual dimensions of this submatrix
                start_h = j * self.array_v
                end_h = min((j + 1) * self.array_v, output_dim)
                start_v = i * self.array_h
                end_v = min((i + 1) * self.array_h, input_dim)
                
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
                glu_coords = base_coords.copy()
                glu_coords['i'] = 1
                glu_coords['j'] = compute_i
                glu_coords['k'] = 1
                glu_coords['m'] = 3
                glu_coords['n'] = 1
                glu_subfunction = self.find_subfunction(
                    compiled_model,
                    coords=glu_coords,
                    op_type=OperationType.GLU,
                )
                input_tensors = glu_subfunction.output_tensors
                for tensor in input_tensors:
                    subfunc.add_input_tensor(tensor_id=tensor.tensor_id, size_h = tensor.size_params)
                                
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
        
        # Step 3: Create addition functions for each column to combine vertical slices
        add_output_tensors = []
        
        for j in range(h_divisions):
            start_h = j * self.array_v
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
        concat_output_tensor_id = self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            add_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
        
    
    def _divide_glu(self, function: Function, compiled_model: CompiledModel):
        """
        Handle GLU operation specifically.
        This combines gate and input projections using element-wise multiplication.
        
        Args:
            function: GLU function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing GLU operation: {function.name}")
        # Use elementwise division logic for GLU
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in horizontal dimension for elementwise ops)
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

            # Find input tensors for this GLU operation
            # Add addition outputs
            add_subfunction_coords = base_coords.copy()
            add_subfunction_coords['i'] = 0
            add_subfunction_coords['j'] = j + 1
            add_subfunction_coords['k'] = 1
            add_subfunction_coords['m'] = 1
            add_subfunction_coords['n'] = 1

            add_subfunction = self.find_subfunction(
                compiled_model,
                coords=add_subfunction_coords,
                op_type=OperationType.ADD,
            )
            input_tensors1 = add_subfunction.output_tensors
            for tensor in input_tensors1:
                subfunc.add_input_tensor(tensor_id=tensor.tensor_id, size_h = tensor.size_params)
            # Add activation outputs
            activation_coords = base_coords.copy()
            activation_coords['i'] = 1
            activation_coords['j'] = j + 1
            activation_coords['k'] = 2
            activation_coords['m'] = 2
            activation_coords['n'] = 1
            
            activation_subfunction = self.find_subfunction(
                compiled_model,
                coords=activation_coords,
                op_type=OperationType.ACTIVATION,
            )
            input_tensors2 = activation_subfunction.output_tensors
            for tensor in input_tensors2:
                subfunc.add_input_tensor(tensor_id=tensor.tensor_id, size_h = tensor.size_params)
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)

    
    def _divide_activation(self, function: Function, compiled_model: CompiledModel):
        """
        Handle activation function specifically.
        
        Args:
            function: Activation function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing activation: {function.name}")
        # Use elementwise division logic for activation
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in horizontal dimension for elementwise ops)
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

            # Find input tensors for this Activation operation
            add_subfunction_coords = base_coords.copy()
            add_subfunction_coords['i'] = 0
            add_subfunction_coords['j'] = j + 1
            add_subfunction_coords['k'] = 2
            add_subfunction_coords['m'] = 1
            add_subfunction_coords['n'] = 1

            add_subfunction = self.find_subfunction(
                compiled_model,
                coords=add_subfunction_coords,
                op_type=OperationType.ADD,
            )
            input_tensors1 = add_subfunction.output_tensors
            for tensor in input_tensors1:
                subfunc.add_input_tensor(tensor_id=tensor.tensor_id, size_h = tensor.size_params)
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)    

    def _divide_trivial_copy(self, function: Function, compiled_model: CompiledModel):
        """
        Handle trivial copy specifically.
        
        Args:
            function: Trivial copy function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing trivial copy: {function.name}")
        # Use elementwise division logic for trivial copy
        self._divide_generic_elementwise(function, compiled_model)
    
    def _divide_dot_product(self, function: Function, compiled_model: CompiledModel):
        """
        Handle dot product specifically.
        
        Args:
            function: Dot product function
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing dot product: {function.name}")
        # Use elementwise division logic for dot product
        self._divide_generic_elementwise(function, compiled_model)
    
    def _divide_generic_function(self, function: Function, compiled_model: CompiledModel):
        """
        Generic handler for functions without a specialized handler.
        
        Args:
            function: Function to process
            compiled_model: Compiled model to add subfunctions to
        """
        print(f"Processing generic function: {function.name} with op type: {function.op_type}")
        if function.op_type == OperationType.MVM:
            self._divide_generic_mvm(function, compiled_model)
        else:
            self._divide_generic_elementwise(function, compiled_model)
    
    def _divide_generic_mvm(self, function: Function, compiled_model: CompiledModel):
        """
        Generic implementation for dividing MVM functions.
        
        Args:
            function: MVM function to process
            compiled_model: Compiled model to add subfunctions to
        """
        if not function.shape:
            raise ValueError(f"Function {function.name} has no shape defined, required for MVM division")
            
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
        
        # Get standard input key from metadata or use default (assuming 'k' for parallel paths)
        k_key = 'k'
        default_input_k = base_coords.get(k_key, 1)
        
        # Input tensor for distribution (from previous layer/step)
        input_coords = {k: v for k, v in base_coords.items()}
        input_coords[k_key] = default_input_k
        input_tensor_id = self._create_tensor_id(input_coords)
        
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
        concat_output_tensor_id = self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            add_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
        
        # Step 5: Add pass function to next step
        self._add_pass_function(function, compiled_model, output_dim)
    
    def _divide_generic_elementwise(self, function: Function, compiled_model: CompiledModel, is_glu: bool = False):
        """
        Generic implementation for dividing element-wise operations.
        
        Args:
            function: Element-wise function to process
            compiled_model: Compiled model to add subfunctions to
            is_glu: Whether this is a GLU operation (needing two inputs)
        """
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in horizontal dimension for elementwise ops)
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
        
        # Step 2: Handle different distribution needs for GLU vs. other operations
        if is_glu or function.op_type == OperationType.GLU:
            # GLU needs two distribution functions (one for each input)
            for k_idx in [1, 2]:  # GLU has two inputs
                k_key = 'k'
                dist_coords = base_coords.copy()
                dist_coords[k_key] = k_idx
                
                # Input tensor for this distribution function
                input_tensor_id = self._create_tensor_id(dist_coords)
                
                # Prepare output tensors for distribution
                for j in range(h_divisions):
                    # Calculate slice dimensions
                    start_h = j * self.array_h
                    end_h = min((j + 1) * self.array_h, output_dim)
                    slice_size = end_h - start_h
                    
                    # Create tensor ID for distribution output
                    output_coords = base_coords.copy()
                    output_coords[k_key] = k_idx
                    output_coords['i'] = element_idx
                    output_coords['j'] = -(j+1)


                
        else:
            # Single distribution function for non-GLU operations
            # Input tensor for distribution
            input_tensor_id = self._create_tensor_id(base_coords)
            
