# Implementation of specific compiler strategy for GLU-FFN using the base compiler

from model_compiler.utils import OperationType, TensorId, Function, Model, CompiledModel
from model_compiler.compiler_base import CompilerBase
from typing import List, Dict, Tuple


class Compiler(CompilerBase):
    """Specific compiler implementation that divides the model according to hardware constraints"""
    
    def divide_model(self, model: Model) -> CompiledModel:
        """
        Divide the model according to hardware constraints
        
        Args:
            model: High-level model description
            
        Returns:
            Compiled model with subfunctions
        """
        compiled_model = CompiledModel()
        
        # Process each function in the model
        for function in model.functions:
            if function.op_type == OperationType.MVM:
                self._divide_mvm(function, compiled_model)
            elif function.op_type in [OperationType.ACTIVATION, OperationType.TRIVIAL_COPY, 
                                     OperationType.DOT_PRODUCT, OperationType.GLU]:
                self._divide_elementwise(function, compiled_model)
        
        # Build the dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model

    def _divide_mvm(self, function: Function, compiled_model: CompiledModel):
        """Divide an MVM function into subfunctions based on array size constraints"""
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

    def _divide_elementwise(self, function: Function, compiled_model: CompiledModel):
        """Divide element-wise operations (activation, GLU, etc.) into subfunctions"""
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
        if function.op_type == OperationType.GLU:
            # GLU needs two distribution functions (one for each input)
            for k_idx in [1, 2]:  # GLU has two inputs
                k_key = 'k'
                dist_coords = base_coords.copy()
                dist_coords[k_key] = k_idx
                
                # Input tensor for this distribution function
                input_tensor_id = self._create_tensor_id(dist_coords)
                
                # Prepare output tensors for distribution
                dist_output_tensors = []
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
        else:
            # Single distribution function for non-GLU operations
            # Input tensor for distribution
            input_tensor_id = self._create_tensor_id(base_coords)
            
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
        concat_output_tensor_id = self._create_tensor_id(base_coords, i=0, j=0)
        concat_output_size = {'size_h': output_dim, 'size_v': 1}
        
        self._create_concat_function(
            base_coords,
            compute_output_tensors,
            concat_output_tensor_id,
            concat_output_size,
            compiled_model
        )
        
        # Step 4: Add pass function to next step
        self._add_pass_function(function, compiled_model, output_dim)

