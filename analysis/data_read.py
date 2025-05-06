import os
import glob
import pandas as pd
import numpy as np

class SimulationDataLoader:
    """
    A class to load and process simulation data from CSV files generated
    by the simulation framework.
    """
    
    def __init__(self, data_path=None, filename=None, dimension_parameter=None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the directory containing CSV files
        filename : str, optional
            Specific CSV filename to load. If None, the most recent file will be used.
        """
        self.data = None
        self.dimension_para = dimension_parameter
        
        if data_path or filename:
            self.load_data(data_path, filename)
    
    def load_data(self, data_path='.', filename=None):
        """
        Load simulation data from CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing CSV files
        filename : str, optional
            Specific CSV filename to load. If None, the most recent file will be used.
        
        Returns:
        --------
        pd.DataFrame
            The loaded data
        """
        if filename:
            file_path = os.path.join(data_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No file found at {file_path}")
        else:
            # Find the most recent simulation results file
            csv_files = glob.glob(os.path.join(data_path, "simulation_results_*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No simulation results files found in {data_path}")
            
            # Sort by modification time (most recent first)
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            file_path = csv_files[0]
            print(f"Loading most recent file: {os.path.basename(file_path)}")
        
        # Load the data
        self.data = pd.read_csv(file_path)
        
        # Convert columns to appropriate types
        if self.data is not None:
            self.data['Compiler'] = self.data['Compiler'].astype(str)
            self.data['Array Size'] = self.data['Array Size'].astype(int)
            self.data[self.dimension_para] = self.data[self.dimension_para].astype(int)
            self.data['Time'] = self.data['Time'].astype(float)
            self.data['Energy'] = self.data['Energy'].astype(float)
        
        return self.data
    
    def get_compiler_names(self):
        """Get unique compiler names in the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.data['Compiler'].unique()
    
    def get_array_sizes(self):
        """Get unique array sizes in the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return sorted(self.data['Array Size'].unique())
    
    def get_scale_dimensions(self):
        """Get unique scale dimensions in the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return sorted(self.data[self.dimension_para].unique())
    
    def filter_data(self, compiler=None, array_size=None, scale_dim=None):
        """
        Filter the data based on specified criteria.
        
        Parameters:
        -----------
        compiler : str or list, optional
            Compiler name(s) to filter by
        array_size : int or list, optional
            Array size(s) to filter by
        scale_dim : int or list, optional
            scale dimension(s) to filter by
        
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        filtered_data = self.data.copy()
        
        if compiler:
            if isinstance(compiler, list):
                filtered_data = filtered_data[filtered_data['Compiler'].isin(compiler)]
            else:
                filtered_data = filtered_data[filtered_data['Compiler'] == compiler]
        
        if array_size:
            if isinstance(array_size, list):
                filtered_data = filtered_data[filtered_data['Array Size'].isin(array_size)]
            else:
                filtered_data = filtered_data[filtered_data['Array Size'] == array_size]
        
        if scale_dim:
            if isinstance(scale_dim, list):
                filtered_data = filtered_data[filtered_data[self.dimension_para].isin(scale_dim)]
            else:
                filtered_data = filtered_data[filtered_data[self.dimension_para] == scale_dim]
        
        return filtered_data
    
    def prepare_data_for_line_plot(self, x_axis=None, y_axis='Time', fixed_param=None, fixed_value=None):
        """
        Prepare data for line plotting with one parameter varied on x-axis.
        
        Parameters:
        -----------
        x_axis : str
            Column to use for x-axis (self.dimension_para or 'Array Size')
        y_axis : str
            Column to use for y-axis ('Time' or 'Energy')
        fixed_param : str, optional
            Parameter to keep fixed ('Array Size' or self.dimension_para)
        fixed_value : int, optional
            Value to keep the fixed parameter at
        
        Returns:
        --------
        dict
            Dictionary with compiler names as keys and DataFrames as values,
            ready for plotting
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if x_axis not in [self.dimension_para, 'Array Size']:
            raise ValueError("x_axis must be self.dimension_para or 'Array Size'")
        
        if y_axis not in ['Time', 'Energy']:
            raise ValueError("y_axis must be 'Time' or 'Energy'")
        
        filtered_data = self.data.copy()
        
        # Apply filter if a parameter is fixed
        if fixed_param and fixed_value is not None:
            filtered_data = filtered_data[filtered_data[fixed_param] == fixed_value]
        
        # Group data by compiler and x_axis
        result = {}
        for compiler in filtered_data['Compiler'].unique():
            compiler_data = filtered_data[filtered_data['Compiler'] == compiler]
            # Sort by x_axis
            compiler_data = compiler_data.sort_values(by=x_axis)
            result[compiler] = compiler_data
        
        return result
    
    def prepare_data_for_speedup_plot(self, baseline_compiler='b_b', x_axis=None, metric='Time', fixed_param=None, fixed_value=None):
        """
        Prepare data for speedup/improvement plotting relative to a baseline compiler.
        
        Parameters:
        -----------
        baseline_compiler : str
            Compiler to use as baseline for speedup calculation
        x_axis : str
            Column to use for x-axis (self.dimension_para or 'Array Size')
        metric : str
            Metric to calculate speedup/improvement for ('Time' or 'Energy')
        fixed_param : str, optional
            Parameter to keep fixed ('Array Size' or self.dimension_para)
        fixed_value : int, optional
            Value to keep the fixed parameter at
        
        Returns:
        --------
        dict
            Dictionary with compiler names as keys and DataFrames with speedup values as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if x_axis not in [self.dimension_para, 'Array Size']:
            raise ValueError("x_axis must be self.dimension_para or 'Array Size'")
        
        if metric not in ['Time', 'Energy']:
            raise ValueError("metric must be 'Time' or 'Energy'")
        
        filtered_data = self.data.copy()
        
        # Apply filter if a parameter is fixed
        if fixed_param and fixed_value is not None:
            filtered_data = filtered_data[filtered_data[fixed_param] == fixed_value]
        
        # Group data by compiler and x_axis
        result = {}
        
        # First, extract baseline data
        baseline_data = filtered_data[filtered_data['Compiler'] == baseline_compiler]
        baseline_data = baseline_data.sort_values(by=x_axis)
        
        # For each compiler, calculate speedup
        for compiler in filtered_data['Compiler'].unique():
            if compiler == baseline_compiler:
                continue  # Skip baseline
                
            compiler_data = filtered_data[filtered_data['Compiler'] == compiler]
            compiler_data = compiler_data.sort_values(by=x_axis)
            
            # Ensure we have matching x_axis values
            merged_data = pd.merge(
                baseline_data[[x_axis, metric]],
                compiler_data[[x_axis, metric]],
                on=x_axis,
                suffixes=('_baseline', '_current')
            )
            
            # Calculate speedup (baseline / current for time, current / baseline for energy)
            if metric == 'Time':
                merged_data['Speedup'] = merged_data[f'{metric}_baseline'] / merged_data[f'{metric}_current']
            else:  # Energy
                merged_data['Improvement'] = merged_data[f'{metric}_baseline'] / merged_data[f'{metric}_current']
            
            result[compiler] = merged_data
        
        return result
    
    def prepare_data_for_heatmap(self, compiler, metric='Time'):
        """
        Prepare data for heatmap plotting with Array Size on one axis and scale Dimension on the other.
        
        Parameters:
        -----------
        compiler : str
            Compiler to generate heatmap for
        metric : str
            Metric to visualize ('Time' or 'Energy')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame in matrix format suitable for heatmap plotting
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in ['Time', 'Energy']:
            raise ValueError("metric must be 'Time' or 'Energy'")
        
        # Filter data for the specified compiler
        compiler_data = self.data[self.data['Compiler'] == compiler]
        
        # Get unique values for axes
        array_sizes = sorted(compiler_data['Array Size'].unique())
        scale_dims = sorted(compiler_data[self.dimension_para].unique())
        
        # Create an empty matrix
        matrix = np.zeros((len(array_sizes), len(scale_dims)))
        
        # Fill the matrix with metric values
        for i, array_size in enumerate(array_sizes):
            for j, scale_dim in enumerate(scale_dims):
                cell_data = compiler_data[
                    (compiler_data['Array Size'] == array_size) & 
                    (compiler_data[self.dimension_para] == scale_dim)
                ]
                if not cell_data.empty:
                    matrix[i, j] = cell_data[metric].values[0]
        
        # Create a DataFrame for easier plotting
        heatmap_df = pd.DataFrame(matrix, index=array_sizes, columns=scale_dims)
        heatmap_df.index.name = 'Array Size'
        heatmap_df.columns.name = self.dimension_para
        
        return heatmap_df
    
    def prepare_data_for_grouped_comparison(self, metric='Time'):
        """
        Prepare data for grouped bar charts comparing all compilers.
        Each subplot is for a fixed array size, with scale dimension as x-axis.
        At each x-axis point, all compilers are grouped together for direct comparison.
        
        Parameters:
        -----------
        metric : str
            Metric to visualize ('Time' or 'Energy')
        
        Returns:
        --------
        dict
            Nested dictionary with structure:
            {
                array_size_1: {
                    scale_dim_1: {compiler_1: value, compiler_2: value, ...},
                    scale_dim_2: {compiler_1: value, compiler_2: value, ...},
                    ...
                },
                array_size_2: {
                    ...
                },
                ...
            }
            Also includes 'dimensions' key containing:
            {
                'array_sizes': [...],  # Sorted array sizes
                'scale_dims': [...],  # Sorted scale dimensions
                'compilers': [...]     # List of compiler names
            }
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in ['Time', 'Energy']:
            raise ValueError("metric must be 'Time' or 'Energy'")
        
        # Get all unique values
        array_sizes = sorted(self.data['Array Size'].unique())
        scale_dims = sorted(self.data[self.dimension_para].unique())
        compilers = sorted(self.data['Compiler'].unique())
        
        # Create result structure
        result = {
            'dimensions': {
                'array_sizes': array_sizes,
                'scale_dims': scale_dims,
                'compilers': compilers
            }
        }
        
        # Fill in data
        for array_size in array_sizes:
            result[array_size] = {}
            
            for scale_dim in scale_dims:
                result[array_size][scale_dim] = {}
                
                # Get data for this combination
                subset = self.data[
                    (self.data['Array Size'] == array_size) & 
                    (self.data[self.dimension_para] == scale_dim)
                ]
                
                # Extract values for each compiler
                for compiler in compilers:
                    compiler_data = subset[subset['Compiler'] == compiler]
                    if not compiler_data.empty:
                        result[array_size][scale_dim][compiler] = compiler_data[metric].values[0]
                    else:
                        # Handle missing data
                        result[array_size][scale_dim][compiler] = None
        
        return result

    def get_compiler_friendly_names(self):
        """
        Returns a dictionary mapping compiler codes to more descriptive names.
        
        Returns:
        --------
        dict
            Dictionary mapping compiler codes to friendly names
        """
        return {
            'b_b': 'Baseline Compiler',
            's_p': 'Parallel Compiler',
            's_s': 'Scatter Compiler',
            's_sp': 'Scatter-Parallel Compiler'
        }

    def get_summary_statistics(self):
        """
        Generate summary statistics for each compiler across all configurations.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Group by compiler and calculate statistics
        summary = self.data.groupby('Compiler').agg({
            'Time': ['mean', 'min', 'max', 'std'],
            'Energy': ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        return summary