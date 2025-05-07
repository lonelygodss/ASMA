import os
import pandas as pd
import numpy as np
import re

class DetailedMetricsLoader:
    """
    A class to load and process detailed simulation metrics from a specially
    formatted CSV file that contains data organized by array size.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the detailed metrics loader.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the CSV file to load
        """
        self.data = None
        self.array_sizes = []
        self.compilers = []
        self.metrics = ['Bank-Tile', 'Intra-Tile', 'Excecution']  # Default metrics names from the file
        
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        """
        Load and parse the specially formatted CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        
        Returns:
        --------
        pd.DataFrame
            The processed data as a DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        
        # Read the raw file content to parse the special format
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Process the data
        processed_data = []
        current_array_size = None
        header_row = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is an array size row
            array_size_match = re.match(r'Array size,(\d+),,', line)
            if array_size_match:
                current_array_size = int(array_size_match.group(1))
                if current_array_size not in self.array_sizes:
                    self.array_sizes.append(current_array_size)
                continue
                
            # Check if this is the header row
            if line.startswith('compiler,'):
                header_row = [col.strip() for col in line.split(',')]
                self.metrics = header_row[1:]  # Store metrics names
                continue
                
            # This should be a data row
            if current_array_size is not None and header_row is not None:
                values = line.split(',')
                compiler = values[0]
                if compiler not in self.compilers:
                    self.compilers.append(compiler)
                
                # Create a row for each compiler and array size
                row_data = {
                    'Array Size': current_array_size,
                    'Compiler': compiler
                }
                
                # Add metric values
                for i, metric in enumerate(self.metrics):
                    if i + 1 < len(values):
                        try:
                            row_data[metric] = float(values[i + 1])
                        except ValueError:
                            row_data[metric] = None
                    else:
                        row_data[metric] = None
                
                processed_data.append(row_data)
        
        # Convert to DataFrame
        self.data = pd.DataFrame(processed_data)
        self.array_sizes.sort()
        
        return self.data
    
    def get_array_sizes(self):
        """Get the array sizes found in the data."""
        return self.array_sizes
    
    def get_compilers(self):
        """Get the compilers found in the data."""
        return self.compilers
    
    def get_metrics(self):
        """Get the metrics found in the data."""
        return self.metrics
    
    def filter_data(self, compiler=None, array_size=None):
        """
        Filter the data based on specified criteria.
        
        Parameters:
        -----------
        compiler : str or list, optional
            Compiler name(s) to filter by
        array_size : int or list, optional
            Array size(s) to filter by
        
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
        
        return filtered_data
    
    def prepare_data_by_array_size(self):
        """
        Prepare data grouped by array size.
        
        Returns:
        --------
        dict
            Dictionary with array sizes as keys and DataFrames as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        result = {}
        for array_size in self.array_sizes:
            array_data = self.data[self.data['Array Size'] == array_size]
            result[array_size] = array_data.copy()
        
        return result
    
    def prepare_stacked_bar_data(self, metrics=None):
        """
        Prepare data for creating stacked bar charts.
        
        Parameters:
        -----------
        metrics : list, optional
            List of metrics to include in the stacked bars.
            Default is ['Bank-Tile', 'Intra-Tile', 'Excecution']
        
        Returns:
        --------
        dict
            Dictionary with array sizes as keys and nested dictionaries as values.
            The nested dictionaries have compilers as keys and dictionaries of metric values as values.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metrics is None:
            metrics = self.metrics
        
        result = {}
        for array_size in self.array_sizes:
            array_data = self.data[self.data['Array Size'] == array_size]
            
            compiler_data = {}
            for compiler in self.compilers:
                compiler_row = array_data[array_data['Compiler'] == compiler]
                if not compiler_row.empty:
                    # Extract metric values for this compiler
                    metric_values = {}
                    for metric in metrics:
                        if metric in compiler_row.columns:
                            metric_values[metric] = compiler_row[metric].values[0]
                    
                    compiler_data[compiler] = metric_values
            
            result[array_size] = compiler_data
        
        return result
    
    def prepare_comparison_data(self, metric):
        """
        Prepare data for comparing a single metric across compilers and array sizes.
        
        Parameters:
        -----------
        metric : str
            The metric to compare
        
        Returns:
        --------
        pd.DataFrame
            Pivoted DataFrame with array sizes as index and compilers as columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")
        
        # Filter to just the required columns
        filtered_data = self.data[['Array Size', 'Compiler', metric]].copy()
        
        # Pivot the data
        pivot_data = filtered_data.pivot(index='Array Size', columns='Compiler', values=metric)
        
        return pivot_data
    
    def calculate_total_per_compiler(self):
        """
        Calculate the total of all metrics for each compiler at each array size.
        Useful for comparing overall performance.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with array sizes and compilers, and a 'Total' column
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        
        # Calculate total across all numeric metrics
        numeric_metrics = [col for col in self.metrics if pd.api.types.is_numeric_dtype(data_copy[col])]
        data_copy['Total'] = data_copy[numeric_metrics].sum(axis=1)
        
        # Return with only necessary columns
        return data_copy[['Array Size', 'Compiler', 'Total'] + numeric_metrics]
    
    def get_compiler_friendly_names(self):
        """
        Returns a dictionary mapping compiler codes to more descriptive names.
        
        Returns:
        --------
        dict
            Dictionary mapping compiler codes to friendly names
        """
        return {
            'bb': 'Baseline Compiler',
            'sp': 'Parallel Compiler',
            'ss': 'Scatter Compiler',
            'ssp': 'Scatter-Parallel Compiler'
        }
    
    def prepare_normalized_data(self, baseline_compiler='bb'):
        """
        Prepare normalized data relative to a baseline compiler.
        
        Parameters:
        -----------
        baseline_compiler : str
            Compiler to use as baseline
        
        Returns:
        --------
        dict
            Dictionary with array sizes as keys and DataFrames with normalized values as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        result = {}
        for array_size in self.array_sizes:
            array_data = self.data[self.data['Array Size'] == array_size].copy()
            
            # Get baseline data
            baseline_data = array_data[array_data['Compiler'] == baseline_compiler]
            if baseline_data.empty:
                continue
                
            normalized_rows = []
            
            # For each compiler, normalize against baseline
            for compiler in self.compilers:
                compiler_data = array_data[array_data['Compiler'] == compiler]
                if compiler_data.empty:
                    continue
                
                normalized_row = {
                    'Array Size': array_size,
                    'Compiler': compiler
                }
                
                # Normalize each metric
                for metric in self.metrics:
                    if metric in baseline_data.columns and metric in compiler_data.columns:
                        baseline_value = baseline_data[metric].values[0]
                        if baseline_value != 0:  # Avoid division by zero
                            normalized_row[f'Normalized {metric}'] = compiler_data[metric].values[0] / baseline_value
                
                normalized_rows.append(normalized_row)
            
            if normalized_rows:
                result[array_size] = pd.DataFrame(normalized_rows)
        
        return result
    
    def get_improvement_over_baseline(self, metric, baseline_compiler='bb'):
        """
        Calculate improvement ratio over baseline for a specific metric.
        
        Parameters:
        -----------
        metric : str
            The metric to compare
        baseline_compiler : str
            Compiler to use as baseline
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with improvement ratios for each compiler and array size
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")
        
        improvement_data = []
        
        for array_size in self.array_sizes:
            array_data = self.data[self.data['Array Size'] == array_size]
            
            # Get baseline value
            baseline_row = array_data[array_data['Compiler'] == baseline_compiler]
            if baseline_row.empty or pd.isna(baseline_row[metric].values[0]) or baseline_row[metric].values[0] == 0:
                continue
                
            baseline_value = baseline_row[metric].values[0]
            
            # Calculate improvement for each compiler
            for compiler in self.compilers:
                if compiler == baseline_compiler:
                    improvement = 1.0  # Baseline is same as itself
                else:
                    compiler_row = array_data[array_data['Compiler'] == compiler]
                    if compiler_row.empty or pd.isna(compiler_row[metric].values[0]):
                        continue
                    
                    compiler_value = compiler_row[metric].values[0]
                    
                    # Lower is better for most metrics (time, cycles, etc.)
                    improvement = baseline_value / compiler_value if compiler_value != 0 else float('inf')
                
                improvement_data.append({
                    'Array Size': array_size,
                    'Compiler': compiler,
                    'Improvement': improvement
                })
        
        return pd.DataFrame(improvement_data)