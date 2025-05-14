import os
import pandas as pd
import numpy as np
import re

class ModelPerformanceLoader:
    """
    A class to load and process model performance data from a specially
    formatted CSV file with batch size and memory utilization groups.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the model performance data loader.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the CSV file to load
        """
        self.data = {}
        self.batch_sizes = []
        self.mem_utils = []
        self.models = []
        self.metrics = ['Latency(ms)', 'Energy(mJ)']
        
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
        dict
            The processed data in a nested dictionary structure
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        
        # Read the raw file content to parse the special format
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Process the data
        current_batch_size = None
        current_mem_util = None
        header_row = None
        
        # Initialize data structure
        self.data = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a batch size row
            batch_size_match = re.match(r'Batch size,(\d+),', line)
            if batch_size_match:
                current_batch_size = int(batch_size_match.group(1))
                if current_batch_size not in self.batch_sizes:
                    self.batch_sizes.append(current_batch_size)
                # Initialize data structure for this batch size if needed
                if current_batch_size not in self.data:
                    self.data[current_batch_size] = {}
                continue
                
            # Check if this is a memory utilization row
            mem_util_match = re.match(r'Mem Util,(.+),', line)
            if mem_util_match:
                current_mem_util = mem_util_match.group(1)
                if current_mem_util not in self.mem_utils:
                    self.mem_utils.append(current_mem_util)
                # Initialize data structure for this memory utilization if needed
                if current_batch_size is not None and current_mem_util not in self.data[current_batch_size]:
                    self.data[current_batch_size][current_mem_util] = {}
                continue
                
            # Check if this is the "Proposed" section
            if line.startswith('Proposed'):
                current_mem_util = 'Proposed'
                if current_mem_util not in self.mem_utils:
                    self.mem_utils.append(current_mem_util)
                # Initialize data structure for the Proposed section if needed
                if current_batch_size is not None and current_mem_util not in self.data[current_batch_size]:
                    self.data[current_batch_size][current_mem_util] = {}
                continue
                
            # Check if this is the header row
            if line.startswith('Model,'):
                header_row = [col.strip() for col in line.split(',')]
                self.metrics = header_row[1:]  # Store metrics names
                continue
                
            # This should be a data row
            if current_batch_size is not None and current_mem_util is not None and header_row is not None:
                values = line.split(',')
                model = values[0]
                if model not in self.models:
                    self.models.append(model)
                
                # Create an entry for this model in the current batch size and memory utilization
                if model not in self.data[current_batch_size][current_mem_util]:
                    self.data[current_batch_size][current_mem_util][model] = {}
                
                # Add metric values
                for i, metric in enumerate(self.metrics):
                    if i + 1 < len(values) and values[i + 1]:
                        try:
                            self.data[current_batch_size][current_mem_util][model][metric] = float(values[i + 1])
                        except ValueError:
                            self.data[current_batch_size][current_mem_util][model][metric] = None
                    else:
                        self.data[current_batch_size][current_mem_util][model][metric] = None
        
        # Sort lists
        self.batch_sizes.sort()
        self.models.sort()
        
        return self.data
    
    def get_batch_sizes(self):
        """Get the batch sizes found in the data."""
        return self.batch_sizes
    
    def get_mem_utils(self):
        """Get the memory utilization settings found in the data."""
        return self.mem_utils
    
    def get_models(self):
        """Get the models found in the data."""
        return self.models
    
    def get_metrics(self):
        """Get the metrics found in the data."""
        return self.metrics
    
    def filter_data(self, batch_size=None, mem_util=None, model=None, metric=None):
        """
        Filter the data based on specified criteria.
        
        Parameters:
        -----------
        batch_size : int or list, optional
            Batch size(s) to filter by
        mem_util : str or list, optional
            Memory utilization setting(s) to filter by
        model : str or list, optional
            Model name(s) to filter by
        metric : str, optional
            Specific metric to return (if None, returns all metrics)
        
        Returns:
        --------
        dict or pd.DataFrame
            Filtered data, either as a nested dictionary or as a DataFrame if it can be flattened
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Start with a copy of the full data
        filtered_data = {}
        
        # Filter by batch size
        batch_sizes_to_include = self.batch_sizes
        if batch_size:
            if isinstance(batch_size, list):
                batch_sizes_to_include = [b for b in batch_size if b in self.batch_sizes]
            else:
                batch_sizes_to_include = [batch_size] if batch_size in self.batch_sizes else []
        
        # Build filtered data structure
        for b in batch_sizes_to_include:
            filtered_data[b] = {}
            
            # Filter by memory utilization
            mem_utils_to_include = self.mem_utils
            if mem_util:
                if isinstance(mem_util, list):
                    mem_utils_to_include = [m for m in mem_util if m in self.mem_utils]
                else:
                    mem_utils_to_include = [mem_util] if mem_util in self.mem_utils else []
            
            for m in mem_utils_to_include:
                if m in self.data[b]:
                    filtered_data[b][m] = {}
                    
                    # Filter by model
                    models_to_include = self.models
                    if model:
                        if isinstance(model, list):
                            models_to_include = [md for md in model if md in self.models]
                        else:
                            models_to_include = [model] if model in self.models else []
                    
                    for md in models_to_include:
                        if md in self.data[b][m]:
                            if metric and metric in self.metrics:
                                # Return just the specified metric
                                if self.data[b][m][md].get(metric) is not None:
                                    filtered_data[b][m][md] = {metric: self.data[b][m][md][metric]}
                            else:
                                # Return all metrics
                                filtered_data[b][m][md] = self.data[b][m][md].copy()
        
        # If the filtered result has a simple structure, convert to DataFrame for easier use
        try:
            if len(filtered_data.keys()) == 1:
                b = list(filtered_data.keys())[0]
                if len(filtered_data[b].keys()) == 1:
                    m = list(filtered_data[b].keys())[0]
                    # Convert single batch size, single mem util to DataFrame
                    df_data = []
                    for md, metrics in filtered_data[b][m].items():
                        row = {'Model': md, 'Batch Size': b, 'Mem Util': m}
                        row.update(metrics)
                        df_data.append(row)
                    if df_data:
                        return pd.DataFrame(df_data)
        except:
            pass  # If conversion fails, just return the dictionary
        
        return filtered_data
    
    def prepare_data_by_batch_size(self):
        """
        Prepare data grouped primarily by batch size.
        
        Returns:
        --------
        dict
            Dictionary with batch sizes as keys and nested dictionaries as values
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data.copy()
    
    def prepare_data_by_model(self):
        """
        Reorganize data to be grouped primarily by model.
        
        Returns:
        --------
        dict
            Dictionary with models as keys and nested dictionaries as values
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        model_data = {}
        
        for model in self.models:
            model_data[model] = {}
            
            for batch_size in self.batch_sizes:
                model_data[model][batch_size] = {}
                
                for mem_util in self.mem_utils:
                    if mem_util in self.data[batch_size] and model in self.data[batch_size][mem_util]:
                        model_data[model][batch_size][mem_util] = self.data[batch_size][mem_util][model].copy()
        
        return model_data
    
    def prepare_performance_comparison(self, metric='Latency(ms)', baseline_mem_util='100%'):
        """
        Prepare data for comparing performance of models across memory utils,
        normalized to a baseline memory utilization.
        
        Parameters:
        -----------
        metric : str
            Metric to compare (default: 'Latency(ms)')
        baseline_mem_util : str
            Memory utilization to use as baseline (default: '100%')
        
        Returns:
        --------
        dict
            Dictionary with structure {batch_size: {model: {mem_util: normalized_value}}}
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")
            
        if baseline_mem_util not in self.mem_utils:
            raise ValueError(f"Baseline memory utilization '{baseline_mem_util}' not found in the data")
        
        comparison_data = {}
        
        for batch_size in self.batch_sizes:
            comparison_data[batch_size] = {}
            
            for model in self.models:
                comparison_data[batch_size][model] = {}
                
                # Get baseline value if available
                baseline_value = None
                if (baseline_mem_util in self.data[batch_size] and 
                    model in self.data[batch_size][baseline_mem_util] and
                    metric in self.data[batch_size][baseline_mem_util][model]):
                    baseline_value = self.data[batch_size][baseline_mem_util][model][metric]
                
                if baseline_value is None or baseline_value == 0:
                    # If baseline is missing or zero, can't normalize
                    continue
                
                # Calculate normalized values
                for mem_util in self.mem_utils:
                    if (mem_util in self.data[batch_size] and 
                        model in self.data[batch_size][mem_util] and
                        metric in self.data[batch_size][mem_util][model]):
                        value = self.data[batch_size][mem_util][model][metric]
                        if value is not None:
                            # Normalize: for latency/energy, lower is better, so baseline/value
                            comparison_data[batch_size][model][mem_util] = baseline_value / value
        
        return comparison_data
    
    def prepare_data_for_plotting(self, metric='Latency(ms)'):
        """
        Prepare data in a format convenient for various plotting functions.
        Groups data by batch size, then by model, with values for each memory utilization.
        
        Parameters:
        -----------
        metric : str
            Metric to prepare for plotting (default: 'Latency(ms)')
        
        Returns:
        --------
        dict
            Dictionary with structure {batch_size: {model: {mem_util: value}}}
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")
        
        plot_data = {}
        
        for batch_size in self.batch_sizes:
            plot_data[batch_size] = {}
            
            for model in self.models:
                plot_data[batch_size][model] = {}
                
                for mem_util in self.mem_utils:
                    if (mem_util in self.data[batch_size] and 
                        model in self.data[batch_size][mem_util] and
                        metric in self.data[batch_size][mem_util][model]):
                        plot_data[batch_size][model][mem_util] = self.data[batch_size][mem_util][model][metric]
        
        return plot_data
    
    def prepare_speedup_data(self, reference_mem_util='100%', metric='Latency(ms)'):
        """
        Calculate speedup factors compared to a reference memory utilization.
        
        Parameters:
        -----------
        reference_mem_util : str
            Memory utilization to use as reference (default: '100%')
        metric : str
            Metric to calculate speedup for (default: 'Latency(ms)')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with batch sizes, models, memory utilizations, and speedup factors
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")
            
        if reference_mem_util not in self.mem_utils:
            raise ValueError(f"Reference memory utilization '{reference_mem_util}' not found in the data")
        
        speedup_data = []
        
        for batch_size in self.batch_sizes:
            for model in self.models:
                # Get reference value
                reference_value = None
                if (reference_mem_util in self.data[batch_size] and 
                    model in self.data[batch_size][reference_mem_util] and
                    metric in self.data[batch_size][reference_mem_util][model]):
                    reference_value = self.data[batch_size][reference_mem_util][model][metric]
                
                if reference_value is None or reference_value == 0:
                    # If reference is missing or zero, can't calculate speedup
                    continue
                
                # Calculate speedups for each memory utilization
                for mem_util in self.mem_utils:
                    if mem_util == reference_mem_util:
                        # Speedup against itself is always 1.0
                        speedup = 1.0
                    elif (mem_util in self.data[batch_size] and 
                          model in self.data[batch_size][mem_util] and
                          metric in self.data[batch_size][mem_util][model]):
                        value = self.data[batch_size][mem_util][model][metric]
                        if value is not None and value != 0:
                            # For metrics like latency and energy, lower is better, so reference/value
                            speedup = reference_value / value
                        else:
                            continue
                    else:
                        continue
                    
                    # Add to results
                    speedup_data.append({
                        'Batch Size': batch_size,
                        'Model': model,
                        'Mem Util': mem_util,
                        'Speedup': speedup
                    })
        
        return pd.DataFrame(speedup_data)
    
    def convert_to_dataframe(self):
        """
        Convert the entire hierarchical data structure to a flat DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Flattened data with columns for batch size, memory utilization, model, and metrics
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df_data = []
        
        for batch_size in self.batch_sizes:
            for mem_util in self.mem_utils:
                if mem_util not in self.data[batch_size]:
                    continue
                    
                for model in self.models:
                    if model not in self.data[batch_size][mem_util]:
                        continue
                    
                    row = {
                        'Batch Size': batch_size,
                        'Mem Util': mem_util,
                        'Model': model
                    }
                    
                    for metric in self.metrics:
                        if metric in self.data[batch_size][mem_util][model]:
                            row[metric] = self.data[batch_size][mem_util][model][metric]
                    
                    df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def get_model_friendly_names(self):
        """
        Returns a dictionary mapping model codes to more descriptive names.
        
        Returns:
        --------
        dict
            Dictionary mapping model codes to friendly names
        """
        return {
            'Llama2': 'Llama-2',
            'Llama3': 'Llama-3',
            'Gemma': 'Gemma',
            'Phi': 'Phi-3'
        }
    
    def get_mem_util_friendly_names(self):
        """
        Returns a dictionary mapping memory utilization codes to more descriptive names.
        
        Returns:
        --------
        dict
            Dictionary mapping memory utilization codes to friendly names
        """
        return {
            '70%': '70% Memory Utilization',
            '100%': '100% Memory Utilization',
            'Proposed': 'Proposed Method'
        }