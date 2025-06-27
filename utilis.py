import os
import torch
import pandas as pd
import numpy as np
import random
import time
from torch.utils.data import DataLoader, Dataset


class EnzymeData(Dataset):
    """Custom Dataset for loading preprocessed enzyme data.
    
    Loads tensor data from file and provides dual-format conversion (long/float)
    for compatibility with different neural network layers.
    """
    def __init__(self, data_path):
        """Initialize the dataset with preprocessed tensor data.
        
        Args:
            data_path (str): Path to .pt file containing preprocessed tensor
        """
        super(EnzymeData, self).__init__()    
        self.x = torch.load(data_path)
    def __len__(self):
        """Get total number of samples in the dataset.
        
        Returns:
            int: Length of the data tensor
        """
        return len(self.x)

    def __getitem__(self, idx):
        """Retrieve and convert a single sample from the dataset.
        
        Args:
            idx (int): Index position of the sample
            
        Returns:
            tuple: (x_long, x_float) where:
                x_long (torch.LongTensor): For embedding layer input
                x_float (torch.FloatTensor): For dense layer input
        """
        x_long = self.x[idx].to(torch.long)
        x_float = self.x[idx].to(torch.float32)
        return x_long, x_float

class EnzData(Dataset):
    """Dataset for enzyme prompt data handling and conversion.
    
    Converts input data to numpy array and provides float tensor output
    for model processing.
    """
    def __init__(self, prompt_data):
        """Initialize with enzyme prompt data.
        
        Args:
            prompt_data (array-like): Input data containing enzyme prompts
        """
        super(EnzData, self).__init__()    
        self.x = np.array(prompt_data)

    def __len__(self):
        """Get total number of data samples.
        
        Returns:
            int: Total count of enzyme prompts in dataset
        """
        return len(self.x)

    def __getitem__(self, idx):
        """Retrieve and convert single data sample.
        
        Args:
            idx (int): Index position of the sample
            
        Returns:
            torch.FloatTensor: Floating-point tensor representation
                              of the enzyme prompt data
        """
        x_float = torch.tensor(self.x[idx], dtype=torch.float32)
        return x_float

class Timer:
    """Time tracking utility for measuring code execution durations.
    
    Maintains a history of time intervals and provides formatted output
    """
    def __init__(self):
        """Initialize timer with empty records and automatically start timing"""
        self.times = []
        self.start()

    def start(self):
        """Reset and begin a new timing measurement"""
        self.tik = time.time()

    def stop(self):
        """Finalize current measurement and store the duration
        
        Returns:
            float: Elapsed time in seconds since last start()
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def sum(self):
        """Calculate and format total time from most recent interval
        
        Prints human-readable time breakdown (hours, minutes, seconds)
        """
        run_time = round(self.times[-1])
        hour = run_time // 3600
        minute = (run_time - 3600 * hour) // 60
        second = run_time - 3600 * hour - 60 * minute
        print(f'finished use time: {hour}h{minute}m{second}s')

def set_cuda(strgpu, seed = 2024):
    """Configure GPU devices and initialize random seeds for reproducibility.
    
    Args:
        strgpu (str|None): Comma-separated GPU device IDs (e.g., '0,1'), or None for CPU
        seed (int): Random seed value for all supported libraries
    
    Returns:
        torch.device: Configured computation device (cuda or cpu)    
    """
    if strgpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = strgpu # Maintains existing Chinese comment about supercomputer compatibility
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device

