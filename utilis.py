import os
import torch
import pandas as pd
import numpy as np
import random
import time
from torch.utils.data import DataLoader, Dataset


class EnzymeData(Dataset):
    """DataLoader"""
    def __init__(self, data_path):
        """Initialization"""
        super(EnzymeData, self).__init__()    
        self.x = np.array(pd.read_table(data_path, header=None, index_col=None))

    def __len__(self):
        """Number of samples"""
        return len(self.x)

    def __getitem__(self, idx):
        """Load data in batches"""
        x_long = torch.tensor(self.x[idx], dtype=torch.long)
        x_float = torch.tensor(self.x[idx], dtype=torch.float32)
        return x_long, x_float

class EnzData(Dataset):
    """DataLoader"""
    def __init__(self, prompt_data):
        """Initialization"""
        super(EnzData, self).__init__()    
        self.x = np.array(prompt_data)

    def __len__(self):
        """Number of samples"""
        return len(self.x)

    def __getitem__(self, idx):
        """Load data in batches"""
        x_float = torch.tensor(self.x[idx], dtype=torch.float32)
        return x_float

class Timer:
    """Record the time of a run"""
    def __init__(self):
        """Log time and auto-start"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer"""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and log time"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def sum(self):
        """返回时间总和"""
        run_time = round(self.times[-1])
        hour = run_time // 3600
        minute = (run_time - 3600 * hour) // 60
        second = run_time - 3600 * hour - 60 * minute
        print(f'finished use time: {hour}h{minute}m{second}s')

def set_cuda(strgpu, seed = 2024):
    """GPUs and random number seeds"""
    if strgpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = strgpu
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device

