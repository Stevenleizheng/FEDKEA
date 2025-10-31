import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import EsmEnzymeFirst
from utilis import EnzymeData, set_cuda

def argument():
    """Parse command line arguments for enzyme prediction task
    
    Returns:
        tuple: (strgpu, batch_size, data_path, output)
            - strgpu: GPU device numbers 
            - batch_size: Batch size for model inference
            - data_path: Path to directory containing processed data
            - output: Directory for saving prediction results
    """
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size,(eg.2)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    parser.add_argument('-r', '--running_mode', type=int, default=1, help='running_mode: 1 for both enzyme identification and function prediction, 2 for function prediction only')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    data_path = args.data_path
    output = args.output
    running_mode = args.running_mode
    return strgpu, batch_size, data_path, output, running_mode

def prompt(strgpu, data_path, batch_size, output, 
           param_path, running_mode, seed=2024):
    """Generate enzyme classification prompts using pre-trained model
    
    Args:
        strgpu: GPU devices to use 
        data_path: Directory containing enzyme_token.pt
        batch_size: Number of samples per inference batch
        output: Directory for saving results
        param_path: Path to pre-trained model parameters
        running_mode: 1 for both enzyme identification and function prediction, 2 for function prediction only
        seed: Random seed for reproducibility
    
    Workflow:
        1. Configure hardware device
        2. Load filtered enzyme tokens
        3. Initialize model with pre-trained weights
        4. Generate prompt embeddings
        5. Save embeddings for downstream tasks
    """
    if running_mode == 2:
        subprocess.run(["rm", "-rf", os.path.join(data_path, "enzyme_token.pt")], check=True)
        subprocess.run(["rm", "-rf", os.path.join(data_path, "enzyme_accession.txt")], check=True)
        subprocess.run(["cp", os.path.join(data_path, "accession.txt"), os.path.join(data_path, "enzyme_accession.txt")], check=True)
        subprocess.run(["cp", os.path.join(data_path, "token.pt"), os.path.join(data_path, "enzyme_token.pt")], check=True)
    # Configure CUDA device and reproducibility
    device = set_cuda(strgpu, seed)
    # Dataloader
    dataset = EnzymeData(os.path.join(data_path, "enzyme_token.pt"))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)    
    # Model
    ESMenzyme = EsmEnzymeFirst(n_class=7).to(device)
    if  torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ESMenzyme = nn.DataParallel(ESMenzyme)
        param = torch.load(param_path)
        param = dict(param)
        param = {'module.' + key: value for key, value in param.items()}
        ESMenzyme.load_state_dict(param)
    elif torch.cuda.device_count() == 1:
        ESMenzyme.load_state_dict(torch.load(param_path))
    else:
        ESMenzyme.load_state_dict(torch.load(param_path, map_location=device))
    # Prompt
    os.makedirs(output, exist_ok = True)
    prompt = torch.empty((0,1280))
    with torch.no_grad():
        ESMenzyme.eval()
        for data, _ in dataloader:
            data = data.to(device)
            tmp = ESMenzyme(data)[0].cpu()
            prompt = torch.cat((prompt, tmp), dim=0)
        torch.save(prompt.clone(), os.path.join(data_path, "enzyme_prompt_first.pt"))
def main():  
    """Orchestrate the enzyme prompt generation pipeline
    
    Execution Flow:
    1. Parse command line arguments for runtime configuration
    2. Execute prompt generation workflow with pretrained model
    3. Save embeddings for downstream EC number prediction
    """
    # Retrieve configuration parameters from command line
    strgpu, batch_size, data_path, output, running_mode = argument()
    
    # Launch prompt generation with:
    # - param_path: Location of fine-tuned model weights
    # - seed: Fixed random seed for reproducibility (2024)
    prompt(
        strgpu=strgpu,
        data_path=data_path,
        batch_size=batch_size,
        output=output,
        param_path=f'{sys.path[0]}/model_param/first/ft5_MLP_BN_epoch5.pt',
        running_mode=running_mode,
        seed=2024
    )

if __name__ == '__main__':
    main()
