import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import EsmEnzymeFirst
from utilis import EnzymeData, set_cuda

def argument():
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-g', '--gpu', type=str, help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, help='batch size,(eg. 32)')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    return strgpu, batch_size

def prompt(strgpu, data_path, batch_size, 
               param_path, seed=2024):
    """Prompt"""
    # Device
    device = set_cuda(strgpu, seed)
    # Dataloader
    dataset = EnzymeData(data_path)
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
    prompt = pd.DataFrame(data=None, columns=range(1280))
    with torch.no_grad():
        ESMenzyme.eval()
        for data, _ in dataloader:
            data = data.to(device)
            pro = pd.DataFrame(ESMenzyme(data)[0].cpu().detach().numpy(), columns=range(1280))
            prompt = pd.concat([prompt,pro])
        prompt.to_csv("data/enzyme_prompt_first.txt", sep='\t', header=False, index=False)
def main():
    """Main program running!"""
    strgpu, batch_size = argument()
    prompt(strgpu=strgpu, data_path='data/enzyme_token.txt', batch_size=batch_size, 
                param_path='model_param/first/ft3_MLP_BN_save.pt', seed=2024)
if __name__ == '__main__':
    main()
