import os
import sys
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
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size,(eg. 32)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    data_path = args.data_path
    output = args.output
    return strgpu, batch_size, data_path, output

def prompt(strgpu, data_path, batch_size, output, 
               param_path, seed=2024):
    """Prompt"""
    # Device
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
    prompt = torch.empty((0,1280))
    with torch.no_grad():
        ESMenzyme.eval()
        for data, _ in dataloader:
            data = data.to(device)
            tmp = ESMenzyme(data)[0].cpu()
            prompt = torch.cat((prompt, tmp), dim=0)
        torch.save(prompt.clone(), os.path.join(data_path, "enzyme_prompt_first.pt"))
def main():  
    """Main program running!"""
    strgpu, batch_size, data_path, output = argument()
    prompt(strgpu=strgpu, data_path=data_path, batch_size=batch_size, output=output,
                param_path=f'{sys.path[0]}/model_param/first/ft5_MLP_BN_epoch5.pt', seed=2024)
if __name__ == '__main__':
    main()
