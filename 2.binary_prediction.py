import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import EnzymeBinary
from utilis import EnzymeData, set_cuda


def argument():
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1', '1,2', '1,3,5')")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size,(eg. 32)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold,(eg. 0.5)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    threshold = args.threshold
    data_path = args.data_path
    output = args.output
    return strgpu, batch_size, threshold, data_path, output

def prediction(strgpu, data_path, batch_size, output,
               param_path, threshold=0.5, seed=2024):
    """Binary classification task"""
    # Device
    device = set_cuda(strgpu, seed)
    # Dataloader
    dataset = EnzymeData(os.path.join(data_path, "token.pt"))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Model
    ESMenzyme = EnzymeBinary(n_class=2).to(device)
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
    # Prediction
    all_prob = np.array([])
    all_max = np.array([]).astype(bool)
    with torch.no_grad():
        ESMenzyme.eval()
        for data, _ in dataloader:
            data = data.to(device)
            prediction = ESMenzyme(data)    
            pred = F.softmax(prediction,1).cpu().detach().numpy()
            all_prob = np.concatenate((all_prob, pred[:, 1]))
            all_max = np.concatenate((all_max, (pred[:, 1]>threshold).astype(bool)))
        string_list = ['True' if item else 'False' for item in all_max]
    # write into the specified file
    os.makedirs(output, exist_ok = True)
    accession = []
    with open(os.path.join(data_path, "accession.txt"), 'r') as f:
        while 1:
            a = f.readline().strip()
            if not a:
                break
            accession.append(a)
    with open(os.path.join(output, "binary_result.txt"), 'w') as fw:
        fw.write('Accession\tIsEnzyme\tProbability\n')
        for a, i, p in zip(accession, string_list, all_prob):
            fw.write(f'{a}\t{i}\t{p}\n')
    # acquire the enzyme token
    token = torch.load(os.path.join(data_path, "token.pt")).to(torch.long)
    enzyme_token = token[all_max] # os.path.join(data_path, "enzyme_accession.txt")
    torch.save(enzyme_token.clone(), os.path.join(data_path, "enzyme_token.pt"))
    accession = pd.read_table(os.path.join(output, "binary_result.txt"))
    enzyme_accession = list(accession[all_max]['Accession'])
    with open(os.path.join(data_path, "enzyme_accession.txt"),'w') as fw:
        for i in enzyme_accession:
            fw.write(str(i)+'\n')

def main():
    """Main program running!"""
    strgpu, batch_size, threshold, data_path, output = argument()
    prediction(strgpu=strgpu, data_path=data_path, batch_size=batch_size, output=output,
                param_path=f'{sys.path[0]}/model_param/binary/esm650_2layer_epoch2.pt', threshold=threshold, seed=2024)
if __name__ == '__main__':
    main()

