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
    """Parse command line arguments for binary classification prediction
    
    Returns:
        tuple: (strgpu, batch_size, threshold, data_path, output)
            - strgpu: GPU device numbers
            - batch_size: Batch size for inference
            - threshold: Classification probability cutoff
            - data_path: Directory containing input data
            - output: Results output directory
    """
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1', '1,2', '1,3,5')")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size,(eg. 1)')
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
    """Execute end-to-end binary classification pipeline
    
    Args:
        strgpu: GPU devices to use (comma-separated string)
        data_path: Directory containing token.pt and accession.txt
        batch_size: Number of samples per inference batch
        output: Directory to save prediction results
        param_path: Path to pretrained model parameters
        threshold: Decision boundary for positive class
        seed: Random seed for reproducibility
        
    Workflow:
        1. Configure hardware device
        2. Load tokenized protein sequences
        3. Initialize model with pretrained weights
        4. Run batch inference with probability calculation
        5. Save predictions and filter enzyme sequences
    """
    # Hardware configuration using CUDA if available
    device = set_cuda(strgpu, seed)
    # Initialize dataloader for efficient batch processing
    dataset = EnzymeData(os.path.join(data_path, "token.pt"))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Model initialization and weight loading
    ESMenzyme = EnzymeBinary(n_class=2).to(device)
    # Multi-GPU configuration with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        ESMenzyme = nn.DataParallel(ESMenzyme)
        # Adjust parameter keys for multi-GPU format
        param = torch.load(param_path)
        param = dict(param)
        param = {'module.' + key: value for key, value in param.items()}
        ESMenzyme.load_state_dict(param)
    # Single GPU or CPU configuration
    elif torch.cuda.device_count() == 1:
        ESMenzyme.load_state_dict(torch.load(param_path))
    else:
        ESMenzyme.load_state_dict(torch.load(param_path, map_location=device))
    # Prediction buffers initialization
    all_prob = np.array([])  # Store class probabilities
    all_max = np.array([]).astype(bool)  # Store boolean predictions
    # Inference loop with gradient disabled
    with torch.no_grad():
        ESMenzyme.eval()
        for data, _ in dataloader:
            data = data.to(device)
            prediction = ESMenzyme(data)    
            pred = F.softmax(prediction, 1).cpu().detach().numpy()
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
    # Write comprehensive prediction results
    with open(os.path.join(output, "binary_result.txt"), 'w') as fw:
        fw.write('Accession\tIsEnzyme\tProbability\n')
        for a, i, p in zip(accession, string_list, all_prob):
            fw.write(f'{a}\t{i}\t{p}\n')
    # acquire the enzyme token
    token = torch.load(os.path.join(data_path, "token.pt")).to(torch.long)
    enzyme_token = token[all_max]
    torch.save(enzyme_token.clone(), os.path.join(data_path, "enzyme_token.pt"))
    accession = pd.read_table(os.path.join(output, "binary_result.txt"))
    enzyme_accession = list(accession[all_max]['Accession'])
    with open(os.path.join(data_path, "enzyme_accession.txt"),'w') as fw:
        for i in enzyme_accession:
            fw.write(str(i)+'\n')

def main():
    """Orchestrate the binary classification prediction pipeline
    
    Execution Flow:
    1. Parse command line arguments for runtime configuration
    2. Execute prediction workflow with pretrained model
    3. Generate enzyme identification results and filtered data
    """
    # Retrieve all runtime parameters from command line
    strgpu, batch_size, threshold, data_path, output = argument()
    
    # Execute core prediction workflow with parameters:
    # - param_path: Location of pretrained model weights
    # - seed: Fixed random seed for reproducibility (2024)
    prediction(
        strgpu=strgpu,
        data_path=data_path,
        batch_size=batch_size,
        output=output,
        param_path=f'{sys.path[0]}/model_param/binary/esm650_2layer_epoch2.pt',
        threshold=threshold,
        seed=2024
    )
    if os.path.getsize(os.path.join(data_path, "enzyme_accession.txt")) == 0:
        print("No enzyme sequences were identified. Please check the input data or adjust the threshold.")
        sys.exit(1)

if __name__ == '__main__':
    main()


