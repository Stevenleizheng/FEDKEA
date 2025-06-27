import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier

from model import EnzymeGeneral
from utilis import EnzymeData, EnzData, set_cuda

def argument():
    """Configure command-line arguments for enzyme commission prediction task.
    
    Returns:
        tuple: Contains four elements:
            - strgpu (str): GPU device specification or 'None' for CPU
            - batch_size (int): Batch size for model inference
            - data_path (str): Path to directory containing input data files
            - output (str): Output directory for prediction results
    """
    parser = argparse.ArgumentParser(description='Enzyme commission prediction')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size,(eg. 1)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    data_path = args.data_path
    output = args.output
    return strgpu, batch_size, data_path, output

def first_prediction(strgpu, batch_size, seed, output,
                     data_path, param_path):
    """Perform first-level EC number prediction using trained model.
    
    Args:
        strgpu: GPU device specification string
        batch_size: Number of samples per inference batch
        seed: Random seed for reproducibility
        output: Output directory path (unused in current implementation)
        data_path: Directory containing enzyme_prompt_first.pt and enzyme_accession.txt
        param_path: Path to pre-trained model parameters
    
    Returns:
        tuple: (enzyme_result_sort, prompt_data_sort)
            - Sorted DataFrame with first-level predictions and probabilities
            - Sorted prompt data for downstream processing
    """
    # Device configuration using CUDA if available
    device = set_cuda(strgpu, seed)
    # Load preprocessed enzyme prompt data
    dataset = EnzymeData(os.path.join(data_path, "enzyme_prompt_first.pt"))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Model initialization and parallelization handling
    ESMenzyme = EnzymeGeneral(n_class=7).to(device)
    if  torch.cuda.device_count() > 1:
        ESMenzyme = nn.DataParallel(ESMenzyme)
        param = torch.load(param_path)
        param = dict(param)
        mlp_keys = ['fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'fc3.weight', 'fc3.bias', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 'fc4.weight', 'fc4.bias']
        mlp_param = {i : param[i] for i in mlp_keys}
        mlp_param = {'module.' + key: value for key, value in mlp_param.items()}
        ESMenzyme.load_state_dict(mlp_param)
    elif torch.cuda.device_count() == 1:
        param = torch.load(param_path)
        param = dict(param)
        mlp_keys = ['fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'fc3.weight', 'fc3.bias', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 'fc4.weight', 'fc4.bias']
        mlp_param = {i : param[i] for i in mlp_keys}
        ESMenzyme.load_state_dict(mlp_param)        
    else:
        param = torch.load(param_path, map_location=device)
        param = dict(param)
        mlp_keys = ['fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'fc3.weight', 'fc3.bias', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 'fc4.weight', 'fc4.bias']
        mlp_param = {i : param[i] for i in mlp_keys}
        ESMenzyme.load_state_dict(mlp_param)
    # Prediction
    all_prob = np.array([])
    all_max = np.array([])
    with torch.no_grad():
        ESMenzyme.eval()
        for _, data in dataloader:
            data = data.to(device)
            prediction = ESMenzyme(data)    
            pred = F.softmax(prediction,1).cpu().detach().numpy()
            all_prob = np.concatenate((all_prob, np.max(pred, axis=1)))
            all_max = np.concatenate((all_max, np.argmax(pred, axis=1)+1)).astype(int)
    # Result
    enzyme_accession = [] 
    with open(os.path.join(data_path, "enzyme_accession.txt"),'r') as f:
        while 1:
            a = f.readline().strip()
            if not a:
                break
            enzyme_accession.append(a)
    enzyme_result = {}
    enzyme_result['Accession'] = enzyme_accession
    enzyme_result['First'] = list(all_max)
    enzyme_result['FirstProbability'] = list(all_prob)
    enzyme_result = pd.DataFrame(enzyme_result)
    enzyme_result_sort = enzyme_result.sort_values(by='First')
    sort_index = enzyme_result.sort_values(by='First').index
    prompt_data = torch.load(os.path.join(data_path, "enzyme_prompt_first.pt")).to(torch.float32).numpy()
    prompt_data = pd.DataFrame(prompt_data)
    prompt_data_sort = prompt_data.reindex(sort_index)
    return enzyme_result_sort, prompt_data_sort

def prediction(strgpu, batch_size, prompt_data, decode_dict, n_class, seed, param_path):
    """Execute enzyme commission number prediction for a specific EC level.
    
    Args:
        strgpu: GPU device specification string
        batch_size: Number of samples per inference batch
        prompt_data: Input tensor data for prediction
        decode_dict: Mapping dictionary for converting model output to EC numbers
        n_class: Number of output classes for this prediction level
        seed: Random seed for reproducibility
        param_path: Path to pre-trained model parameters
    
    Returns:
        tuple: (all_prob, all_max)
            - Array of maximum probabilities for each prediction
            - Array of decoded EC number indices
    """
    # Configure computation device (GPU/CPU)
    device = set_cuda(strgpu, seed)
    # Prepare dataloader for batch processing
    dataset = EnzData(prompt_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Model
    ESMenzyme = EnzymeGeneral(n_class).to(device)
    if  torch.cuda.device_count() > 1:
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
    all_max = np.array([])
    with torch.no_grad():
        ESMenzyme.eval()
        for data in dataloader:
            data = data.to(device)
            prediction = ESMenzyme(data)    
            pred = F.softmax(prediction,1).cpu().detach().numpy()
            all_prob = np.concatenate((all_prob, np.max(pred, axis=1)))
            all_max = np.concatenate((all_max, np.argmax(pred, axis=1)))
    all_max = np.array([decode_dict[item] for item in all_max])
    return all_prob, all_max

def second_prediction(enzyme_result_sort, prompt_data_sort, seed, 
                      strgpu, batch_size):
    """Perform second-level EC number prediction using specialized sub-models.
    
    Args:
        enzyme_result_sort: DataFrame sorted by first-level predictions
        prompt_data_sort: Corresponding sorted prompt data for inference
        seed: Random seed for reproducibility
        strgpu: GPU device specification string
        batch_size: Number of samples per inference batch
    
    Returns:
        tuple: (enzyme_result_sort2, prompt_data_sort2)
            - DataFrame with second-level predictions and probabilities
            - Sorted prompt data for third-level processing
    """
    # Get unique first-level EC numbers to process
    nums = list(set(list(enzyme_result_sort['First'])))
    second_result = np.array([])
    second_probability = np.array([])
    # Process each first-level EC group separately
    for n in nums:
        need_index = enzyme_result_sort[enzyme_result_sort['First'] == n].index
        prompt_data = prompt_data_sort.reindex(need_index)
        # Model configuration for each EC class
        if n == 1:
            # EC 1.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:9, 13:10, 14:11, 17:12, 18:13, 0:14}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/1_MLP_BN_CLoss_epoch76.pt' )
        if n == 2:
            # EC 2.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 0:8}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/2_MLP_BN_CLoss_epoch15.pt' )
        if n == 3:
            # EC 3.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 4:2, 5:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/3_MLP_BN_LocalLoss_epoch12.pt' )
        if n == 4:
            # EC 4.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 6:3, 0:4}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/4_MLP_BN_CLoss_epoch38.pt' )
        if n == 5:
            # EC 5.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 4:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/5_MLP_BN_CLoss_epoch49.pt' )
        if n == 6:
            # EC 6.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 5:3, 0:4}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/6_MLP_BN_CLoss_epoch34.pt' )
        if n == 7:
            # EC 7.x.x.x class decoder and model
            decode_dict = {1:0, 2:1, 3:2, 4:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/7_MLP_BN_LocalLoss_epoch24.pt' )
        second_result = np.concatenate((second_result,all_max))
        second_probability = np.concatenate((second_probability,all_prob))
    # Result
    enzyme_result_sort['Second'] = [int(x) for x in second_result]
    enzyme_result_sort['SecondProbability'] = list(second_probability)
    enzyme_result_sort2 = enzyme_result_sort.sort_values(by=['First', 'Second'])
    sort_index = enzyme_result_sort.sort_values(by=['First', 'Second']).index
    prompt_data_sort2 = prompt_data_sort.reindex(sort_index)
    return enzyme_result_sort2, prompt_data_sort2

def calculate_knn_score(prompt_data, first, second, mode='third'):
    """Calculate K-nearest neighbors similarity score for EC number prediction.
    
    Args:
        prompt_data: Input feature vectors for prediction (n_samples, n_features)
        first: First-level EC number (enzyme class)
        second: Second-level EC number (enzyme subclass)
        mode: Model type directory name (default: 'third' level models)
    
    Returns:
        tuple: (pred, prob)
            - pred: Array of predicted EC numbers (third-level)
            - prob: Array of maximum probability scores for predictions
    
    File Path Structure:
        Loads precomputed embeddings and labels from:
        model_param/{mode}/ex_{first}/{first}_{second}_total_embedding.txt
        model_param/{mode}/ex_{first}/{first}_{second}_total_label.txt
    """
    # Load precomputed embedding vectors from training data
    data = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/{first}_{second}_total_embedding.txt', header = None, index_col= None)
    data = np.array(data)
    # Load corresponding EC number labels
    label = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/{first}_{second}_total_label.txt', header = None, index_col= None)
    label = np.array(label).flatten()
    # Initialize KNN classifier with 1 neighbor (best performing in experiments)
    neigh = KNeighborsClassifier(n_neighbors=1) 
    neigh.fit(data, label)
    # Generate predictions and probability estimates
    pred = neigh.predict(np.array(prompt_data))
    prob = neigh.predict_proba(np.array(prompt_data))
    # Extract maximum probability for each prediction
    prob = np.max(prob,axis=1)
    return pred, prob

def calculate_knn_score_fourth(prompt_data, first, second, third, mode='fourth'):
    """Calculate K-nearest neighbors similarity score for fourth-level EC prediction.
    
    Args:
        prompt_data: Input feature vectors for prediction (n_samples, n_features)
        first: First-level EC number (enzyme class)
        second: Second-level EC number (enzyme subclass)
        third: Third-level EC number (enzyme sub-subclass)
        mode: Model type directory name (default: 'fourth' level models)
    
    Returns:
        tuple: (pred, prob)
            - pred: Array of predicted EC numbers (fourth-level)
            - prob: Array of maximum probability scores for predictions
    
    File Path Structure:
        Loads precomputed embeddings and labels from:
        model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_total_embedding.txt
        model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_total_label.txt
    """
    # Load fourth-level embedding vectors from training data
    data = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_total_embedding.txt', header = None, index_col= None)
    data = np.array(data)
    # Load corresponding fourth-level EC labels
    label = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_total_label.txt', header = None, index_col= None)
    label = np.array(label).flatten()
    # Initialize KNN classifier with 1 neighbor (optimal from previous experiments)
    neigh = KNeighborsClassifier(n_neighbors=1) 
    neigh.fit(data, label) 
    # Generate predictions and probability estimates
    pred = neigh.predict(np.array(prompt_data))
    prob = neigh.predict_proba(np.array(prompt_data))
    prob = np.max(prob,axis=1)
    return pred, prob

def third_prediction(enzyme_result_sort2, prompt_data_sort2, seed, 
                      strgpu, batch_size):
    """Perform third-level EC number prediction using hybrid KNN/model approaches.
    
    Args:
        enzyme_result_sort2: DataFrame sorted by first and second-level predictions
        prompt_data_sort2: Corresponding sorted prompt data for inference
        seed: Random seed for reproducibility (unused in current implementation)
        strgpu: GPU device specification string
        batch_size: Number of samples per inference batch
    
    Returns:
        tuple: (enzyme_result_sort3, prompt_data_sort3)
            - DataFrame with third-level predictions and probabilities
            - Sorted prompt data for fourth-level processing
    
    Processing Logic:
        1. Identify unique (first, second) EC number pairs
        2. For each pair:
           - Use KNN scoring for common EC combinations
           - Use specialized models for complex cases
        3. Aggregate and sort results for final output
    """
    # Extract unique first-second EC combinations
    first_store = list(enzyme_result_sort2['First'])
    second_store = list(enzyme_result_sort2['Second'])
    fir_sec_store = []
    for i, n in zip(first_store, second_store):
        fir_sec_store.append((int(i),int(n)))
    # Identify unique EC pairs without duplicates
    seen = set()
    fir_sec_set = []
    for item in fir_sec_store:
        if item not in seen:
            fir_sec_set.append(item)
            seen.add(item)
    # Initialize result containers
    third_result = np.array([])
    third_probability = np.array([])
    # Process each unique EC 
    for i, n in fir_sec_set:
        # Handle EC 1.x.x.x class predictions
        if i == 1:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 3:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 4:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 5:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 7:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 8:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 10:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 11:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 13:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 18:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 14:
                # EC 1.14.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {14:0, 11:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss_epoch37.pt' )
            elif n == 17:
                # EC 1.17.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {7:0, 1:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch21.pt' )
        # Handle EC 2.x.x.x class predictions
        if i == 2:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 5:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                # EC 2.1.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 3:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch32.pt' )
            elif n == 3:
                # EC 2.3.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 3:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch25.pt' )
            elif n == 4:
                # EC 2.4.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 1:1, 99:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch44.pt' )
            elif n == 7:
                # EC 2.7.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {7:0, 1:1, 11:2, 4:3, 2:4, 8:5, 10:6, 0:7}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch32.pt' )
            elif n == 8:
                # EC 2.8.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch32.pt' )
        # Handle EC 3.x.x.x class predictions
        if i == 3:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                # EC 3.1.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 3:1, 26:2, 21:3, 11:4, 2:5, 4:6, 0:7}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch64.pt' )
            elif n == 2:
                # EC 3.2.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch10.pt' )
            elif n == 4:
                # EC 3.4.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {21:0, 24:1, 11:2, 23:3, 22:4, 19:5, 25:6, 0:7}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch65.pt' )
            elif n == 5:
                # EC 3.5.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 2:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch60.pt' )
            elif n == 6:
                # EC 3.6.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 5:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch43.pt' )
        # Handle EC 4.x.x.x class predictions
        if i == 4:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                # EC 4.1.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 99:1, 2:2, 3:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch33.pt' )
            elif n == 2:
                # EC 4.2.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 3:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch41.pt' )
            elif n == 3:
                # EC 4.3.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 3:1, 1:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch12.pt' )
        # Handle EC 5.x.x.x class predictions
        if i == 5:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 3:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                # EC 5.1.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 3:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch32.pt' )
            elif n == 4:
                # EC 5.4.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {99:0, 2:1, 3:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch29.pt' )
            elif n == 6:
                # EC 5.6.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss_epoch16.pt' )
        # Handle EC 6.x.x.x class predictions
        if i == 6:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 5:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 3:
                # EC 6.3.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 4:1, 5:2, 1:3, 3:4}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_CLoss_epoch55.pt' )
        # Handle EC 7.x.x.x class predictions
        if i == 7:
            # KNN-based predictions for common subclasses
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 3:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 4:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                # EC 7.1.x.x specialized model
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss_epoch10.pt' )
        third_result = np.concatenate((third_result,all_max))
        third_probability = np.concatenate((third_probability,all_prob))
    # Result
    enzyme_result_sort2['Third'] = [int(i) if isinstance(i, float) else i for i in third_result]
    enzyme_result_sort2['ThirdProbability'] = list(third_probability)
    enzyme_result_sort3 = enzyme_result_sort2.sort_values(by=['First', 'Second', 'Third'])
    sort_index = enzyme_result_sort2.sort_values(by=['First', 'Second', 'Third']).index
    prompt_data_sort3 = prompt_data_sort2.reindex(sort_index)
    return enzyme_result_sort3, prompt_data_sort3

def fourth_prediction(enzyme_result_sort3, prompt_data_sort3, seed, output, 
                      strgpu, batch_size):
    """Perform fourth-level EC number prediction using KNN similarity scoring.
    
    Args:
        enzyme_result_sort3: DataFrame sorted by first three EC levels
        prompt_data_sort3: Corresponding sorted prompt data for inference
        seed: Random seed for reproducibility (unused in current implementation)
        output: Output directory path for saving results
        strgpu: GPU device specification string (unused in KNN implementation)
        batch_size: Batch size parameter (unused in KNN implementation)
    
    Returns:
        None: Saves final results to enzyme_result_with_probability.csv in output directory
    """
    # Split data based on third-level prediction type (int vs string)
    enzyme_result_sort4 = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, int))]
    sort_index = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, int))].index
    prompt_data_sort4 = prompt_data_sort3.reindex(sort_index)
    # Store invalid entries for later recombination
    enzyme_result_sort5 = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, str))]
    sort_index = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, str))].index
    prompt_data_sort5 = prompt_data_sort3.reindex(sort_index)
    # Identify unique EC triplets (first.second.third) for processing
    first_store = list(enzyme_result_sort4['First'])
    second_store = list(enzyme_result_sort4['Second'])
    third_store = list(enzyme_result_sort4['Third'])
    fir_sec_thi_store = []
    for i, n, m in zip(first_store, second_store, third_store):
        fir_sec_thi_store.append((int(i),int(n),int(m)))
    # Remove duplicate EC triplets
    seen = set()
    fir_sec_thi_set = []
    for item in fir_sec_thi_store:
        if item not in seen:
            fir_sec_thi_set.append(item)
            seen.add(item)
    # Process each unique EC triplet with fourth-level KNN
    fourth_result = np.array([])
    fourth_probability = np.array([])
    for i, n, m in fir_sec_thi_set:
        if i == 1:
            if n == 14:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 11:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 14:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 17:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 7:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
        if i == 2:
            if n == 1:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 3:
                if m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 4:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')     
                elif m == 99:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')  
            if n == 7:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')     
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')      
                elif m == 7:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 8:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')     
                elif m == 10:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')  
                elif m == 11:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')  
            if n == 8:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')          
        if i == 3:
            if n == 1:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 11:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 21:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 26:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 2:
                if m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 4:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 11:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 19:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 21:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 22:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 23:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 24:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 25:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 5:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 6:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 5:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
        if i == 4:
            if n == 1:
                if m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 99:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 2:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 3:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
        if i == 5:
            if n == 1:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 4:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 99:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
            if n == 6:
                if m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
        if i == 6:
            if n == 3:
                if m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 3:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 4:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')  
                elif m == 5:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')            
        if i == 7:
            if n == 1:
                if m == 0:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 1:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
                elif m == 2:
                    need_index = enzyme_result_sort4[(enzyme_result_sort4['First'] == i) & (enzyme_result_sort4['Second'] == n) & (enzyme_result_sort4['Third'] == m)].index
                    prompt_data = prompt_data_sort4.reindex(need_index)
                    all_max, all_prob = calculate_knn_score_fourth(prompt_data, first=i, second=n, third=m, mode='fourth')
        fourth_result = np.concatenate((fourth_result,all_max))
        fourth_probability = np.concatenate((fourth_probability,all_prob))
    # Result
    enzyme_result_sort4 = enzyme_result_sort4.copy()
    enzyme_result_sort5 = enzyme_result_sort5.copy()
    enzyme_result_sort4['Fourth'] = list(fourth_result)
    enzyme_result_sort4['FourthProbability'] = list(fourth_probability)
    enzyme_result_sort5['Fourth'] = 0  
    enzyme_result_sort5['FourthProbability'] = 0
    enzyme_result_fourth = pd.concat([enzyme_result_sort4, enzyme_result_sort5], ignore_index=True)
    pattern = re.compile(r'[^.]*\.[^.]*\.[^.]*\.[^.]*[^.]*')
    result = []
    for _, row in enzyme_result_fourth.iterrows():
        matched_values = [value for value in row if isinstance(value, str) and pattern.match(value)]
        result.append(matched_values[0])
    enzyme_result_fourth['FinalResult'] = result
    enzyme_result_fourth.to_csv(os.path.join(output, "enzyme_result_with_probability.csv"), index=False)



def main():
    """Orchestrate the hierarchical enzyme classification pipeline.
    
    Workflow:
    1. Parse command line arguments
    2. Execute 4-level prediction cascade:
       - First: Binary enzyme classification
       - Second: Major EC class prediction
       - Third: EC subclass refinement
       - Fourth: Final EC number assignment
    3. Save final results to CSV
    """
    # Parse input arguments (GPU config, data paths, etc.)
    strgpu, batch_size, data_path, output = argument()
    # First-level prediction: Enzyme/non-enzyme classification
    enzyme_result_sort, prompt_data_sort = first_prediction(
                strgpu=strgpu, batch_size=batch_size, seed=2024, data_path=data_path, output=output, 
                param_path=f'{sys.path[0]}/model_param/first/ft5_MLP_BN_epoch5.pt')
    # Second-level prediction: Major EC class (1-7) determination
    enzyme_result_sort2, prompt_data_sort2 = second_prediction(
                      enzyme_result_sort=enzyme_result_sort, prompt_data_sort=prompt_data_sort, seed=2024, 
                      strgpu=strgpu, batch_size=batch_size)
    # Third-level prediction: EC subclass refinement
    enzyme_result_sort3, prompt_data_sort3 = third_prediction(
                      enzyme_result_sort2=enzyme_result_sort2, prompt_data_sort2=prompt_data_sort2, seed=2024, 
                      strgpu=strgpu, batch_size=batch_size)    
    # Fourth-level prediction: Final EC number assignment and output
    fourth_prediction(enzyme_result_sort3=enzyme_result_sort3, prompt_data_sort3=prompt_data_sort3, seed=2024,  
                      output=output, strgpu=strgpu, batch_size=batch_size)
if __name__ == '__main__':
    main()