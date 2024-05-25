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
    parser = argparse.ArgumentParser(description='Enzyme commission prediction')
    parser.add_argument('-g', '--gpu', type=str, help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, help='batch size,(eg. 32)')
    args = parser.parse_args()
    strgpu = args.gpu
    batch_size = args.batch_size
    return strgpu, batch_size

def first_prediction(strgpu, batch_size, seed, 
                     data_path, param_path):
    """First prediction"""
    # Device
    device = set_cuda(strgpu, seed)
    # Dataloader
    dataset = EnzymeData(data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Model
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
    with open(f'{sys.path[0]}/data/enzyme_accession.txt','r') as f:
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
    prompt_data = pd.read_table(data_path, header=None, index_col=None)
    prompt_data_sort = prompt_data.reindex(sort_index)
    return enzyme_result_sort, prompt_data_sort

def prediction(strgpu, batch_size, prompt_data, decode_dict, n_class, seed, param_path):
    # Device
    device = set_cuda(strgpu, seed)
    # Dataloader
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
    """Second prediction"""
    nums = list(set(list(enzyme_result_sort['First'])))
    second_result = np.array([])
    second_probability = np.array([])
    for n in nums:
        need_index = enzyme_result_sort[enzyme_result_sort['First'] == n].index
        prompt_data = prompt_data_sort.reindex(need_index)
        if n == 1:
            decode_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:9, 13:10, 14:11, 17:12, 18:13, 0:14}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/1_MLP_BN_LocalLoss.pt' )
        if n == 2:
            decode_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 0:8}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/2_MLP_BN_LocalLoss.pt' )
        if n == 3:
            decode_dict = {1:0, 2:1, 4:2, 5:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/3_MLP_BN_LocalLoss.pt' )
        if n == 4:
            decode_dict = {1:0, 2:1, 3:2, 6:3, 0:4}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/4_MLP_BN_LocalLoss.pt' )
        if n == 5:
            decode_dict = {1:0, 2:1, 3:2, 4:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/5_MLP_BN_LocalLoss.pt' )
        if n == 6:
            decode_dict = {1:0, 2:1, 3:2, 5:3, 0:4}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/6_MLP_BN_LocalLoss.pt' )
        if n == 7:
            decode_dict = {1:0, 2:1, 3:2, 4:3, 6:4, 0:5}
            decode_dict = {value: key for key, value in decode_dict.items()}
            all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                    decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                    param_path=f'{sys.path[0]}/model_param/second/7_MLP_BN_LocalLoss.pt' )
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
    """Calculate KNN score"""
    data = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/{first}_{second}_train_embedding.txt', header = None, index_col= None)
    data = np.array(data)
    label = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/{first}_{second}_train_label.txt', header = None, index_col= None)
    label = np.array(label).flatten()
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
    neigh.fit(data, label) 
    pred = neigh.predict(np.array(prompt_data))
    prob = neigh.predict_proba(np.array(prompt_data))
    prob = np.max(prob,axis=1)
    return pred, prob

def calculate_knn_score_fourth(prompt_data, first, second, third, mode='fourth'):
    """Calculate KNN score"""
    data = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_train_embedding.txt', header = None, index_col= None)
    data = np.array(data)
    label = pd.read_table(f'{sys.path[0]}/model_param/{mode}/ex_{first}/ex_{first}_{second}/{first}_{second}_{third}_train_label.txt', header = None, index_col= None)
    label = np.array(label).flatten()
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
    neigh.fit(data, label) 
    pred = neigh.predict(np.array(prompt_data))
    prob = neigh.predict_proba(np.array(prompt_data))
    prob = np.max(prob,axis=1)
    return pred, prob

def third_prediction(enzyme_result_sort2, prompt_data_sort2, seed, 
                      strgpu, batch_size):
    """Third prediction"""
    first_store = list(enzyme_result_sort2['First'])
    second_store = list(enzyme_result_sort2['Second'])
    fir_sec_store = []
    for i, n in zip(first_store, second_store):
        fir_sec_store.append((int(i),int(n)))
    seen = set()
    fir_sec_set = []
    for item in fir_sec_store:
        if item not in seen:
            fir_sec_set.append(item)
            seen.add(item)
    third_result = np.array([])
    third_probability = np.array([])
    for i, n in fir_sec_set:
        if i == 1:
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
            elif n == 8:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {4:0, 1:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 14:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {14:0, 11:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 17:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {7:0, 1:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 2:
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
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 3:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 3:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 3:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 4:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 1:1, 99:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 7:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {7:0, 1:1, 11:2, 4:3, 2:4, 8:5, 0:6}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 8:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 3:
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {3:0, 1:1, 26:2, 21:3, 11:4, 4:5, 2:6, 0:7}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 4:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {21:0, 24:1, 11:2, 23:3, 19:4, 22:5, 25:6, 0:7}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 5:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 2:2, 3:3, 0:4}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 4:1, 5:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 4:
            if n == 0:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                all_max, all_prob = calculate_knn_score(prompt_data, first=i, second=n, mode='third')
            elif n == 1:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 99:1, 2:2, 3:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 2:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 3:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 3:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 3:1, 1:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 5:
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
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 3:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 4:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {99:0, 2:1, 3:2, 0:3}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
            elif n == 6:
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 6:
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
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {2:0, 4:1, 5:2, 1:3, 3:4}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        if i == 7:
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
                need_index = enzyme_result_sort2[(enzyme_result_sort2['First'] == i) & (enzyme_result_sort2['Second'] == n)].index
                prompt_data = prompt_data_sort2.reindex(need_index)
                decode_dict = {1:0, 2:1, 0:2}
                decode_dict = {value: key for key, value in decode_dict.items()}
                all_prob, all_max = prediction(strgpu, batch_size, prompt_data = prompt_data,   
                decode_dict=decode_dict, n_class=len(decode_dict), seed=2024,
                param_path=f'{sys.path[0]}/model_param/third/ex_{i}/{i}_{n}_MLP_BN_LocalLoss.pt' )
        third_result = np.concatenate((third_result,all_max))
        third_probability = np.concatenate((third_probability,all_prob))
    # Result
    enzyme_result_sort2['Third'] = [int(i) if isinstance(i, float) else i for i in third_result]
    enzyme_result_sort2['ThirdProbability'] = list(third_probability)
    enzyme_result_sort3 = enzyme_result_sort2.sort_values(by=['First', 'Second', 'Third'])
    sort_index = enzyme_result_sort2.sort_values(by=['First', 'Second', 'Third']).index
    prompt_data_sort3 = prompt_data_sort2.reindex(sort_index)
    return enzyme_result_sort3, prompt_data_sort3

def fourth_prediction(enzyme_result_sort3, prompt_data_sort3, seed, 
                      strgpu, batch_size):
    """Fourth prediction"""
    enzyme_result_sort4 = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, int))]
    sort_index = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, int))].index
    prompt_data_sort4 = prompt_data_sort3.reindex(sort_index)
    enzyme_result_sort5 = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, str))]
    sort_index = enzyme_result_sort3[enzyme_result_sort3['Third'].apply(lambda x: isinstance(x, str))].index
    prompt_data_sort5 = prompt_data_sort3.reindex(sort_index)
    first_store = list(enzyme_result_sort4['First'])
    second_store = list(enzyme_result_sort4['Second'])
    third_store = list(enzyme_result_sort4['Third'])
    fir_sec_thi_store = []
    for i, n, m in zip(first_store, second_store, third_store):
        fir_sec_thi_store.append((int(i),int(n),int(m)))
    seen = set()
    fir_sec_thi_set = []
    for item in fir_sec_thi_store:
        if item not in seen:
            fir_sec_thi_set.append(item)
            seen.add(item)
    fourth_result = np.array([])
    fourth_probability = np.array([])
    for i, n, m in fir_sec_thi_set:
        if i == 1:
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
                elif m == 3:
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
    enzyme_result_fourth.to_csv(f"{sys.path[0]}/result/enzyme_result.csv", index=False)

def main():
    strgpu, batch_size = argument()
    seed=2024
    enzyme_result_sort, prompt_data_sort = first_prediction(strgpu, batch_size, seed, data_path=f'{sys.path[0]}/data/enzyme_prompt_first.txt', 
                param_path=f'{sys.path[0]}/model_param/first/ft3_MLP_BN_save.pt')
    enzyme_result_sort2, prompt_data_sort2 = second_prediction(enzyme_result_sort, prompt_data_sort, seed, 
                      strgpu, batch_size)
    enzyme_result_sort3, prompt_data_sort3 = third_prediction(enzyme_result_sort2, prompt_data_sort2, seed, 
                      strgpu, batch_size)
    fourth_prediction(enzyme_result_sort3, prompt_data_sort3, seed, 
                      strgpu, batch_size)

if __name__ == '__main__':
    main()