# -*- CODING: UTF-8 -*-
# @time 2023/5/2 23:34
# @Author tyqqj
# @File deap_loader.py


import os
import pickle
import torch
import numpy as np


def load_dat_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def numpy_to_tensor(numpy_array):
    return torch.from_numpy(numpy_array)


def load_all_s_files(folder_path):
    print('folder_path: {}'.format(folder_path))
    print('loading all s files...')
    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".dat") and file.startswith("s"):
            file_path = os.path.join(folder_path, file)
            data_dict = load_dat_file(file_path)
            eeg_data_numpy = data_dict['data']
            eeg_data_numpy = np.expand_dims(eeg_data_numpy, axis=0)
            # print('eeg_data_numpy.shape: {}'.format(eeg_data_numpy.shape))
            all_data.append(eeg_data_numpy)

    all_data_numpy = np.concatenate(all_data, axis=0)
    all_data_torch = numpy_to_tensor(all_data_numpy)
    return all_data_torch


folder_path = "D:/Data/deap"
result_tensor = load_all_s_files(folder_path)
print(result_tensor.shape)
