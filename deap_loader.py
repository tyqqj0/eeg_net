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
    print("Loading data from", folder_path)
    print('...')

    all_data = []
    all_labels = []

    for file in os.listdir(folder_path):
        if file.endswith(".dat") and file.startswith("s"):
            file_path = os.path.join(folder_path, file)
            data_dict = load_dat_file(file_path)
            eeg_data_numpy = data_dict['data']

            # 增加一个新维度
            eeg_data_numpy = np.expand_dims(eeg_data_numpy, axis=0)
            all_data.append(eeg_data_numpy)

            # 提取 labels 并存储在 all_labels 列表中
            labels_numpy = data_dict['labels']

            # 增加一个新维度
            labels_numpy = np.expand_dims(labels_numpy, axis=0)
            all_labels.append(labels_numpy)

    # 使用 np.concatenate 合并数据和标签
    all_data_numpy = np.concatenate(all_data, axis=0)
    all_labels_numpy = np.concatenate(all_labels, axis=0)

    # 转换为 PyTorch 张量
    all_data_torch = numpy_to_tensor(all_data_numpy)
    all_labels_torch = numpy_to_tensor(all_labels_numpy)

    print('Done.')

    return all_data_torch, all_labels_torch

# folder_path = "D:/Data/deap"  # 请将此替换为实际文件夹路径
# data_tensor, labels_tensor = load_all_s_files(folder_path)
# print("Data tensor shape:", data_tensor.size())
# print("Labels tensor shape:", labels_tensor.size())
