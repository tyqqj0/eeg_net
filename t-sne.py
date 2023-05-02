# -*- CODING: UTF-8 -*-
# @time 2023/5/1 22:59
# @Author tyqqj
# @File t-sne.py


import pickle
import torch


# 用这个函数读取 .dat 文件
def load_dat_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


# 用这个函数将 numpy 数组转换为 PyTorch 张量
def numpy_to_tensor(numpy_array):
    return torch.from_numpy(numpy_array)


# 读取.dat文件，并将其转换为PyTorch张量
file_path = "D:/Data/deap/s01.dat"  # 请将此替换为实际文件路径
data_dict = load_dat_file(file_path)

# 示例：从数据字典中提取 EEG 数据并转换为 PyTorch 张量
eeg_data_numpy = data_dict['data']
eeg_data_torch = numpy_to_tensor(eeg_data_numpy)

# 输出结果
print(eeg_data_torch)
