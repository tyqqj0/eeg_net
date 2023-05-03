# -*- CODING: UTF-8 -*-
# @time 2023/5/1 22:59
# @Author tyqqj
# @File t_sne.py

import deap_loader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# 样本随机提取函数
# 传入参数：feature_data(n*m), labels(n), n为样本数，m为特征数, t为提取的特征数
# 返回值：feature_data(t*r), labels(t), n为样本数，m为特征数
def feature_random_extract(feature_data, labels, t=1000, r=0):
    # 计算输入样本数
    n = feature_data.shape[0]
    # 判断输入是否合法
    if n != labels.shape[0]:
        print("Error: feature_data.shape[0] != labels.shape[0]")
        return
    # 均匀随机提取t个样本
    index = np.random.randint(0, n, t)
    # 提取特征
    feature_data = feature_data[index, :]
    # 随机提取r个特征
    if r != 0:
        index = np.random.randint(0, 128, r)
        feature_data = feature_data[:, index]
    # 提取标签
    labels = labels[index]
    # 返回值
    return feature_data, labels


# t_sne函数
# 传入参数：feature_data(n*m), labels(n), n为样本数，m为特征数
# 直接显示图像
def t_sne(feature_data, labels, title="t-SNE Visualization"):
    # 判断输入是否合法
    if feature_data.shape[0] != labels.shape[0]:
        print("Error: feature_data.shape[0] != labels.shape[0]")
        return
    # 如果样本数大于1000，则随机提取1000个样本
    if feature_data.shape[0] > 1000:
        feature_data, labels = feature_random_extract(feature_data, labels, 1000)
    # 使用 t-SNE 将特征降维到 2D 空间
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    low_dim_features = tsne.fit_transform(feature_data)
    print("Done.")
    # print(low_dim_features.shape) # (1280, 2)

    # 按照label设置颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, 4))
    color_array = np.repeat(colors, 250, axis=0)
    # print(color_array.shape)

    # 绘制散点图
    scatter = plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=color_array, s=6)

    # 创建图例
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='label {}'.format(labels[i]),
                              markerfacecolor=color, markersize=6)
                       for i, color in enumerate(colors)]

    legend1 = plt.legend(handles=legend_elements, loc="best", title="Labels")
    plt.gca().add_artist(legend1)

    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.show()
    return True

#
# # 数据处理
# feature_data, labels = deap_loader.load_all_s_files("D:/Data/deap")
# # feature_data : 32 * 40 * 40 * 8064
# # labels : 32 * 40 * 4
#
# # 转换为 (32*40) * 40 * 8064，并取8064的前40个
# n = 40
# feature_data = feature_data.reshape(-1, 40, 8064)[:, :, 1000:1000 + n]
# # print(feature_data.shape)
# # 转换为 1280 * 160
# feature_data = feature_data.reshape(-1, 40 * n)
# # 转换为 (32*40) * 4
# labels = labels.reshape(-1, 4)
#
# print(feature_data.shape)
# print(labels.shape)
#
# # 使用 t-SNE 将特征降维到 2D 空间
# print("Running t-SNE...")
# tsne = TSNE(n_components=2, random_state=42)
# low_dim_features = tsne.fit_transform(feature_data)
# print("Done.")
# # print(low_dim_features.shape) # (1280, 2)
#
# # 为每个人分配一个颜色
# num_people = 32
# colors = plt.cm.rainbow(np.linspace(0, 1, num_people))
#
# # 创建颜色数组，将每个人的数据点分配给不同颜色
# color_array = np.repeat(colors, 40, axis=0)
# # print(color_array.shape)
#
# scatter = plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=color_array, s=6)
#
# step = 5  # 每隔5个人显示一个标签
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Person {}'.format(i + 1),
#                           markerfacecolor=color, markersize=6)
#                    for i, color in enumerate(colors) if (i + 1) % step == 0]
#
# # 创建图例
# legend1 = plt.legend(handles=legend_elements, loc="best", title="Person ID")
# plt.gca().add_artist(legend1)
#
# # plt.xlabel('t-SNE Component 1')
# # plt.ylabel('t-SNE Component 2')
# plt.title('t-SNE Visualization of DEAP Features')
# plt.show()
