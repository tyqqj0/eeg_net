# -*- CODING: UTF-8 -*-
# @time 2023/5/1 22:59
# @Author tyqqj
# @File t-sne.py

import deap_loader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 数据处理
feature_data, labels = deap_loader.load_all_s_files("D:/Data/deap")
# feature_data : 32 * 40 * 40 * 8064
# labels : 32 * 40 * 4

# 转换为 (32*40) * 40 * 8064，并取8064的前40个
n = 40
feature_data = feature_data.reshape(-1, 40, 8064)[:, :, 1000:1000 + n]
# print(feature_data.shape)
# 转换为 1280 * 160
feature_data = feature_data.reshape(-1, 40 * n)
# 转换为 (32*40) * 4
labels = labels.reshape(-1, 4)

print(feature_data.shape)
print(labels.shape)

# 使用 t-SNE 将特征降维到 2D 空间
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
low_dim_features = tsne.fit_transform(feature_data)
print("Done.")
# print(low_dim_features.shape) # (1280, 2)

# 为每个人分配一个颜色
num_people = 32
colors = plt.cm.rainbow(np.linspace(0, 1, num_people))

# 创建颜色数组，将每个人的数据点分配给不同颜色
color_array = np.repeat(colors, 40, axis=0)
# print(color_array.shape)

scatter = plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=color_array, s=6)
from matplotlib.lines import Line2D

step = 5  # 每隔5个人显示一个标签
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Person {}'.format(i + 1),
                          markerfacecolor=color, markersize=6)
                   for i, color in enumerate(colors) if (i + 1) % step == 0]

# 创建图例
legend1 = plt.legend(handles=legend_elements, loc="best", title="Person ID")
plt.gca().add_artist(legend1)

# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of DEAP Features')
plt.show()
