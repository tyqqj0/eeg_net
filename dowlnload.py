# -*- CODING: UTF-8 -*-
# @time 2023/5/3 20:57
# @Author tyqqj
# @File dowlnload.py

import os
import subprocess
import mne

# 更改此变量以指向您的 aria2c 可执行文件的路径
aria2c_path = "S:/program/aria2/aria2c.exe"

url = "https://storage.googleapis.com/inria-unicog-public/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif"

# 将此变量更改为您希望存储数据集的目录
your_custom_directory = "D:/Data/mne/"

output_dir = os.path.join(your_custom_directory, "MNE-sample-data")
output_path = os.path.join(output_dir, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")

# 创建目录
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 使用 aria2c 下载文件
subprocess.run([aria2c_path, "-x", "8", "-s", "8", "-d", os.path.dirname(output_path), url])
