# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   demo.py
@Time    :   2019/4/19 16:26
@Desc    :
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

file_path = r'F:\pycharm-workspace\GAN_TRAFFIC_COMPUTATION\traffic_data\data_all.csv'
traffic_data = pd.read_csv(file_path, index_col=0)
data = traffic_data.values[:, 80: 140]  # 选择数据
data_daily = data.reshape(58, 288, 60)
daily_average = data_daily.mean(axis=0)
daily_std = np.std(data_daily, axis=0)

for temp in zip(*np.where(data_daily == 0)):
    data_daily[temp] = round(daily_average[temp[1:]])

# 处理突变异常点，使用3-sigma准则
# TODO: 将异常点剔除

data_daily = data_daily.reshape(-1, 60)

# 数据的训练和测试样本构造
time_steps = 1
size = data_daily.shape
res = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
for i in range(size[0] - time_steps + 1):
    res[i, :] = data_daily[i:i + time_steps, :].flatten()
