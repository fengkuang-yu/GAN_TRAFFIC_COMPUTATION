#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/12/2 16:28
# software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def bar_mae_baselines():
    # MAE
    HA_res = list(map(float, '28.29	28.33	28.58	28.28	28.21	27.88	27.66	27.46'.split()))
    KNN_res = list(map(float, '19.64	19.35	19.25	19.28	19.33	19.25	19.32	19.39'.split()))
    DAE_res = list(map(float, '22.03	21.89	21.89	21.78	21.68	22.36	22.11	22.6'.split()))
    M_RNN_res = list(map(float, '14.92	15.42	15.59	15.44	15.42	15.37	15.37	15.29'.split()))
    SAA_AE_res = list(map(float, '12.82	12.27	12.24	11.95	11.21	10.71	9.91	10.29'.split()))
    
    # plt.style.use('seaborn')
    n_groups = 8
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    index = np.arange(n_groups)
    index = index + 0.125
    bar_width = 0.3
    opacity = 0.9
    
    plt.bar(index, HA_res, bar_width / 2, alpha=opacity, label=u'历史平均模型')
    plt.bar(index + 0.5 * bar_width, KNN_res, bar_width / 2, alpha=opacity, label=u'K-最近邻模型')
    plt.bar(index + 1.0 * bar_width, DAE_res, bar_width / 2, alpha=opacity, label=u'深度自编码模型')
    plt.bar(index + 1.5 * bar_width, M_RNN_res, bar_width / 2, alpha=opacity, label=u'双向循环神经网络模型')
    plt.bar(index + 2.0 * bar_width, SAA_AE_res, bar_width / 2, alpha=opacity, label=u'本章所提出的模型')
    
    plt.xlabel(u'数据的缺失率（%）', fontsize=18, color='k')
    plt.ylabel(u'平均绝对误差 (MAE)', fontsize=18, color='k')
    plt.xticks(index - 0.3 + 2 * bar_width, [str(x) for x in range(10, 90, 10)], fontsize=16, color='k')
    plt.yticks(fontsize=14, color='k')  # change the num axis size
    plt.ylim(0, 35)  # The ceil
    plt.legend(ncol=3, loc=2, mode='expand', fontsize=18, handlelength=1, handletextpad=0.2)
    plt.tight_layout()
    plt.show()


def bar_rmse_baselines():
    # MAE
    HA_res = list(map(float, '40.95	40.89	41.35	40.86	40.59	40.01	39.59	39.46'.split()))
    KNN_res = list(map(float, '31.01	29.18	28.64	28.5	28.46	28.13	28.27	28.39'.split()))
    DAE_res = list(map(float, '32.22	32.26	31.8	31.66	31.73	32.43	32.24	33.1'.split()))
    M_RNN_res = list(map(float, '21.19	21.75	21.81	21.69	21.62	21.53	21.45	21.3'.split()))
    SAA_AE_res = list(map(float, '17.17	16.37	16.41	16.01	14.97	14.28	13.16	13.71'.split()))
    
    # plt.style.use('seaborn')
    n_groups = 8
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    index = np.arange(n_groups)
    index = index + 0.125
    bar_width = 0.3
    opacity = 0.9
    
    plt.bar(index, HA_res, bar_width / 2, alpha=opacity, label=u'历史平均模型')
    plt.bar(index + 0.5 * bar_width, KNN_res, bar_width / 2, alpha=opacity, label=u'K-最近邻模型')
    plt.bar(index + 1.0 * bar_width, DAE_res, bar_width / 2, alpha=opacity, label=u'深度自编码模型')
    plt.bar(index + 1.5 * bar_width, M_RNN_res, bar_width / 2, alpha=opacity, label=u'双向循环神经网络模型')
    plt.bar(index + 2.0 * bar_width, SAA_AE_res, bar_width / 2, alpha=opacity, label=u'本章所提出的模型')
    
    plt.xlabel(u'数据的缺失率（%）', fontsize=18, color='k')
    plt.ylabel(u'均方根误差 (RMAE)', fontsize=18, color='k')
    plt.xticks(index - 0.3 + 2 * bar_width, [str(x) for x in range(10, 90, 10)], fontsize=16, color='k')
    plt.yticks(fontsize=14, color='k')  # change the num axis size
    plt.ylim(0, 50)  # The ceil
    plt.legend(ncol=3, loc=2, mode='expand', fontsize=18, handlelength=1, handletextpad=0.2)
    plt.tight_layout()
    plt.show()


def plot_heat_map():
    flow_data_all = pd.read_csv(r'traffic_data/data_all.csv')
    display_day = 10
    image_array = flow_data_all.iloc[288 * display_day:288 * (display_day + 1), 80:140]
    image_array = image_array.values
    
    # 归一化处理
    image_min = image_array.min(axis=0)
    image_max = image_array.max(axis=0)
    image_array_uniform = (image_array - image_min) / (image_max - image_min)
    plt.figure(figsize=(6, 8), dpi=300)
    ax = plt.subplot()
    cmap = sns.cubehelix_palette(start=0., rot=3, gamma=1, as_cmap=True)
    sns.heatmap(image_array_uniform, ax=ax, cmap=cmap)
    plt.imshow(image_array_uniform, cmap='gray')
    ax.set_ylabel(u'时间戳（5分钟）', fontsize=16)
    ax.set_xlabel(u'检测线圈编号', fontsize=16, )
    ax.xaxis.set_tick_params(rotation=90, labelsize=10)
    ax.yaxis.set_tick_params(rotation=0, labelsize=10)
    plt.show()
