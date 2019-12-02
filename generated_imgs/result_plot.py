#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/12/2 16:28
# software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题



def bar_mae_baselines():
    # MAE
    HA_res = list(map(float, '28.29	28.33	28.58	28.28	28.21	27.88	27.66	27.46'.split()))
    KNN_res = list(map(float, '19.64	19.35	19.25	19.28	19.33	19.25	19.32	19.39'.split()))
    DAE_res = list(map(float, '20.56	20.73	20.67	20.81	20.88	20.91	21.05	21.57'.split()))
    M_RNN_res = list(map(float, '16.38	16.76	16.39	16.06	15.84	15.76	15.59	15.41'.split()))
    SAA_AE_res = list(map(float, '16.79	15.08	14.29	13.41	12.83	12.51	11.18	11.45'.split()))
    
    # plt.style.use('seaborn')
    n_groups = 8
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    index = np.arange(n_groups)
    index = index + 0.125
    bar_width = 0.3
    opacity = 0.9

    plt.bar(index, HA_res, bar_width/2, alpha=opacity, label=u'历史平均模型')
    plt.bar(index + 0.5 * bar_width, KNN_res, bar_width/2, alpha=opacity, label=u'K-最近邻模型')
    plt.bar(index + 1.0 * bar_width, DAE_res, bar_width/2, alpha=opacity, label=u'深度自编码模型')
    plt.bar(index + 1.5 * bar_width, M_RNN_res, bar_width/2, alpha=opacity, label=u'双向循环神经网络模型')
    plt.bar(index + 2.0 * bar_width, SAA_AE_res, bar_width/2, alpha=opacity, label=u'本章所提出的模型')

    plt.xlabel(u'数据的缺失率（%）', fontsize=18, color='k')
    plt.ylabel(u'平均绝对误差 (MAE)', fontsize=18, color='k')
    plt.xticks(index-0.3 + 2 * bar_width, [str(x) for x in range(10, 90, 10)], fontsize=14, color='k')
    plt.yticks(fontsize=14, color='k')  # change the num axis size
    plt.ylim(0, 35)  # The ceil
    plt.legend(ncol=3, loc=2, mode='expand', fontsize=16, handlelength=1, handletextpad=0.2)
    plt.tight_layout()
    plt.show()


def bar_rmse_baselines():
    # MAE
    HA_res = list(map(float, '40.95	40.89	41.35	40.86	40.59	40.01	39.59	39.46'.split()))
    KNN_res = list(map(float, '31.01	29.18	28.64	28.5	28.46	28.13	28.27	28.39'.split()))
    DAE_res = list(map(float, '29.52	29.94	30.01	29.99	30.27	30.32	30.73	31.41'.split()))
    M_RNN_res = list(map(float, '23.61	24.06	23.19	22.75	22.35	22.1	21.92	21.63'.split()))
    SAA_AE_res = list(map(float, '24.16	20.9	19.75	18.52	17.47	17.75	15.02	15.6'.split()))
    
    # plt.style.use('seaborn')
    n_groups = 8
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.xticks(index - 0.3 + 2 * bar_width, [str(x) for x in range(10, 90, 10)], fontsize=14, color='k')
    plt.yticks(fontsize=14, color='k')  # change the num axis size
    plt.ylim(0, 50)  # The ceil
    plt.legend(ncol=3, loc=2, mode='expand', fontsize=16, handlelength=1, handletextpad=0.2)
    plt.tight_layout()
    plt.show()

