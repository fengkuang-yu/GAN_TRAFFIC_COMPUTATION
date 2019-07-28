
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def plot_heat_map(array):
    plt.figure(figsize=(5, 8))
    ax = plt.subplot()
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(array, ax=ax, cmap=cmap)
    plt.imshow(array, cmap='gray')
    ax.set_title('A working day traffic flow(5min)')
    ax.set_ylabel('Time lags during a working day', fontsize=12)
    ax.set_xlabel('Loop Detector Number', fontsize=12)
    ax.xaxis.set_tick_params(rotation=90, labelsize=8)
    ax.yaxis.set_tick_params(rotation=0, labelsize=8)
    plt.show()


flow_data_all = pd.read_csv(r'traffic_data/data_all.csv')
display_day = 10
image_array = flow_data_all.iloc[288*display_day:288*(display_day + 1), 1:]
image_array = image_array.values

# 归一化处理
image_min = image_array.min(axis=0)
image_max = image_array.max(axis=0)
image_array_uniform = (image_array - image_min) / (image_max - image_min)


plot_heat_map(image_array_uniform)
