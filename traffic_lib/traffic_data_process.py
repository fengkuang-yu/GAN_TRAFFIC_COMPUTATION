#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/1 16:10
# software: PyCharm

import os
import numpy as np

def train_test_data(param):
    """
    生成训练和测试数据
    :param param: 数据配置参数
    :return: x_train, x_test, y_train, y_test
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 96表示的是159.57号检测线圈的数据
    select_loop = [x for x in range(param.predict_loop - param.loop_num // 2,
                                    param.predict_loop + param.loop_num // 2)]
    
    def data_pro(data, time_steps=None):
        """
        数据处理，将列状的数据拓展开为行形式

        :param data: 输入交通数据
        :param time_steps: 分割时间长度
        :return: 处理过的数据
        """
        if time_steps is None:
            time_steps = 1
        size = data.shape
        data = np.array(data)
        temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(data.shape[0] - time_steps + 1):
            temp[i, :] = data[i:i + time_steps, :].flatten()
        return temp
    
    data = pd.read_csv(os.path.join(param.file_path, 'data\\' 'data_all.csv'))
    label = np.array(data.iloc[param.time_intervals + param.predict_intervals:, param.predict_loop]).reshape(-1, 1)
    data = data.iloc[:, select_loop]
    data = data_pro(data, time_steps=param.time_intervals)
    data = data[: -(1 + param.predict_intervals)]
    return train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)


def mask_randomly(images, mask_height=2, mask_width=2, ):
    """
    接收一个四维的矩阵，返回已经处理过的矩阵
    :param images:
    :param mask_height:
    :param mask_width:
    :return:
    """
    img_shape = images.shape
    img_rows = img_shape[1]
    img_width = img_shape[2]
    mask_rows_start = np.random.randint(0, img_rows - mask_height, img_shape[0])
    mask_rows_end = mask_rows_start + mask_height
    mask_cols_start = np.random.randint(0, img_width - mask_width, img_shape[0])
    mask_cols_end = mask_cols_start + mask_width
    
    masks = np.ones_like(images)
    for i, img in enumerate(images):
        _y1, _y2, _x1, _x2 = mask_rows_start[i], mask_rows_end[i], mask_cols_start[i], mask_cols_end[i],
        masks[i][_y1:_y2, _x1:_x2, :] = 0
    return masks
