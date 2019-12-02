#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/11/22 11:22
# software: PyCharm
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/10/16 15:30
# software: PyCharm

import os

import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='knn_model_result.log',
                    level=logging.ERROR)

from multiprocessing import Pool
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from fancyimpute import KNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KNN_imputation():
    def __init__(self, missing_percentage=0.8, test_percent=0.15,
                 data_path=r'traffic_data/data_all.csv', miss_mode='patch'):
        # 设置训练超参数
        self.batch_size = 100
        self.cols = 60
        self.rows = 60
        self.img_shape = (self.rows, self.cols)
        
        # 数据处理相关
        self.missing_percentage = missing_percentage
        self.test_percent = test_percent
        self.scalar = MinMaxScaler()
        self.data = self.load_data(data_path)
        self.index = np.array([[x] for x in range(len(self.data))])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)
        self.miss_mode = miss_mode
        
        # 构建模型
        self.attention_model = self.knn_imputation_model()
    
    def knn_imputation_model(self):
        x_test = self.X_test.reshape(-1, *self.img_shape)
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        masked = x_test * masks
        masked_reshape = masked.reshape(-1, np.prod(masked.shape[1:]))
        x_train = self.X_train.reshape(-1, np.prod(self.X_train.shape[1:]))
        
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(masked_reshape).replace(0., np.NaN)
        
        train_data_total = pd.concat((x_train_df, x_test_df), axis=0, ignore_index=True)
        # 数据修复
        data_complete = KNN(k=3, min_value=0., max_value=1.).fit_transform(train_data_total)
        
        x_test_imputed = data_complete[-len(masked_reshape):, :]
        real_test = self.scalar.inverse_transform(x_test.reshape(-1, np.prod(x_test.shape[1:])))
        fake_test = self.scalar.inverse_transform(x_test_imputed)
        
        # 填充部分的误差
        masks_rmse = np.sqrt(np.sum(np.power(fake_test - real_test, 2)) / np.sum(1 - masks))
        masks_mae = np.sum(np.abs(fake_test - real_test)) / np.sum(1 - masks)
        logging.error('\n miss_mode: {} '
                     '\n miss_rate: {} '
                     '\n masks_rmse: {} '
                     '\n masks_mae: {}'.format(self.miss_mode, self.missing_percentage, masks_rmse, masks_mae))
        
        return data_complete
    
    def mask_randomly(self, shape, mode='patch'):
        """
        接收一个三维以上的矩阵，返回已经处理过的矩阵
        :param images:
        :param mode:
        :param percent:
        :param mask_height:
        :param mask_width:
        :return:
        """
        assert len(shape) >= 3
        assert isinstance(shape, tuple)
        percent = self.missing_percentage if self.missing_percentage else 0.1
        masks = np.ones(shape=shape)
        if mode == 'patch':
            img_rows, img_width = shape[1], shape[2]
            mask_height = mask_width = int(np.sqrt(img_rows * img_width * percent))
            
            assert img_rows - mask_height > 0
            assert img_width - mask_width > 0
            mask_rows_start = np.random.randint(0, img_rows - mask_height, shape[0])
            mask_rows_end = mask_rows_start + mask_height
            mask_cols_start = np.random.randint(0, img_width - mask_width, shape[0])
            mask_cols_end = mask_cols_start + mask_width
            
            for i in range(shape[0]):
                _y1, _y2, _x1, _x2 = mask_rows_start[i], mask_rows_end[i], mask_cols_start[i], mask_cols_end[i],
                masks[i][_y1:_y2, _x1:_x2] = 0
            return masks
        
        elif mode == 'spatial_line':
            for i in range(shape[0]):
                index_cols = np.random.randint(0, shape[2], int(percent * shape[2]))
                masks[i][:, index_cols] = 0
            return masks
        
        elif mode == 'temporal_line':
            for i in range(shape[0]):
                index_rows = np.random.randint(0, shape[1], int(percent * shape[1]))
                masks[i][index_rows, :] = 0
            return masks
        
        elif mode == 'random':
            pass
    
    def load_data(self, dir):
        # 从excel表中选出需要的数据
        traffic_data = pd.read_csv(dir, index_col=0)
        data = traffic_data.values[:, 80: 140]  # 选择数据
        time_steps = self.cols if self.cols else 1
        
        # 数据处理
        size = data.shape
        data = np.array(data)
        reshaped_data = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(size[0] - time_steps + 1):
            reshaped_data[i, :] = data[i:i + time_steps, :].flatten()
        
        # data = self.data_pro(data, time_steps=60)
        data = self.scalar.fit_transform(reshaped_data)
        data = data.reshape(-1, *self.img_shape)
        return data
    
    def plot_test_mask(self, file_name, full=False):
        x_test = self.X_test
        idx = np.random.randint(0, 100, self.batch_size)
        x_test = x_test[idx] if not full else x_test
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        corrupted = x_test * masks
        gen_imgs, masks_gen = self.attention_model.predict([x_test, masks], batch_size=self.batch_size)
        x_flatten = x_test.reshape((-1, self.rows * self.cols))
        g_flatten = gen_imgs.reshape((-1, self.rows * self.cols))
        m_flatten = masks_gen.reshape((-1, self.rows * self.cols))
        x_flatten = self.scalar.inverse_transform(x_flatten)
        g_flatten = self.scalar.inverse_transform(g_flatten)
        m_flatten = self.scalar.inverse_transform(m_flatten)
        
        # 生成样本的误差
        total_rmse = np.sqrt(np.sum(np.power(g_flatten - x_flatten, 2)) / np.prod(x_flatten.shape))
        total_mae = np.sum(np.abs(g_flatten - x_flatten)) / np.prod(x_flatten.shape)
        
        # 填充部分的误差
        masks_rmse = np.sqrt(np.sum(np.power(m_flatten - x_flatten, 2)) / np.sum(1 - masks))
        masks_mae = np.sum(np.abs(m_flatten - x_flatten)) / np.sum(1 - masks)
        
        r, c = 2, 8
        fig, axs = plt.subplots(r * 3 + 1, c)
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[index, j].imshow(temp[j, :, :], cmap='gray')
                axs[index, j].axis('off')
        for j in range(c):
            axs[3, j].axis('off')
        for j in range(c):
            for index, temp in enumerate([x_test, corrupted, gen_imgs]):
                axs[4 + index, j].imshow(temp[c + j, :, :], cmap='gray')
                axs[4 + index, j].axis('off')
        fig.suptitle('total_rmse:{:.3f};total_mae:{:.3f}\n'
                     'masks_rmse:{:.3f};masks_mae:{:.3f}'.format(total_rmse, total_mae, masks_rmse, masks_mae))
        fig.savefig(os.path.join(os.getcwd(), 'generated_images', 'ae', file_name), dpi=300)
        plt.close()


if __name__ == '__main__':
    logging.error('\n' + '*'*20 + 'begin' + '*'*20)
    def basic_task(miss_percent=0.1):
        print('task running')
        for miss_mode in ['patch', 'spatial_line', 'temporal_line']:
            ae = KNN_imputation(missing_percentage=miss_percent, test_percent=0.15, miss_mode=miss_mode)
    
    # 多进程调用
    p = Pool(8)
    for i in range(9):
        p.apply_async(basic_task, args=(0.1 * (i + 1),))
    p.close()
    p.join()
    logging.error('\n' + '*'*20 + 'end' + '*'*20)

