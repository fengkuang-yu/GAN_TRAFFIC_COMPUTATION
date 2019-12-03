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
                    filename='mf_model_result.log',
                    level=logging.ERROR)

from multiprocessing import Pool
import matplotlib.pylab as plt
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from fancyimpute import MatrixFactorization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)

class MF_imputation():
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
        
    
    def mf_imputation_model(self):
        x_test = self.X_test.reshape(-1, *self.img_shape)
        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        masked = x_test * masks
        masked_reshape = masked.reshape(-1, np.prod(masked.shape[1:]))
        x_train = self.X_train.reshape(-1, np.prod(self.X_train.shape[1:]))
        
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(masked_reshape).replace(0., np.NaN)
        
        train_data_total = pd.concat((x_train_df, x_test_df), axis=0, ignore_index=True)
        # 数据修复
        data_complete = MatrixFactorization().fit_transform(x_test_df)
        
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
    
    def train(self):
        # 构建模型
        self.attention_model = self.mf_imputation_model()


class MF():

    def __init__(self, X, k, alpha, beta, iterations):
        """
        Perform matrix factorization to predict np.nan entries in a matrix.
        Arguments
        - X (ndarray)   : sample-feature matrix
        - k (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.X = X
        self.num_samples, self.num_features = X.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # True if not nan
        self.not_nan_index = (np.isnan(self.X) == False)

    def train(self):
        # Initialize factorization matrix U and V
        self.U = np.random.normal(scale=1./self.k, size=(self.num_samples, self.k))
        self.V = np.random.normal(scale=1./self.k, size=(self.num_features, self.k))

        # Initialize the biases
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)
        self.b = np.mean(self.X[np.where(self.not_nan_index)])
        # Create a list of training samples
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            # total square error
            se = self.square_error()
            training_process.append((i, se))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, se))

        return training_process

    def square_error(self):
        """
        A function to compute the total square error
        """
        predicted = self.full_matrix()
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i, j]:
                    error += pow(self.X[i, j] - predicted[i, j], 2)
        return error

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, x in self.samples:
            # Computer prediction and error
            prediction = self.get_x(i, j)
            e = (x - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

            # Update factorization matrix U and V
            """
            If RuntimeWarning: overflow encountered in multiply,
            then turn down the learning rate alpha.
            """
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i,:])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j,:])

    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X



if __name__ == '__main__':
    logging.error('\n' + '*' * 20 + 'begin' + '*' * 20)
    mf_model = MF_imputation(missing_percentage=0.1, test_percent=0.15, miss_mode='patch')
    mf_model.train()
    
    # x_train = mf_model.X_train
    # mf = MF(x_train, k=2, alpha=0.1, beta=0.1, iterations=100)
    # mf.train()
    # X_hat = mf.full_matrix()
    # X_comp = mf.replace_nan(X_hat)
    

    # # 多进程调用
    # def basic_task(miss_percent=0.1):
    #     print('task running')
    #     for miss_mode in ['patch', 'spatial_line', 'temporal_line']:
    #         ae = MF_imputation(missing_percentage=miss_percent, test_percent=0.15, miss_mode=miss_mode)
    #
    # p = Pool(8)
    # for i in range(9):
    #     p.apply_async(basic_task, args=(0.1 * (i + 1),))
    # p.close()
    # p.join()
    # logging.error('\n' + '*' * 20 + 'end' + '*' * 20)
