#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/11/6 9:26
# software: PyCharm

# -*- coding: utf-8 -*-

"""
本程序的修改点主要在以下：
1. 使用缺失部分计算误差而不是使用全部的图像重构误差
2. 对缺失部分使用负数标记如-1
3. 将输入理解为两个时间序列，第一个是检测器之间的注意力，第二个是时间上的注意力
4. 减少判别器的判别能力
5. 添加时间和空间的两种交通表示
6. 添加重构误差为l1损失
7. 添加u-net结构

"""

import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.layers import Add, Flatten, Dense, Activation, Conv2D
from keras.layers import Input, LeakyReLU, Lambda, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
# from keras_multi_head import MultiHeadAttention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import FeedForward, MultiHeadAttention
from utils import LayerNormalization
from utils import PositionEmbedding

from utils import spectral_normalization

# spectral_normalization = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU

# 适用于tensorflow 1.+版本
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
K.set_session(session)

# 适用于tensorflow 2.+版本
# tf.config.gpu.set_per_process_memory_growth(enabled=True)


class AttentionGAN():
    def __init__(self, train_data_path):
        # 设置超参数
        self.batch_size = 32
        self.spatial_dim = 85
        self.time_dim = 60
        self.img_shape = (self.time_dim, self.spatial_dim)
        self.latent_dim = 100
        self.multi_head_num = 5
        self.ff_hidden_unit_num = 1024
        self.epochs = 20000
        self.generator_optimizer = Adam(lr=1e-3, beta_1=0., beta_2=0.9)
        self.critic_optimizer = Adam(lr=4e-3, beta_1=0., beta_2=0.9)
        self.critic_loss_list, self.generator_loss_list, self.generator_valid = [], [], []

        # 优化的超参数
        self.reconstruct_weight = 1  # 重构误差在生成器中的占比
        self.power_iter_num = 1

        # 数据处理相关
        self.missing_percentage = 0.3  # 缺失面积百分比
        self.test_percent = 0.15  # 测试集所占比例
        self.scalar = MinMaxScaler()  # 归一化的类
        self.data = self.load_data(train_data_path, time_dim=60)
        self.index = np.array([[x] for x in range(len(self.data))])  # 假象的标签用来表示当前取出的样本在原始文件位置
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.index, test_size=self.test_percent, random_state=12)  # 切分训练集和测试集
        self.miss_mode = 'patch'  # 缺失类型共计四种：[spatial_line, temporal_line, patch, random]

        # 构建模型
        self.generator = self.build_generator()  # basic生成器
        self.critic = self.build_critic()  # basic判别器

        self.generator_trainer = self.build_generator_trainer()
        self.critic_trainer = self.build_critic_trainer()

    def attention_block(self, input_x, head_num, feed_forward_hidden_units, kernel_constraint=None):
        """
        multi-head attention block
        :param input_x:
        :return:
        """
        attention_multi_head = MultiHeadAttention(head_num=head_num, kernel_constraint=kernel_constraint)(input_x)
        add_attention = Add()([input_x, attention_multi_head])
        layer_normal = LayerNormalization(gamma_constraint=kernel_constraint)(add_attention)
        feed_forward = FeedForward(units=feed_forward_hidden_units, kernel_constraint=kernel_constraint)(layer_normal)
        feed_forward_res_1 = Add()([feed_forward, layer_normal])
        return feed_forward_res_1

    def positional_encoding(self, x:np.ndarray, start_index, mode='sum', min_timescale=1.0, max_timescale=1.0e4):

        batch_size, length, channels = x.shape
        position = np.cumsum(np.ones_like(x[:, :, 0], dtype=float), 1) - 1
        position_i = np.expand_dims(position, 2)
        num_timescales = channels // 2
        log_timescale_increment = (np.log(max_timescale / min_timescale) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * np.exp(np.arange(num_timescales, dtype=float) * -log_timescale_increment)
        position_j = np.expand_dims(inv_timescales, 0)
        position_ij = np.dot(position_i, position_j)
        position_ij = np.concatenate([np.sin(position_ij), np.cos(position_ij)], 2)
        position_ij = np.pad(position_ij, [[0, 0], [0, 0], [0, channels % 2]], 'constant')

        assert len(start_index) == len(position_ij)

        for matrix_index, temp_index in enumerate(start_index):
            position_ij[matrix_index] = np.roll(position_ij[matrix_index], shift=-temp_index[0], axis=0)
        if mode == 'sum':
            return position_ij + x
        elif mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        elif mode == 'single':
            return position_ij

    def build_generator(self):

        input_x = Input(shape=self.img_shape, name='real_input')
        masks_x = Input(shape=self.img_shape, name='masks')
        position = Input(shape=self.img_shape, name='position')

        # 缺失部分按-1.处理
        input_masked_x_minus = Lambda(function=lambda x: x[0] * x[1] + -1. * (1 - x[1]) + x[2],
                                      output_shape=self.img_shape,
                                      name='masked_input')([input_x, masks_x, position])

        # 以不同的空间点的交通流量信息作为输入
        feed_forward_res_1 = self.attention_block(input_masked_x_minus,
                                                  kernel_constraint=spectral_normalization,
                                                  head_num=self.multi_head_num,
                                                  feed_forward_hidden_units=self.ff_hidden_unit_num)

        # 以不同的时间点的交通流量作为输入
        input_masked_trans = Lambda(function=lambda x: K.permute_dimensions(x, [0, 2, 1]),
                                    output_shape=self.img_shape[::-1],
                                    name='transpose_layer_1')(input_masked_x_minus)  # 将输入旋转了一下
        add_position_trans = Add()([input_masked_trans, input_masked_trans])
        feed_forward_res_trans = self.attention_block(add_position_trans,
                                                      kernel_constraint=spectral_normalization,
                                                      head_num=self.multi_head_num,
                                                      feed_forward_hidden_units=self.ff_hidden_unit_num)
        feed_forward_res_trans_1 = Lambda(function=lambda x: K.permute_dimensions(x, [0, 2, 1]),
                                          output_shape=self.img_shape,
                                          name='transpose_layer_2')(feed_forward_res_trans)  # 将输入转置过来

        # 将两个特征图加起来作为后续的网络输入
        feed_forward_res_hybrid = Add()([feed_forward_res_1, feed_forward_res_trans_1])
        feed_forward_res_3 = self.attention_block(feed_forward_res_hybrid,
                                                  kernel_constraint=spectral_normalization,
                                                  head_num=self.multi_head_num,
                                                  feed_forward_hidden_units=self.ff_hidden_unit_num)

        fake_res = Activation('sigmoid')(feed_forward_res_3)

        imputed_img = Lambda(function=lambda x: x[0] * x[1] + (1 - x[0]) * x[2],
                             output_shape=self.img_shape,
                             name='imputation_layer')([masks_x, input_x, fake_res])

        attention_model = Model([input_x, masks_x, position], [fake_res, imputed_img], name='basic_generator')
        return attention_model

    def build_critic(self):

        input_x = Input(shape=self.img_shape)
        pos_encoding_1 = PositionEmbedding()(input_x)
        add_position_1 = Add()([input_x, pos_encoding_1])

        feed_forward_res_1 = self.attention_block(add_position_1,
                                                  kernel_constraint=spectral_normalization,
                                                  head_num=self.multi_head_num,
                                                  feed_forward_hidden_units=self.ff_hidden_unit_num)

        flatten_1 = Flatten()(feed_forward_res_1)
        dense_1 = Dense(self.latent_dim, kernel_constraint=spectral_normalization)(flatten_1)
        # relu_1 = LeakyReLU()(dense_1)
        # output = Dense(1, kernel_constraint=spectral_normalization, activation='tanh')(relu_1)  # 添加了激活函数
        output = Dense(1, kernel_constraint=spectral_normalization)(dense_1)  # 添加了激活函数

        basic_critic = Model(input_x, [output, feed_forward_res_1], name='basic_discriminator')
        return basic_critic

    def build_critic_trainer(self):
        ### build discriminator ###
        self.generator.trainable = False
        self.critic.trainable = True
        x_real = Input(shape=self.img_shape, name='real_image')
        masks = Input(shape=self.img_shape, name='mask_layer')
        position = Input(shape=self.img_shape, name='position')

        # index_time = Input(shape=(1,), name='time_stamp')
        x_fake, x_imputed = self.generator([x_real, masks, position])  # todo: 修复后的图片作为损失计算目标

        rec_valid, rec_d_llike = self.critic(x_imputed)
        real_valid, real_d_llike = self.critic(x_real)

        critic_loss = Lambda(self.critic_loss, output_shape=(1,), name='critic_loss')([rec_valid, real_valid])

        critic_trainer = Model([x_real, masks, position], [critic_loss], name='discriminator')
        critic_trainer.add_loss(critic_loss)
        critic_trainer.compile(optimizer=self.critic_optimizer)
        critic_trainer.summary()
        return critic_trainer

    def build_generator_trainer(self):
        ### build generator ###
        self.generator.trainable = True
        self.critic.trainable = False
        real_img = Input(shape=self.img_shape, name='real_image')
        masks_generator = Input(shape=self.img_shape, name='masks')
        position = Input(shape=self.img_shape, name='position')

        # time_stamps = Input(shape=(1,), name='time_stamps')

        fake_img, imputed_img = self.generator([real_img, masks_generator, position])  # 返回两个值，第一个是纯生成的图，第二个是修复图
        real_valid_critic, real_hidden_critic = self.critic(real_img)
        fake_valid_critic, fake_hidden_critic = self.critic(imputed_img)

        gen_loss = Lambda(self.generative_loss, output_shape=(1,),
                          name='generative_loss')([real_img, fake_img, real_hidden_critic, fake_hidden_critic])

        generator_trainer = Model([real_img, masks_generator, position], [gen_loss, fake_valid_critic],
                                  name='generator_trainer')
        generator_trainer.add_loss(gen_loss)
        generator_trainer.compile(optimizer=self.generator_optimizer)
        generator_trainer.summary()
        return generator_trainer

    def critic_loss(self, args):
        """
        使用spectral norm 文章内的hinge损失
        :param args:
        :return:
        """
        rec_valid, real_valid = args

        rec_loss = K.mean(K.minimum(0., -1. + real_valid))
        real_loss = K.mean(K.minimum(0., -1. - rec_valid))
        #
        # rec_loss = K.mean(K.abs(-1. - rec_valid))
        # real_loss = K.mean(K.abs(1. - real_valid))

        return rec_loss + real_loss

    def generative_loss(self, args):
        x_real, x_rec, real_d_llike, rec_d_llike = args
        reconstruct_loss = K.mean(K.square(x_real - x_rec))
        critic_hidden_loss = K.mean(K.abs(real_d_llike - rec_d_llike))
        return self.reconstruct_weight * reconstruct_loss + critic_hidden_loss

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

    def plot_test_mask(self, file_name, full=False):
        x_test = self.X_test
        time_stamp = self.y_test
        idx = np.random.randint(0, 100, self.batch_size)
        x_test = x_test[idx] if not full else x_test  # 如果full有值的话，计算全部样本的误差
        time_stamp = time_stamp[idx] if not full else time_stamp

        masks = self.mask_randomly(x_test.shape, mode=self.miss_mode)
        corrupted = x_test * masks
        gen_img, imputed_img = self.generator.predict([x_test, masks, x_test], batch_size=self.batch_size)

        x_test = self.scalar.inverse_transform(x_test.reshape((-1, self.spatial_dim)))
        gen_img = self.scalar.inverse_transform(gen_img.reshape((-1, self.spatial_dim)))
        imputed_img = self.scalar.inverse_transform(imputed_img.reshape((-1, self.spatial_dim)))

        x_real_img = x_test.reshape((-1, *self.img_shape))
        g_fake_img = gen_img.reshape((-1, *self.img_shape))
        m_fake_img = imputed_img.reshape((-1, *self.img_shape))

        # 生成样本的误差
        total_rmse = np.sqrt(np.sum(np.power(g_fake_img - x_real_img, 2)) / np.prod(x_real_img.shape))
        total_mae = np.sum(np.abs(g_fake_img - x_real_img)) / np.prod(x_real_img.shape)

        # 填充部分的误差
        masks_rmse = np.sqrt(np.sum(np.power(m_fake_img - x_real_img, 2)) / np.sum(1 - masks))
        masks_mae = np.sum(np.abs(m_fake_img - x_real_img)) / np.sum(1 - masks)

        r, c = 2, 8
        fig, axs = plt.subplots(r * 3 + 1, c)
        for j in range(c):
            for index, temp in enumerate([x_real_img, corrupted, m_fake_img]):
                axs[index, j].imshow(temp[j, :, :], cmap='gray')
                axs[index, j].axis('off')
        for j in range(c):
            axs[3, j].axis('off')
        for j in range(c):
            for index, temp in enumerate([x_real_img, corrupted, m_fake_img]):
                axs[4 + index, j].imshow(temp[c + j, :, :], cmap='gray')
                axs[4 + index, j].axis('off')
        fig.suptitle('total_rmse:{:.3f};total_mae:{:.3f}\n'
                     'masks_rmse:{:.3f};masks_mae:{:.3f}'.format(total_rmse, total_mae, masks_rmse, masks_mae))
        fig.savefig(os.path.join(os.getcwd(), 'generated_images', 'gan_backprop_with_imputed', file_name), dpi=300)
        plt.close()

    def load_data(self, dir, time_dim=60, selected_cols=None):
        """
        从csv文件中提取序列数据
        :param dir: csv文件的位置
        :param selected_cols: [start:end)选中的列
        :return:
        """
        # 从excel表中选出需要的数据
        traffic_data = pd.read_csv(dir, index_col=0)
        traffic_data = traffic_data.fillna(0)
        if selected_cols is None:
            selected_data = traffic_data.values
        else:
            selected_data = traffic_data.values[:, selected_cols[0]: selected_cols[1]]  # 选择数据
        time_steps = time_dim
        size = selected_data.shape

        # 数据处理
        selected_data = self.scalar.fit_transform(selected_data)
        reshaped_data = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(size[0] - time_steps + 1):
            reshaped_data[i, :] = selected_data[i:i + time_steps, :].flatten()

        data = reshaped_data.reshape((-1, time_dim, size[1]))
        return data

    def train(self, epochs=None):
        x_train = self.X_train
        train_index = self.y_train
        total_epoch = self.epochs if epochs == None else epochs

        for epoch in tqdm(range(total_epoch)):
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            real_images = x_train[idx].reshape((-1, *self.img_shape))
            masks = self.mask_randomly(real_images.shape, mode=self.miss_mode)
            real_index = train_index[idx]
            pos_enc = self.positional_encoding(real_images, start_index=real_index, mode='single')
            # 判别器训练
            critic_loss = self.critic_trainer.train_on_batch([real_images, masks, pos_enc], None)
            # 生成器训练
            generator_loss = self.generator_trainer.train_on_batch([real_images, masks, pos_enc], None)

            if epoch % 10 == 0:
                self.critic_loss_list.append((epoch, critic_loss))
                self.generator_loss_list.append((epoch, generator_loss))
                # self.generator_valid.append((epoch, np.mean(gen_valid)))

            if epoch % 5000 == 0:
                ae_gan.plot_test_mask(ae_gan.miss_mode + 'u_net_GAN_new_{:.1%}_{:0>5d}epochs.png'.format(
                    ae_gan.missing_percentage, epoch), full=True)
            #     self.plot_test_mask(
            #         self.miss_mode + '{:.1%}_{:0>5d}epochs_gen.png'.format(self.missing_percentage, epoch))
            # print('\ngenerator_loss:{};\ncritic_loss:{}'.format(generator_loss, critic_loss))

    def plot_loss(self):
        critic_loss = self.critic_loss_list
        generator_loss = self.generator_loss_list
        generator_valid = self.generator_valid

        #
        # loss = pd.DataFrame(data=[[x[1], y[1], z[1]] for x, y, z in zip(critic_loss, generator_loss, generator_valid)],
        #                     index=[x[0] for x in critic_loss],
        #                     columns=['critic_loss', 'generator_loss', 'generator_valid'], )
        loss = pd.DataFrame(data=[[x[1], y[1]] for x, y in zip(critic_loss, generator_loss)],
                            index=[x[0] for x in critic_loss],
                            columns=['critic_loss', 'generator_loss'], )

        plt.figure(figsize=(6, 10), dpi=100)
        ax1 = plt.subplot(211)
        plt.plot(loss.critic_loss, label='critic_loss', color='r')
        plt.ylabel('critic_loss')
        ax2 = plt.subplot(212)
        plt.plot(loss.generator_loss, label='generator_loss', color='b')
        plt.ylabel('generator_loss')
        # ax3 = plt.subplot(313)
        # plt.plot(loss.generator_valid, label='generator_loss', color='g')
        # plt.ylabel('generator_valid')
        # plt.savefig(r'C:\Users\lenovo\Desktop\demo.png')
        plt.savefig(os.path.join(os.getcwd(), 'training_related_img', 'gan_loss_new.png'), dpi=300)
        plt.close()

    def plot_net_struct(self):
        plot_model(self.generator_trainer,
                   to_file=os.path.join(os.getcwd(), 'network_related_img', 'generator_trainer_new.pdf'))
        plot_model(self.critic_trainer,
                   to_file=os.path.join(os.getcwd(), 'network_related_img', 'critic_trainer_new.pdf'))
        plot_model(self.critic, to_file=os.path.join(os.getcwd(), 'network_related_img', 'basic_critic.pdf'))
        plot_model(self.generator, to_file=os.path.join(os.getcwd(), 'network_related_img', 'basic_generator.pdf'))


if __name__ == '__main__':
    iterations = int(1e5)
    ae_gan = AttentionGAN(train_data_path=r'traffic_data/filtered_data_all_005.csv')
    # ae_gan.plot_net_struct()
    ae_gan.train(iterations)
    # ae_gan.plot_test_mask(ae_gan.miss_mode + 'GAN_new_{:.1%}_{:0>5d}epochs_gen.png'.format(
    #     ae_gan.missing_percentage, iterations), full=True)  # 给出填补的修复案例
    # ae_gan.plot_loss()  # 画出损失函数图
