#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/5/31 19:39
# software: PyCharm


import numpy as np
import matplotlib.pylab as plt


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

def corrupt_process_display(real, mask, corrupt, restore):
    r, c = 4, 6
    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0, i].imshow(real[i, :, :, 0], cmap='gray')
        # axs[0, i].axis('off')
        axs[1, i].imshow(mask[i, :, :, 0], cmap='gray')
        # axs[1, i].axis('off')
        axs[2, i].imshow(corrupt[i, :, :, 0], cmap='gray')
        # axs[2, i].axis('off')
        axs[3, i].imshow(restore[i, :, :, 0], cmap='gray')
        # axs[2, i].axis('off')
    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    plt.figure()
    image_size = (28, 28)
    images = np.random.random((16, *image_size, 1))
    plt.imshow(images[0].reshape(image_size), cmap='gray')
    plt.show()
    
    masks = mask_randomly(images, 5, 5)
    mask_image = masks * images
    plt.imshow(mask_image[0].reshape(28, 28), cmap='gray')
    plt.show()
    
    corrupt_process_display(images, masks, mask_image)