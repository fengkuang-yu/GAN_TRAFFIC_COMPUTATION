#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/1 16:10
# software: PyCharm

import scipy.misc
import numpy as np
import matplotlib.pylab as plt

def save_images(images, size, path):
    # 图片归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return scipy.misc.imsave(path, merge_img)

def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("generated_images/WGAN-gp/epoch%d.png" % epoch)
    # fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


def tsne_show(data):
    from sklearn.manifold import TSNE
    data_index = np.arange(data.shape[0])
    np.random.shuffle(data_index)
    data_index = data_index[:2000]
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_vectors = tsne.fit_transform(vec_all[data_index, :])
    puzzles = np.ones((6400, 6400, 3))
    xmin = np.min(two_d_vectors[:, 0])
    xmax = np.max(two_d_vectors[:, 0])
    ymin = np.min(two_d_vectors[:, 1])
    ymax = np.max(two_d_vectors[:, 1])
    
    for i, vector in enumerate(two_d_vectors):
        x, y = two_d_vectors[i, :]
        x = int((x - xmin) / (xmax - xmin) * (6400 - 128) + 64)
        y = int((y - ymin) / (ymax - ymin) * (6400 - 128) + 64)
        puzzles[y - 64: y + 64, x - 64: x + 64, :] = img_all[data_index[i]]
    
