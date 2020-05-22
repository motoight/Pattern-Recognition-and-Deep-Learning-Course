import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

def readData(filepath):
    '''
    points.mat文件中主要有5个二维矩阵 -> [a,b,c,d,xx]
        [a,b,c,d]是四个符合二元正态分布的点集
        看形状分布，xx应该是[a,b,c,d]的组合，并且做了归一化
    为了利用gan学习形状为 'M' 的分布，我们直接返回正例 xx
    :param filepath: points.amt 文件路径
    :return:
    '''
    #
    dict = scio.loadmat(filepath)
    # print(dict)
    # x = []
    # x.append(dict['a'])
    # x.append(dict['b'])
    # x.append(dict['c'])
    # x.append(dict['d'])
    # x.append(dict['xx'])

    # x = dict['xx']
    # plt.scatter(x[:,0],x[:,1])
    # plt.show()
    return dict["xx"]




class Points(Dataset):
    def __init__(self,filepath):
        self.data = readData(filepath)
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()

    def __getitem__(self, idx):

        return self.data[idx]

    def __len__(self):
        return len(self.data)


import imageio
import matplotlib.pyplot as plt
import os
import re


def create_gif():

    def get_epoch(str):
        sm = re.search(r'(\d+)',str)
        return int(sm.group(1))

    frames = []
    root = "C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab5\\nets\\fig"
    pics = os.listdir(root)
    pics = sorted(pics, key = get_epoch)
    print(pics)
    image_list = [os.path.join(root, pic) for pic in pics]

    for img in image_list:
        if img.endswith(".png"):
            frames.append(plt.imread(img))
    imageio.mimsave("fitting_distribution.gif",frames, 'GIF', duration = 0.5)

create_gif()