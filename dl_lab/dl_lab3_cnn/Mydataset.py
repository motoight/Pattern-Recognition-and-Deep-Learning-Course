import pickle
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os


class Cifar10(data.Dataset):
    '''
    CIFAR 10 数据集
    参考 torchvision中Dataset的写法
    '''

    base_folder = "cifar-10-batches-py"
    train_list = ["data_batch_1",
                  "data_batch_2",
                  "data_batch_3",
                  "data_batch_4",
                  "data_batch_5"]

    test_list = ["test_batch"]

    meta = {
        "filename": "batches.meta",
        "key": 'label_names'
    }

    def __init__(self, root, train=True, transform=None, target_transform=None):
        '''
        :param root: Cifar数据的根目录
        :param train: 是否是训练集
        :param transform: 对原数据的操作
        :param target_transform: 对target标签的操作，一般较少用到，可能GAN中需要
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        for filename in file_list:
            file_path = os.path.join(root,self.base_folder,filename)
            with open(file_path,'rb') as ff:
                # entry = pickle.load(ff,encoding='bytes')
                entry = pickle.load(ff,encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        # from list to numpy array
        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        # convert to [nsamples, height, width, channels]
        # self.data = self.data.transpose((0,1,2,3))
        self.data = self.data.transpose([0,2,3,1])
        # load meta data
        self._load_meta()

    def _load_meta(self):
        '''
        加载原数据，包含图片的类别序号和标签名称
        :return: None
        '''
        path = os.path.join(self.root,self.base_folder,self.meta['filename'])
        with open(path,'rb') as ff:
            data = pickle.load(ff,encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def __getitem__(self, index):
        '''
        :param index: 图片的索引值
        :return: 返回图片和对应的标签
        '''

        img, target = self.data[index],self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class ToTensor(object):
    '''
    将数据转换成torch.tensor

    '''
    def __call__(self,pic):
        return torch.from_numpy(pic).float()









