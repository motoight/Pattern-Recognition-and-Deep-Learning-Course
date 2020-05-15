import os
import numpy as np
import torch
from torch.utils.data import Dataset

def read_Dict(filepath):
    '''
    读取glove文件，建立字典
    :param filepath: glovefile 路径
    :return: 字典 {word : vector}
    '''
    dic=[]
    with open(filepath,'r',encoding='utf-8') as ff:
        for line in ff.readlines():
            raw = line.split(' ')
            word = raw[0]
            vector = [float(v) for v in raw[1:]]
            dic.append([word,vector])
        dic = dict(dic)
    return dic


class Movie_Review(Dataset):
    '''
    电影评论数据集，继承自Dataset类
    为了提高读取速度，将根据字典建立的词向量文件保存为pt文件
    '''
    def __init__(self, neg_path, pos_path, glove_path, train=True):

        self.dict = None
        self.neg_data = None
        self.pos_data = None
        self.data = None
        self.target = None
        # 判断pt文件是否存在，若存在直接load即可
        if os.path.exists('neg_rev.pt') and \
            os.path.exists('pos_rev.pt'):
            self.neg_data = torch.load('neg_rev.pt')
            self.pos_data = torch.load('pos_rev.pt')
        # 如不存在，根据字典建立词向量
        else:
            self.dict = read_Dict(glove_path)
            neg_set = []
            with open(neg_path, 'r',encoding= 'windows-1252') as fn:
                lines = fn.readlines()
                for line in lines:
                    words = line.split(' ')
                    embedding = self.__embedding__(words)
                    neg_set.append(embedding)
            pos_set = []
            with open(pos_path, 'r', encoding= 'windows-1252') as fp:
                lines = fp.readlines()
                for line in lines:
                    words = line.split(' ')
                    embedding = self.__embedding__(words)
                    pos_set.append(embedding)
            self.neg_data = np.array(neg_set)
            self.pos_data = np.array(pos_set)
            torch.save(self.neg_data,open('neg_rev.pt','wb'))
            torch.save(self.pos_data,open('pos_rev.pt','wb'))
        self.neg_data = torch.Tensor(self.neg_data)
        self.pos_data = torch.Tensor(self.pos_data)
        # 根据训练集或测试集切分数据
        # neg.pt pos.pt中各取4000条数据用于训练，剩下的作为测试
        if train:
            self.data = torch.cat((self.neg_data[:4000],self.pos_data[:4000]),dim = 0)
            neg_tar = torch.zeros(4000)
            pos_tar = torch.ones(4000)
            self.target = torch.cat((neg_tar,pos_tar), dim = 0)
        else:
            self.data = torch.cat((self.neg_data[4000:], self.pos_data[4000:]),dim = 0)
            neg_tar = torch.zeros(len(self.neg_data[4000:]))
            pos_tar = torch.ones(len(self.pos_data[4000:]))
            self.target = torch.cat((neg_tar, pos_tar), dim=0)

        print(self.data.shape)
        print(self.target.shape)
        # self.data = torch.Tensor(self.data)
        # self.target = torch.Tensor(self.target)


    def __embedding__(self, words):
        '''
        返回单词列表对应的词向量列表
        :param words: 单词列表
        :return:
        '''
        vectors = []
        # 影评数据是变长序列，我们只使用长度为51的序列，后续截断
        zero_list = [0. for i in range(50)]
        cnt = 0
        for word in words:
            if cnt>=50:
                break
            if word in self.dict.keys():
                vectors.append(self.dict[word])
            # 此处有几种策略：
                # 不在字典中的单词用0填充
                # 跳过不在字典中的单词
                # 用特殊的向量表示UKW [Unknown word]
            # 此处选择跳过，是为了保证有效信息不会因为无效的0而损失
            # else:
            #     vectors.append(zero_list)
                cnt+=1
        # 在有效序列之前用zerolist补齐
        # 从前部填充的好处在于能够让有效的信息靠近输出，而不会使得lstm有效信息被冲掉
        for i in range(50 - cnt):
            vectors.insert(0, zero_list)
        vectors = np.array(vectors).astype('float64')
        print(vectors.shape)
        return vectors


    def __getitem__(self, index):
        return self.data[index],self.target[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    negpath= 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\movie_review\\rt-polarity.neg'
    pospath = 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\movie_review\\rt-polarity.pos'
    glovepath= 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\glove.6B.50d.txt'

    train_set = Movie_Review(negpath,pospath,glovepath,train=True)
    test_set = Movie_Review(negpath,pospath,glovepath,train=False)
