# MLE for MNIST

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch import utils
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter

from matplotlib import pyplot as plt
from time import  time
import os


writer = SummaryWriter("runs/exp1-withoutsf-bn",comment=time())

class MLP(nn.Module):
    def __init__(self,input,hidden1,hidden2,output):
        super(MLP,self).__init__()
        self.in_dim = input
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_dim = output
        self.classifier = nn.Sequential(
            nn.Linear(input,hidden1),
            nn.BatchNorm1d(hidden1,momentum=0.9),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden1,hidden2),
            nn.BatchNorm1d(hidden2,momentum=0.1),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden2,output),
            # nn.Softmax() # 最后不需要加softmax,因为交叉熵中自带的对输入的softmax处理
        )

    def forward(self, din):
        din = din.view(-1, self.in_dim)
        return self.classifier(din)



def train(model,train_loader,optimizer,loss_func):
    '''
    :param model: 训练的模型
    :param train_loader: 训练数据
    :param optimizer: 优化器
    :return: training loss
    '''
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        optimizer.zero_grad()  # 清空上一次计算的梯度
        loss = loss_func(output, target)
        loss.backward()  # 误差反向传播
        optimizer.step()  # update params
        # print("data size :{}".format(len(data)))
        train_loss += loss.item() * len(data)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss

def validate(model, test_loader,loss_func):
    '''
    :param model: 验证的模型
    :param test_loader: 测试数据
    :return: test loss and test Accuracy
    '''
    test_loss = 0.0
    correct = 0
    total = 0
    # 验证的过程不需要求导，可以节省显存，提高运算速度
    with torch.no_grad():
        for test, labels in test_loader:
            test, labels = test.cuda(), labels.cuda()
            output = model(test)
            loss = loss_func(output, labels)
            test_loss += loss.item() * len(test)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    return test_loss, accuracy

if __name__ == '__main__':

    batch_size = 75
    num_epoches = 50
    lr = 0.05

    #预处理load之后的数据,这里将数据转换成了tensor数据类型，同时对数据做了归一化操作
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
    )

    # 加载数据集，这里直接使用了torchvision中的datasets，遇到实际情况可以制作自己的dataset，交给dataloader读取
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # torchvision中的数据读取接口，方便对数据设置batchsize，支持打乱和多线程取数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0,shuffle=True)


    # 初始化模型，如果支持cuda，使用cuda
    model = MLP(784, 128, 128, 10)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    # model = MLP(784, 128, 128, 10)
    # model.load_state_dict(torch.load("model/model_pure"))
    # model.eval()
    # model.cuda()

    # 用来显示模型的graph
    # dummy_input = torch.autograd.Variable(torch.rand(10, 784)).cuda()
    # writer.add_graph(model, (dummy_input,))



    # 分类任务，使用交叉熵作为代价函数
    loss_func = nn.CrossEntropyLoss()
    # 优化器，主要有Ada和SGD两大类
    # optimizer = optim.Adam(model.parameters(),lr =lr)
    optimizer = optim.SGD(model.parameters(),lr = lr,momentum=0.9)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    # 训练过程和测试过程
    for epoch in range(num_epoches):

        train_loss = train(model,train_loader,optimizer,loss_func)
        print("Epoch {}\t Train Loss:{}".format(epoch + 1, train_loss))

        # test
        test_loss, accuracy = validate(model,test_loader,loss_func)

        # 更新学习率
        # scheduler.step()

        # tensorboard show plot
        writer.add_scalars("loss curve", {"train_loss":train_loss, "test_loss":test_loss}, epoch+1)
        writer.add_scalar("test_Acc",accuracy,epoch+1)

        print("Epoch {}\t Test Loss:{}".format(epoch + 1, test_loss))
        print('Accuracy of the network on the test images: %.5f' % (
                accuracy))


    # save model's state_dict and reload in the future
    torch.save(model.state_dict(),"model/model_wosf_bn")
