import torch
import numpy as np
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import nets
import datautils
from tensorboardX import SummaryWriter


writer = SummaryWriter('runs-4\sentiment-lstm-sigdp')
model = nets.LSTM(50,50,2)

# model = nets.BasicRnn(50,50,2)
lr = 0.001
batch_size = 200
loss_fn = nn.CrossEntropyLoss()

optim = optim.Adam(model.parameters(),lr =lr)
# optim = optim.SGD(model.parameters(),lr =lr,momentum=0.9)

# 获取训练集和测试集
negpath= 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\movie_review\\rt-polarity.neg'
pospath = 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\movie_review\\rt-polarity.pos'
glovepath= 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\dataset\glove.6B.50d.txt'

train_set = datautils.Movie_Review(negpath,pospath,glovepath,train=True)
test_set = datautils.Movie_Review(negpath,pospath,glovepath,train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

epoches = 200

for epoch in range(epoches):
# train
    model.train()
    train_loss = 0
    for data, target in train_loader:
        output, _ = model(data)
        optim.zero_grad()
        target = target.long()
		# 网络的输出是一个time_step 序列，
		# 对于一句话的分类结果只需要截取最后一个output即可
        output = output[:,-1]
        loss= loss_fn(output,target)
        loss.backward()
        optim.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    writer.add_scalar('train_loss', train_loss, epoch+1)
    print("Epoch [{}/{}] train loss: {}".format(epoch,epoches,train_loss))

# test
    model.eval()
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        output,_ = model(data)
        target = target.long()
		
        output = output[:, -1]
        loss= loss_fn(output,target)
        _, predict = torch.max(output, 1)
        test_loss += loss.item()
        correct += (predict == target).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    writer.add_scalar('test_loss', test_loss,epoch+1)
    writer.add_scalar('test_acc', acc, epoch + 1)
    print("test loss: {}".format(test_loss))
    print("test acc:{} [{}/{}]".format(acc,correct,len(test_loader.dataset)))

