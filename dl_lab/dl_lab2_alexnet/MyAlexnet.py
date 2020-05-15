import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import Mydataset
from Mydataset import ToTensor
import time

from torchvision import datasets
from torchvision import models


class Alexnet(nn.Module):

    def __init__(self):
        super(Alexnet, self).__init__()
        # input [32,32,3]
        self.cnn = nn.Sequential(
            # layer1
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=3, stride=1, padding=1),  # output [32,32,96]
            nn.BatchNorm2d(num_features=96,momentum=0.9),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[16,16,96]

            # layer2
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=3, stride=1, padding=1),  # output [16,16,128]
            nn.BatchNorm2d(num_features=128,momentum=0.9),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # output [15,15,128]

            # layer3
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=0),  # output[13,13,256]
            nn.BatchNorm2d(num_features=256,momentum=0.9),
            # nn.Sigmoid(),
            nn.ReLU(),

            # layer4
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),  # output[13,13,256]
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            # nn.Sigmoid(),
            nn.ReLU(),

            # layer5
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, stride=1, padding=1),  # output[13,13,128]
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0)  # output[6,6,128]
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 128, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, din):
        # din = din.view(-1,32,32,3)
        x = self.cnn(din)
        x = x.view(-1, 6 * 6 * 128)
        x = self.fc(x)
        return x


def train(model, train_loader, optimizer, loss_fn):
    train_loss = 0.0
    # 把模型调到训练模式
    # 此时 BN层 和 Dropout层 是可调整的
    model.train()
    for data, targets in train_loader:

        if torch.cuda.is_available():
            data = data.to(gpu)
            targets = targets.to(gpu)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*len(data)
    train_loss = train_loss/len(train_loader.dataset)
    print("Epoch {}\t Batch_zise:{} Train Loss:{}".format(epoch + 1,batch_size, train_loss))

    return train_loss

def validate(model, test_loader, loss_fn):
    test_loss = 0.0
    total = 0
    correct = 0
    # 模型调成测试模式
    # 此时 BN层 和 Dropout层是fixed 防止测试结果不稳定
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:

            if torch.cuda.is_available():
                data = data.to(gpu)
                labels = labels.to(gpu)

            output = model(data)
            loss = loss_fn(output, labels)
            test_loss += loss.item() * len(data)

            # 测试正确率
            total += len(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)

    accuracy = 100 * correct / total
    print("Epoch {}\t batch_szie: {}\t Test Loss:{}".format(epoch + 1, batch_size, test_loss))
    print('Accuracy of the network on the test images:{:.5f}\t {}\{}'.format(accuracy, correct, total))
    return test_loss, accuracy



cpu = torch.device("cpu")
gpu = torch.device("cuda")

from torchsummary import summary
from torchvision import transforms

logger = SummaryWriter(comment='pure')
if __name__ == '__main__':

    batch_size = 32
    epoches = 20
    lr = 0.005



    root = "dataset/cifar10"
    # train_dataset = Mydataset.Cifar10(root, train=True,
    #                                   transform=lambda x: torch.tensor(x,dtype=torch.float))
    # test_dataset = Mydataset.Cifar10(root, train=False,
    #                                  transform=lambda x: torch.tensor(x,dtype=torch.float))

    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Mydataset.Cifar10(root, train=True,
                                      transform=train_trans)
    test_dataset = Mydataset.Cifar10(root, train=False,
                                     transform=test_trans)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 新建模型
    # model = Alexnet()
    # print(model)

    # 加载模型
    model = Alexnet()
    model.load_state_dict(torch.load("models/model_best"))
    model.eval()
    model.cuda()
    print(model)

    # 将model转移到gpu上
    if torch.cuda.is_available():
        model.to(gpu)

    # 使用 SGD 优化器
    optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9)
    # 交叉熵函数作为代价函数
    loss_fn = nn.CrossEntropyLoss()

    # 输出model的结构
    # summary(model,input_size=(3,32,32))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[20,40],0.1)


    for epoch in range(epoches):

        # 分段训练
        if epoch < 20:
            batch_size = 32
        else:
            batch_size = 128

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # 训练
        train_loss = train(model, train_loader, optimizer, loss_fn)
        # 测试
        test_loss,accuracy = validate(model, test_loader, loss_fn)


        logger.add_scalars("loss curve", {"train_loss": train_loss, "test_loss": test_loss}, epoch + 1)
        logger.add_scalar("test_Acc", accuracy, epoch + 1)
        # 更新参数
        scheduler.step()




