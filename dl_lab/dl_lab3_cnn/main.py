import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import optim
from tensorboardX import SummaryWriter
import Mydataset
import models
import time

class Cifar10_Trainer():

    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    def __init__(self):

        parser = argparse.ArgumentParser(description='Pytroch Cifar10 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate\n default:0.01')
        parser.add_argument('--bs', default=256, type=int, help='batch size\n default:256')
        parser.add_argument('--gpu', default=True, help='use gpu as accelerator')
        parser.add_argument('--net', default='vgg16', choices=['vgg16','resnet18','sevgg16','seresnet18'],
                            help='provided network:vgg16, resnet18, sevgg16, seresnet18')
        parser.add_argument('--verbose',default=True,help='print some useful info')
        parser.add_argument('--e', default=50, type=int, help='the num of training epoch\n default:50')

        args = parser.parse_args()
        # parse and preparing params
        self.batch_size = args.bs
        self.epoches = args.e
        self.lr = args.lr
        self.use_gpu = args.gpu
        self.net = args.net
        self.model = None
        self.verbose = args.verbose

        if self.net == 'vgg16':
            self.model = models.VGGNet()
        elif self.net == 'resnet18':
            self.model = models.Resnet()
        elif self.net == 'sevgg16':
            self.model = models.SeVGGNet()
        elif self.net == 'seresnet18':
            self.model = models.SeResnet()

        if self.use_gpu and torch.cuda.is_available():
            self.model.to(self.gpu)
        if self.verbose:
            print(self.model)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr, momentum=0.9)


    def init_data(self):
        # preparing data
        t0 = time.time()
        print('==> Preparing data..')
        root = "dataset/cifar10"
        self.train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            #         transforms.RandomCrop((32,32),pad_if_needed=True,padding_mode='edge'),
            #         transforms.RandomGrayscale(p=0.1),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = Mydataset.Cifar10(root, train=True,
                                          transform=self.train_trans)
        self.test_dataset = Mydataset.Cifar10(root, train=False,
                                         transform=self.test_trans)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        print('==> Preparing data finished in {:.4f}s  ..'.format(time.time()-t0))

    def train(self,epoch):
        train_loss = 0.0
        # 把模型调到训练模式
        # 此时 BN层 和 Dropout层 是可调整的
        self.model.train()
        for data, targets in self.train_loader:

            if torch.cuda.is_available() and self.use_gpu:
                data = data.to(self.gpu)
                targets = targets.to(self.gpu)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * len(data)
        train_loss = train_loss / len(self.train_loader.dataset)
        print("Epoch {}\t Batch_zise:{} Train Loss:{}".format(epoch + 1, self.batch_size, train_loss))

        return train_loss

    def test(self,epoch):
        test_loss = 0.0
        total = 0
        correct = 0
        # 模型调成测试模式
        # 此时 BN层 和 Dropout层是fixed 防止测试结果不稳定
        self.model.eval()
        with torch.no_grad():
            for data, labels in self.test_loader:

                if torch.cuda.is_available() and self.use_gpu:
                    data = data.to(self.gpu)
                    labels = labels.to(self.gpu)

                output = self.model(data)
                loss = self.loss_fn(output, labels)
                test_loss += loss.item() * len(data)

                # 测试正确率
                total += len(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(self.test_loader.dataset)

        accuracy = 100 * correct / total
        print("Epoch {}\t batch_szie: {}\t Test Loss:{}".format(epoch + 1, self.batch_size, test_loss))
        print('Accuracy of the network on the test images:{:.5f}\t {}\{}'.format(accuracy, correct, total))
        return test_loss, accuracy

    def run(self):
        self.init_data()
        for epoch in range(self.epoches):
            self.train(epoch)
            self.test(epoch)


if __name__ == '__main__':
    trainer = Cifar10_Trainer()
    trainer.run()