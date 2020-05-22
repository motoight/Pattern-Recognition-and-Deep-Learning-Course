import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter

'''
impemention of Wasserstein GAN Gradient penalty
'''
class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    def __init__(self, input_size,hidden_dim, output_size):
        super(Generator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


writer = SummaryWriter("runs-5\compare\wgangp-1")

params = {
"epoches" : 200,
"batch_size" : 500,
"lr" : 0.0002,
"hidden_dim" : 256,
"noise_dim" : 5,
"optim_type":"rmsprop",
"lamda":0.001
}

epoches = params["epoches"]
batch_size = params["batch_size"]
lr = params["lr"]
hidden_dim = params["hidden_dim"]
noise_dim = params["noise_dim"]
optim_type = params["optim_type"]
lamda = params["lamda"]

# -----------------
#  log params
# -----------------
for k,v in params.items():
    writer.add_text(k,str(v))





def make_noise(n_size, n_dim):
    '''
    :param n_size: the noise point num
    :param n_dim: the dim of noise
    :return: nddarray type [n_szie, n_dim]
    '''
    return torch.Tensor(np.random.normal(0, 1, (n_size, n_dim)))

def _cal_gradient_penalty(real_batch, fake_batch, lamda):
    '''
    :param real_batch: real data batch
    :param fake_batch:  fake data batch
    :param lamda: hyperparams weighted penalty
    :return: return gradient penalty based given batch
    '''
    alpha = torch.rand(real_batch.size(0),1)
    alpha = alpha.expand(real_batch.size())
    alpha = alpha.cuda()

    inter_point = alpha*real_batch+(1-alpha)*fake_batch

    inter_point.cuda()

    critic = discriminator(inter_point)

    gradient = torch.autograd.grad(outputs=critic,inputs=inter_point,
                                   grad_outputs=torch.ones(critic.size()).cuda(),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]

    gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * lamda

    return gradient_penalty





filepath = 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab5\points.mat'

points_set = utils.Points(filepath)
data_loader = torch.utils.data.DataLoader(points_set, batch_size=batch_size, shuffle=True, num_workers=0)

discriminator = Discriminator(2,hidden_dim,1)
generator = Generator(noise_dim,hidden_dim, 2)

discriminator.cuda()
generator.cuda()

if optim_type == "rmsprop":
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr = lr)
    optimizer_G = optim.RMSprop(generator.parameters(), lr = lr)
elif optim_type == "adam":
    optimizer_D = optim.Adam(discriminator.parameters(), lr = lr, betas=(0.001, 0.8))
    optimizer_G = optim.Adam(generator.parameters(), lr = lr, betas=(0.001, 0.8))
elif optim_type == "sgd":
    optimizer_D = optim.SGD(discriminator.parameters(), lr=lr, momentum=0)
    optimizer_G = optim.SGD(generator.parameters(), lr=lr, momentum=0)
else:
    raise TypeError("optim type not found %s" % (optim_type))

#calculate background
x = utils.readData(filepath)

lx = np.floor(np.min(x[:,0]))
hx = np.ceil(np.max(x[:,0]))
ly = np.floor(np.min(x[:,1]))
hy = np.ceil(np.min(x[:,1]))

x_aix = np.arange(lx,hx,0.01)
y_aix = np.arange(ly,hy,0.01)
xx,yy = np.meshgrid(x_aix, y_aix)

xx = torch.from_numpy(xx)
yy = torch.from_numpy(yy)
bc = torch.stack((xx,yy),dim = 2)
bc = bc.view(-1,2)
bc_cuda = bc.view(-1,2).cuda().float()





# train
for epoch in range(epoches):
    for n_batch, real_batch in enumerate(data_loader):
        N = real_batch.size(0)

        real_batch = real_batch.cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # for p in discriminator.parameters():
        #     p.requires_grad = True

        optimizer_D.zero_grad()

        z = make_noise(N, noise_dim).cuda()

        gen_data = generator(z)

        gradient_penalty = _cal_gradient_penalty(real_batch,gen_data,lamda)

        d_loss = torch.mean(discriminator(gen_data))-torch.mean(discriminator(real_batch))+gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        # for p in discriminator.parameters():
        #     p.requires_grad = False  # to avoid computation

        optimizer_G.zero_grad()

        z = make_noise(N, noise_dim).cuda()

        gen_data = generator(z)

        g_loss = - torch.mean(discriminator(gen_data))

        g_loss.backward()
        optimizer_G.step()



        if True:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epoches, n_batch, len(data_loader), d_loss.item(), g_loss.item())
            )


            def plt_pic(gen_data, x):
                '''
                visualize training results
                :param gen_data:
                :return:
                '''
                fig = plt.figure()
                plt.axis([lx, hx, ly, hy])
                boundary = discriminator(bc_cuda).detach()
                # cal w-distance
                baseline = torch.from_numpy(x).cuda().float()
                boundary = torch.abs(boundary - torch.mean(discriminator(baseline)))
                boundary = boundary.view(100, 300).detach().to("cpu")
                plt.contourf(xx, yy, boundary, locator=ticker.LinearLocator(), cmap=plt.cm.binary, alpha=.75)
                plt.colorbar()
                plt.scatter(x[:, 0], x[:, 1], color='b', alpha=0.6, s=4)
                plt.scatter(gen_data[:, 0], gen_data[:, 1], color='r', alpha=0.6, s=4)
                # plt.show()
                return fig


            writer.add_scalar('d_loss', d_loss.item(), epoch)
            writer.add_scalar('g_loss', g_loss.item(), epoch)

        if epoch % 10 == 0:
            z = make_noise(1000, noise_dim).cuda()
            gen_data = generator(z).detach().to("cpu")
            fig = plt_pic(gen_data, x)
            tag = 'epoch' + str(epoch)
            writer.add_figure(tag=tag, figure=fig)




