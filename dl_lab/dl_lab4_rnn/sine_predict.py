
import torch
import numpy as np
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import nets
from tensorboardX import SummaryWriter

# writer = SummaryWriter('runs-4\sine')
datapath = 'C:\\Users\Snow\Desktop\大三下\模式识别与深度学习\Dlab\lab4\\traindata.pt'
# load data, get ndarray
data = torch.load(datapath)
# transform data to tensor
data = torch.from_numpy(data)

train_set = data[:98]
val_set = data[98:]

x = train_set[:,:-1]

# convert to [seq_len,batch_size,input_size=1]
x = torch.unsqueeze(x,2).float()

y = train_set[:,1:]
y = torch.unsqueeze(y,2).float()

hidden_size = 10
seq_len = 999
epoches = 1000


# model = nets.BasicRnn(1,hidden_size,1)
model = nets.LSTM(1,hidden_size,1)
print(model)

lr = 0.5
loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(),lr = lr)
optimizer = optim.LBFGS(model.parameters(),lr = lr)



epoch = 0
# train
for i in range(10):
    print('STEP: ', i)
    def closure():
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = loss_fn(outputs, y)
        # writer.add_scalar("train_loss", loss.item())
        print("train loss:{}".format(loss.item()))
        loss.backward()
        return loss
    optimizer.step(closure)

    # optimizer.zero_grad()
    # outputs, _ = model(x)
    # loss = loss_fn(outputs, y)
    # if i% 100 == 0:
    #     print("Epoch[{}/{}] train loss: {}" .format(i, epoches, loss.item()))
    # loss.backward()
    # optimizer.step()


# test and predict
z0 = val_set[:,1:500]
z0 = torch.unsqueeze(z0, 2).float()
zhat = val_set[:,500:]
zhat = torch.unsqueeze(zhat, 2).float()

h_state = torch.autograd.Variable(torch.zeros(1,hidden_size)).float()

with torch.no_grad():
    model.eval()
    outputs, predict = model(z0,500)
    loss = loss_fn(predict, zhat)
    print("test loss:{}".format(loss.item()))
print(predict.shape)

plt.title("sine-wav predict lstm")
def draw(yhat, yi, color):
    plt.plot(np.arange(len(yhat)), yhat, color = 'b',label = 'ground truth')
    plt.plot(np.arange(len(yi)), yi, linestyle=':', color = 'r', label = 'predicted')

draw(zhat[0],predict[0],'b')
# draw(zhat[1],predict[1],'r')
plt.legend()
plt.show()