import torch
import numpy as np
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

class LSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(LSTMCell,self).__init__()
        self.update = nn.Linear(input_size+hidden_size,hidden_size)
        self.forget = nn.Linear(input_size+hidden_size,hidden_size)
        self.output = nn.Linear(input_size+hidden_size,hidden_size)
        self.hidden = nn.Linear(input_size+hidden_size,hidden_size)

    def forward(self, x, h_state):
        # lstm inherent two params
        c_t, a_t = h_state
        combine = torch.cat((x,a_t), dim = 1)
        cdash = torch.tanh(self.hidden(combine))
        fu = torch.sigmoid(self.update(combine))
        ff = torch.sigmoid(self.forget(combine))
        fo = torch.sigmoid(self.forget(combine))
        c = fu.mul(cdash)+ff.mul(c_t)
        a = fo.mul(torch.tanh(c))
        return c,a

class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.lstmcell = LSTMCell(input_size,hidden_size)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size,50),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(50,output_size)
        )
    def forward(self,x, future=0):
        # x shape  [batch,seq_len,input_size=1]
        batch, seq_len, _ = x.shape
        c_t = torch.zeros(batch,self.hidden_dim)
        a_t = torch.zeros(batch,self.hidden_dim)
        output = torch.zeros(batch, seq_len, self.output_dim)
        for i in range(seq_len):
            inp = x[:, i, :]
            c_t, a_t = self.lstmcell(inp,(c_t, a_t))
            output[:,i] = self.linear(a_t)

        predict = torch.zeros(batch, future, self.output_dim)
        inp = x[:, -1, :]
        for i in range(future):  # if we should predict the future
            c_t, a_t = self.lstmcell(inp,(c_t, a_t))
            # h_state = h_state.detach()
            # use current output as new input
            out = self.linear(a_t)
            inp = out
            predict[:, i] = out
        return output, predict




