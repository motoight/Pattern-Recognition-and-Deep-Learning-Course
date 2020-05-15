import torch
import numpy as np
from torch import nn
from torch import optim
from matplotlib import pyplot as plt


class RnnCell(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(RnnCell, self).__init__()
        self.hidden = nn.Linear(
            input_size+hidden_size, hidden_size
        )

    def forward(self, x, h_state):
        combined = torch.cat((x,h_state), 1)
        h_state = self.hidden(combined)
        return torch.tanh(h_state)


class BasicRnn(nn.Module):
    '''
    Rnn module from scratch
    input dim [input,hidden,output]
    '''
    def __init__(self,input_dim, hidden_dim,output_dim):
        super(BasicRnn,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnncell = RnnCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim,output_dim),
            # nn.ReLU(),
            # nn.Linear(50,output_dim)
        )

    def forward(self, x,future=0):
        # x shape  [batch,seq_len,input_size=1]
        batch, seq_len, _ = x.shape
        h_state = torch.zeros(batch, self.hidden_dim)
        output = torch.zeros(batch, seq_len, self.output_dim)
        for i in range(seq_len):
            inp = x[:, i, :]
            h_state = self.rnncell(inp,h_state)
            # h_state = h_state.detach()
            out = self.linear(h_state)
            output[:,i] = out

        predict = torch.zeros(batch, future, self.output_dim)
        inp = x[:, -1 , :]
        for i in range(future):  # if we should predict the future
            h_state = self.rnncell(inp,h_state)
            # h_state = h_state.detach()
            # use current output as new input
            out = self.linear(h_state)
            inp = out
            predict[:, i] = out
        return output, predict


