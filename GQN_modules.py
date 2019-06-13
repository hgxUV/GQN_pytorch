import torch
import torch.nn as nn
from convLSTM import ConvLSTM
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.clstm = ConvLSTM(775, 256, 5)
        self.upsample = nn.ConvTranspose2d(256, 256, 4, stride=4, padding=0)
        self.ones = np.ones((10, 10))

    def forward(self, r, v_query, z, state, hidden, u):

        concatenated = torch.cat([r, v_query, z, hidden], dim=1)
        hidden, state = self.clstm(concatenated, state)
        upsampled_h = self.upsample(hidden)
        u += upsampled_h

        return hidden, state, u

    @classmethod
    def get_init_state(cls, batch_size):
        h = torch.zeros(batch_size, 256, 16, 16)
        s = torch.zeros(batch_size, 256, 16, 16)
        return h, s


class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()

        self.clstm = ConvLSTM(1034, 256, 5)
        self.downsample = nn.Conv2d(256, 256, 5, 4, padding=1)

    def forward(self, x_query, v_query, r, u, gen_hidden, state, hidden):

        downsampled_u = self.downsample(u)
        concatenated = torch.cat([x_query, v_query, r, downsampled_u, gen_hidden, hidden], dim=1)
        hidden, state = self.clstm(concatenated, state)

        return hidden, state

    @classmethod
    def get_init_state(cls, batch_size):
        h = torch.zeros(batch_size, 256, 16, 16)
        s = torch.zeros(batch_size, 256, 16, 16)
        u = torch.zeros(batch_size, 256, 64, 64)
        return h, s, u

