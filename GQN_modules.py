import torch
import torch.nn as nn
from convLSTM import ConvLSTM
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.clstm = ConvLSTM(519, 256, 5)
        self.upsample = nn.ConvTranspose2D(256, 256, 4)
        self.ones = np.ones((10, 10))

    def forward(self, r, v_query, z, state, hidden, u):

        concatenated = torch.cat([r, v_query, z, hidden])
        hidden, state = self.clstm(concatenated, state)
        u += self.upsample(hidden)

        return hidden, state, u


class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()

        self.clstm = ConvLSTM(552, 256, 5)
        self.downsample = nn.Conv2D(256, 256, 5, 2)

    def forward(self, x_query, v_query, r, z, u, gen_hidden, state, hidden):

        downsampled_u = self.downsample(u)
        concatenated = torch.cat([x_query, v_query, r, z, downsampled_u, gen_hidden, hidden])
        hidden, state = self.clstm(concatenated, state)

        return hidden, state
