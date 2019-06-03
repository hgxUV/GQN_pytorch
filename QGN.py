import torch
import torch.nn as nn
from tower import Tower
from convLSTM import ConvLSTM
from utils import Latent, ImageReconstruction
import numpy as np
from GQN_modules import Inference, Generator


class GQN(nn.Module):
    def __init__(self, shared, L):
        super(GQN, self).__init__()

        self.shared = shared
        self.L = L

        self.tower = Tower()
        if shared:
            self.inference = Inference()
            self.generation = Generator()
        else:
            self.inference = nn.ModuleList([Inference() for _ in range(L)])
            self.generation = nn.ModuleList([Generator() for _ in range(L)])

    def forward(self, r, v_query, z, state, hidden, u):

        concatenated = torch.cat([r, v_query, z, hidden])
        hidden, state = self.clstm(concatenated, state)
        u += self.upsample(hidden)

        return hidden, state, u