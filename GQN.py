from tower import Tower
from convLSTM import ConvLSTM
from utils import Latent, ImageReconstruction

class Generator:
    def __init__(self):
        self.clstm = ConvLSTM(519, 256, 5)
        self.upsample = nn.ConvTranspose2D(256, 4)

    def forward(self, r, v_query, z, state, hidden, u):
        hidden, state = self.clstm(torch.cat([r, v_query, z, hidden]), state)
        u += self.upsample(hidden)

        return hidden, state, u


class Inference:
    def __init__(self):
        self.clstm = ConvLSTM(552, 256, 5)
        self.downsample = nn.Conv2D(256, 256, 5, 2)

    def forward(self, x_query, v_query, r, z, u, gen_hidden, state, hidden):
        downsampled_u = self.downsample(u)
        hidden, state = self.clstm(torch.cat([x_query, v_query, r, z, downsampled_u, gen_hidden, hidden]), state)
        return hidden, state
