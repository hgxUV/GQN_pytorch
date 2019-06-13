import torch
import torch.nn as nn
from tower import Tower
from convLSTM import ConvLSTM
from utils import Latent, ImageReconstruction
import numpy as np
from GQN_modules import Inference, Generator
from utils import Latent, ImageReconstruction


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

        self.prior_net = Latent(256, 256, 5)
        self.posterior_net = Latent(256, 256, 5)
        self.img_reconstructor = ImageReconstruction(256, 3, 5, 3)

    def forward(self, x, p, x_q, p_q):
        r = self.tower(x, p)

        gen_h, gen_s, u = Inference.get_init_state(x.shape[0])
        inf_h, inf_s = Generator.get_init_state(x.shape[0])

        if not self.shared:
        posteriors = []
        priors = []

        if not self.shared:
            for i in range(self.L):
                #x_query, v_query, r, z, u, gen_hidden, state, hidden
                inf_s, inf_h = self.inference[i](x_q, p_q, r, u, gen_h, inf_s, inf_h)
                z, post_params = self.posterior_net(inf_h)

                z, prior_params = self.prior_net(gen_h)

                params = post_params if DUPA else prior_params

                #r, v_query, z, state, hidden, u
                gen_s, gen_h, u = self.generation[i](r, p_q, params, gen_s, gen_h, u)


        else:
            for i in range(self.L):
                #x_query, v_query, r, z, u, gen_hidden, state, hidden
                inf_s, inf_h = self.inference(x_q, p_q, r, u, gen_h, inf_s, inf_h)
                z, post_params = self.posterior_net(inf_h)

                z, prior_params = self.prior_net(gen_h)

                params = post_params if DUPA else prior_params

                #r, v_query, z, state, hidden, u
                gen_s, gen_h, u = self.generation(r, p_q, params, gen_s, gen_h, u)

        return self.img_reconstructor(u)