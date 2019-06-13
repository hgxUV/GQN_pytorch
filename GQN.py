import torch
import torch.nn as nn
import torch.nn.functional as F
from tower import Tower
from convLSTM import ConvLSTM
from utils import Latent, ImageReconstruction
import numpy as np
from GQN_modules import Inference, Generator
from utils import Latent, ImageReconstruction, get_pixel_std


class GQN(nn.Module):
    def __init__(self, shared, L, sigma_i, sigma_f, sigma_n, device):
        super(GQN, self).__init__()

        self.shared = shared
        self.L = L
        self.tower = Tower()
        self.sigma_i = sigma_i
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.device = device

        if shared:
            self.inference = Inference()
            self.generation = Generator()
        else:
            self.inference = nn.ModuleList([Inference() for _ in range(L)])
            self.generation = nn.ModuleList([Generator() for _ in range(L)])

        self.prior_net = Latent(256, 256, 5)
        self.posterior_net = Latent(256, 256, 5)
        self.img_reconstructor = ImageReconstruction(256, 3, 5, device)

    def forward(self, x, p, x_q, p_q, global_step, training=True):
        r = self.tower(x, p)
        x_q = F.interpolate(x_q, scale_factor=1/4.0, mode='bilinear')

        v1 = p_q.unsqueeze(-1)
        v2 = v1.expand((v1.shape[0], 7, 16))

        v3 = v2.unsqueeze(-1)
        v4 = v3.expand((v1.shape[0], 7, 16, 16))
        p_q = v4.reshape(-1, 7, 16, 16)

        gen_h, gen_s, u = Inference.get_init_state(x.shape[0])
        inf_h, inf_s = Generator.get_init_state(x.shape[0])

        gen_h = gen_h.to(self.device)
        gen_s = gen_s.to(self.device)
        u = u.to(self.device)

        inf_h = inf_h.to(self.device)
        inf_s = inf_s.to(self.device)

        posteriors = []
        priors = []

        if not self.shared:
            for i in range(self.L):
                #x_query, v_query, r, z, u, gen_hidden, state, hidden
                inf_s, inf_h = self.inference[i](x_q, p_q, r, u, gen_h, inf_s, inf_h)
                post_z, post_params = self.posterior_net(inf_h)
                posteriors.append(post_params)

                prior_z, prior_params = self.prior_net(gen_h)
                priors.append(prior_params)

                params = post_z if training else prior_z

                #r, v_query, z, state, hidden, u
                gen_s, gen_h, u = self.generation[i](r, p_q, params, gen_s, gen_h, u)


        else:
            for i in range(self.L):
                #x_query, v_query, r, z, u, gen_hidden, state, hidden
                inf_s, inf_h = self.inference(x_q, p_q, r, u, gen_h, inf_s, inf_h)
                post_z, post_params = self.posterior_net(inf_h)
                posteriors.append(post_params)

                prior_z, prior_params = self.prior_net(gen_h)
                priors.append(prior_params)

                params = post_z if training else prior_z

                #r, v_query, z, state, hidden, u
                gen_s, gen_h, u = self.generation(r, p_q, params, gen_s, gen_h, u)

        sigma = get_pixel_std(self.sigma_i, self.sigma_f, self.sigma_n, global_step)

        return self.img_reconstructor(u, sigma), priors, posteriors