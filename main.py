from loader import GQNDataset
import matplotlib.pyplot as plt
import numpy as np
from GQN import GQN
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import loss
import torch

BATCH_SIZE = 32
N = 5000
L=2
EPOCHS = 100
shared=True

sigma_i = torch.tensor([2.0])
sigma_f = torch.tensor([0.7])
sigma_n = torch.tensor([2.0 * 1e5])



if __name__ == '__main__':

    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    ts = GQNDataset(mode='test', base_dir='data', scene='rooms_ring_camera')

    cameras, positions = ds[0]
    tc, tp = ts[0]

    qgn = GQN(shared, L, sigma_i, sigma_f, sigma_n)

    optimizer = optim.Adam(qgn.parameters(), lr=1e-3)

    writer = SummaryWriter()
    global_step = torch.tensor([0])

    test_i = 0

    for epoch in range(EPOCHS):

        for i in range(0, N, BATCH_SIZE):
            view_batch, pos_batch = cameras[i:i+BATCH_SIZE], positions[i:i+BATCH_SIZE]
            x = torch.from_numpy(view_batch[:,0:5, ...])
            p = torch.from_numpy(pos_batch[:,0:5, ...])
            x_q = torch.from_numpy(view_batch[:, -1, ...])
            p_q = torch.from_numpy(pos_batch[:, -1, ...])
            output_image, priors, posteriors = qgn(x, p, x_q, p_q, global_step.float(), training=True)

            global_step += 1
            loss = loss(output_image, target_image, priors, posteriors)
            writer.add_scalar('loss', loss, global_step)
            writer.add_image('output_image', output_image, global_step)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                view_batch, pos_batch = tc[test_i:test_i + BATCH_SIZE], tp[test_i:test_i + BATCH_SIZE]
                x = torch.from_numpy(view_batch[:, 0:5, ...])
                p = torch.from_numpy(pos_batch[:, 0:5, ...])
                x_q = torch.from_numpy(view_batch[:, -1, ...])
                p_q = torch.from_numpy(pos_batch[:, -1, ...])
                output_image, target_image, priors, posteriors = qgn(x, p, x_q, p_q, global_step, training=False)
                test_i += BATCH_SIZE

        torch.save(qgn.state_dict(), '/checkpoints/model.pt')
