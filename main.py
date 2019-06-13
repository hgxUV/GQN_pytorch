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
L=12
EPOCHS = 100
shared=True



if __name__ == '__main__':

    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    cameras, positions = ds[0]

    qgn = GQN(shared=shared, L=L)

    optimizer = optim.Adam(qgn.parameters(), lr=1e-3)

    writer = SummaryWriter()
    global_step = 0

    for epoch in range(EPOCHS):

        for i in range(0, N, BATCH_SIZE):
            view_batch, pos_batch = cameras[i:i+BATCH_SIZE], positions[i:i+BATCH_SIZE]
            x = torch.from_numpy(view_batch[:,0:5, ...])
            p = torch.from_numpy(pos_batch[:,0:5, ...])
            x_q = torch.from_numpy(view_batch[:, -1, ...])
            p_q = torch.from_numpy(pos_batch[:, -1, ...])
            output_image, target_image, priors, posteriors = qgn(x, p, x_q, p_q)

            global_step += 1
            loss = loss(output_image, target_image, priors, posteriors)
            writer.add_scalar('loss', loss, global_step)
            writer.add_image('output_image', output_image, global_step)

            loss.backward()
            optimizer.step()

        torch.save(qgn.state_dict(), '/checkpoints/model.pt')
