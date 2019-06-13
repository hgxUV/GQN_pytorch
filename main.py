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
L=1
EPOCHS = 100
shared=True
NO_FEATURES = 256

sigma_i = torch.tensor([2.0])
sigma_f = torch.tensor([0.7])
sigma_n = torch.tensor([2.0 * 1e5])



if __name__ == '__main__':

    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    ts = GQNDataset(mode='test', base_dir='data', scene='rooms_ring_camera')

    cameras, positions = ds[0]
    tc, tp = ts[0]

    device = torch.device("cuda:0")
    qgn = GQN(shared, L, sigma_i.to(device), sigma_f.to(device), sigma_n.to(device), device)
    qgn.to(device)

    optimizer = optim.Adam(qgn.parameters(), lr=1e-3)

    writer = SummaryWriter()
    global_step = torch.tensor([0.0]).to(device)

    test_i = 0

    for epoch in range(EPOCHS):

        for i in range(0, N, BATCH_SIZE):
            view_batch, pos_batch = cameras[i:i+BATCH_SIZE], positions[i:i+BATCH_SIZE]
            x = torch.from_numpy(view_batch[:,0:5, ...]).to(device)
            p = torch.from_numpy(pos_batch[:,0:5, ...]).to(device)
            x_q = torch.from_numpy(view_batch[:, -1, ...]).to(device)
            p_q = torch.from_numpy(pos_batch[:, -1, ...]).to(device)
            output_image, priors, posteriors = qgn(x, p, x_q, p_q, global_step.float(), training=True)

            global_step += 1
            total_loss, model_loss, dist_loss = loss(output_image, x_q, priors, posteriors, NO_FEATURES)
            writer.add_scalar('total loss', total_loss, global_step)
            writer.add_scalar('model loss', model_loss, global_step)
            writer.add_scalar('dist loss', dist_loss, global_step)

            writer.add_image('output_image', output_image[0], global_step)
            writer.add_image('input_image_1', x[0][0], global_step)
            writer.add_image('input_image_2', x[0][1], global_step)
            writer.add_image('input_image_3', x[0][2], global_step)
            writer.add_image('input_image_4', x[0][3], global_step)
            writer.add_image('input_image_5', x[0][4], global_step)


            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                view_batch, pos_batch = tc[test_i:test_i + BATCH_SIZE], tp[test_i:test_i + BATCH_SIZE]
                x = torch.from_numpy(view_batch[:, 0:5, ...]).to(device)
                p = torch.from_numpy(pos_batch[:, 0:5, ...]).to(device)
                x_q = torch.from_numpy(view_batch[:, -1, ...]).to(device)
                p_q = torch.from_numpy(pos_batch[:, -1, ...]).to(device)
                output_image, priors, posteriors = qgn(x, p, x_q, p_q, global_step.float(), training=False)
                test_i += BATCH_SIZE

        torch.save(qgn.state_dict(), '/checkpoints/model.pt')
