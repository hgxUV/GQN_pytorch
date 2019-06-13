from loader import GQNDataset
import matplotlib.pyplot as plt
import numpy as np
from GQN import GQN

BATCH_SIZE = 32
N = 5000
L=12
shared=False



if __name__ == '__main__':


    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    cameras, positions = ds[0]

    qgn = GQN(shared=shared, L=L)

    for i in range(0, N, BATCH_SIZE):
        view_batch, pos_batch = cameras[i:i+BATCH_SIZE], positions[i:i+BATCH_SIZE]
        x = view_batch[:,0:5, ...]
        p = pos_batch[:,0:5, ...]
        x_q = view_batch[:, -1, ...]
        p_q = pos_batch[:, -1, ...]
        qgn(x, p, x_q, p_q)