from loader import GQNDataset
import matplotlib.pyplot as plt
import numpy as np
from QGN import GQN

BATCH_SIZE = 32
N = 5000
L=12
shared=False



if __name__ == '__main__':


    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    x, pos = ds[0]

    qgn = GQN(shared=shared, L=L)

    n = 6
    f = plt.figure(figsize=(12, 8))
    axes = f.subplots(nrows=n, ncols=1, sharex=True, sharey=True)
    for i in range(0, N, BATCH_SIZE):
        view_batch, pos_batch = x[i:i+BATCH_SIZE], pos[i:i+BATCH_SIZE]
        x = view_batch[:,0:5, ...]
        p = pos_batch[:,0:5, ...]
        x_q = view_batch[:, -1, ...]
        p_q = pos_batch[:, -1, ...]
        qgn(x, p, x_q, p_q)
        #grid = np.hstack(images[:10])
        #axes[i].imshow(grid)

    plt.show()
    a = 15