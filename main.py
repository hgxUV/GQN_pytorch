from loader import GQNDataset
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
N = 5000



if __name__ == '__main__':


    ds = GQNDataset(mode='train', base_dir='data', scene='rooms_ring_camera')
    x, pos = ds[0]

    n = 6
    f = plt.figure(figsize=(12, 8))
    axes = f.subplots(nrows=n, ncols=1, sharex=True, sharey=True)
    for i in range(0, N, BATCH_SIZE):
        x_batch, pos_batch = x[i:i+BATCH_SIZE], pos[i:i+BATCH_SIZE]
        images = images_list[i]
        #grid = np.hstack(images[:10])
        #axes[i].imshow(grid)

    plt.show()
    a = 15