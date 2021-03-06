import os
import gzip
import torch
import numpy as np


def collect_files(path, ext=None, key=None):
    if key is None:
        files = sorted(os.listdir(path))
    else:
        files = sorted(os.listdir(path), key=key)

    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[-1] == ext]

    return [os.path.join(path, fname) for fname in files]

_base_dir = os.path.expanduser('~/Workspace/dataset/gqn_dataset')


class GQNDataset:
    def __init__(self, base_dir=_base_dir, scene='shepard_metzler_5_parts',
                 mode='train', transform=None):
        self.base_dir = os.path.expanduser(base_dir)
        self.data_dir = os.path.join(self.base_dir, scene, mode)
        self.filenames = collect_files(self.data_dir, ext='.gz')
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]

        with gzip.open(filename, 'rb') as f:
            data = torch.load(f)

        images_list, poses_list = list(zip(*data))
        images_seqs = np.array(images_list).astype('float32')
        images_seqs *= 1.0 / 255.0
        images_seqs = np.rollaxis(images_seqs, 4, 2)
        poses_seqs = np.array(poses_list)
        pos = poses_seqs[:, :, 0:3]
        yaw = poses_seqs[:, :, 3:4]
        pitch = poses_seqs[:, :, 4:5]
        poses_with_angles = np.concatenate([pos, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)], axis=2)

        return images_seqs, poses_with_angles



