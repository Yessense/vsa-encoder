from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedDspritesDataset(Dataset):
    def __init__(self,
                 dsprites_path='./dataset/data/dsprites_train.npz',
                 paired_dsprites_path='./dataset/data/100_30_dataset/paired_train.npz'):
        # Load npz numpy archive
        dsprites = np.load(dsprites_path, allow_pickle=True)
        paired_dsprites = np.load(paired_dsprites_path, allow_pickle=True)

        self.data = paired_dsprites['data']
        self.exchanges = paired_dsprites['exchanges']

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dsprites['imgs']

        # List of feature names
        self.feature_names: Tuple[str, ...] = ('shape', 'scale', 'orientation', 'posX', 'posY')

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dsprites['latents_classes'][:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.imgs[self.data[idx][0]]
        img = torch.from_numpy(img).float().unsqueeze(0)

        pair_img = self.imgs[self.data[idx][1]]
        pair_img = torch.from_numpy(pair_img).float().unsqueeze(0)

        exchange = torch.from_numpy(self.exchanges[idx]).bool().unsqueeze(-1)
        return img, pair_img, exchange
