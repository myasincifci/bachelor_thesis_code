import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import scipy.stats as stats

def p_uni(i: int, tau: int, I: int):
    '''
    PDF for sampling f_j given reference frame f_i.
    '''

    x = np.arange(I)
    m = ((x != i) & (x <= i + tau) & (x >= i - tau)).astype(int)
    p = 1 / (min(tau, i) + min(tau, I - i - 1))

    return m * p

def p_nml(i: int, sigma: float, I):
    '''
    PDF for sampling f' given reference frame f.
    '''
    x = np.arange(I)
    pdf = stats.norm.pdf(x, i, sigma)
    pdf[i] = 0.0
    p = pdf / pdf.sum()

    return p

class VideoDataset(Dataset):
    def __init__(self, path, transform=None, proximity:int=3) -> None:
        self.p = proximity
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if not p.endswith('.txt')])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. get one element x
        image = Image.open(self.image_paths[index])

        # 2. sample element x' in the neighbourhood of x within proximity (between 1 and p where p is proximity)
        # possbile_l = range(((index - self.p) if (index - self.p) >= 0 else 0), index)
        # possbile_r = range(index + 1, ((index + self.p + 1) if (index + self.p + 1) <= len(self) else len(self)))
        # index_d = np.random.choice(list(possbile_l) + list(possbile_r))
        # image_d = Image.open(self.image_paths[index_d])

        pb = p_nml(index, self.p, self.__len__())
        index_d = np.random.choice(np.arange(len(pb)), p=pb)
        image_d = Image.open(self.image_paths[index_d])

        if self.transform:
            image = self.transform(image)
            image_d = self.transform(image_d)

        # 3. return (x, x')
        return (image, image_d, torch.tensor(0), torch.tensor(0))

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    train_dataset = VideoDataset('./datasets/hand_2', transform=transform, proximity=30, train=True)
    test_dataset = VideoDataset('./datasets/hand_2', transform=transform, proximity=30, train=False)

    a=0