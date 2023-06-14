import os
from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tempo2.data.pdfs import p_uni

class VideoDataset(Dataset):
    def __init__(
            self, 
            path, 
            transform=None, 
            proximity:int=3, 
            pdf:Callable[..., int]=p_uni
        ) -> None:
        
        self.p = proximity
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if not p.endswith('.txt')])
        self.pdf = pdf

        print(self.pdf)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. get one element x
        image = Image.open(self.image_paths[index])

        # 2. sample element x' in the neighbourhood of x
        pb = self.pdf(index, self.p, self.__len__())
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