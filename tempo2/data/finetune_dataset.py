import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class FinetuneDataset(Dataset):
    def __init__(self, path, transform=None, train=True, samples_pc=None) -> None:
        """
        Args:
            path (str): Path to image dataset,
            transform : Transformations applied to the samples, 
            train (bool): Creates training set if true, test set if false,
            samples_pc (int): Number of training samples per class
        """

        self.transform = transform

        class_map = {l:i for l, i in zip('ABCDEFGHIKLMNOPQRSTUVWXY', range(24))}

        self.image_paths = []
        
        if train:
            if samples_pc == None:
                split = 'train'
            else:
                split = 'train-' + str(samples_pc)
        else:
            split = 'test'

        for c in os.listdir(os.path.join(path, split)):
            for name in os.listdir(os.path.join(path, split,c)):
                p = os.path.join(path, split, c, name)
                self.image_paths.append((p, class_map[c]))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns samples of the form (image, label)
        """

        pth, cls = self.image_paths[index]
        image = Image.open(pth)

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(cls))

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    dataset = FinetuneDataset('./datasets/finetune', transform=transform, train=False)

    a=0