from torchvision import transforms as T

from torch.utils.data import DataLoader
from tempo2.data.video_dataset import VideoDataset
from tempo2.data.finetune_dataset import FinetuneDataset

transform = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_jup = T.Compose([
    T.Resize(128),
    T.ToTensor(),
])

transform50 = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def video_dataset50(batch_size=80, proximity=30, pdf=None):
    dataset = VideoDataset('../datasets/ASL-big/frames', transform=transform50, proximity=proximity, pdf=pdf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return dataloader

def finetune_dataset50(name='ASL-big', batch_size=80, train=True, samples_pc=None):
    dataset = FinetuneDataset(f'../datasets/{name}', transform=transform50, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return dataloader

def video_dataset(batch_size=80, proximity=30, pdf=None):
    dataset = VideoDataset('./datasets/ASL-big/frames', transform=transform, proximity=proximity, pdf=pdf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

    return dataloader

def finetune_dataset(name='ASL-big', batch_size=80, train=True, samples_pc=None, drop_last=False):
    dataset = FinetuneDataset(f'./datasets/{name}', transform=transform, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=2)

    return dataloader

def finetune_dataset_jup(name='ASL-big', batch_size=80, train=True, samples_pc=None, drop_last=False):
    dataset = FinetuneDataset(f'../../datasets/{name}', transform=transform_jup, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=2)

    return dataloader