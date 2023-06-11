import argparse
from tqdm import tqdm

import numpy as np

import torch
from torchvision import transforms as T
from lightly.loss import BarlowTwinsLoss

from tempo2.models import Tempo, Baseline, TempoLinear
from tempo2.data.datasets import video_dataset, finetune_dataset

from linear_eval import linear_eval

from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    losses = []
    for image, image_d, _, _ in tqdm(dataloader):
        image = image.to(device)
        image_d = image_d.to(device)
        
        z0 = model(image)
        z1 = model(image_d)
        loss = criterion(z0, z1)
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = torch.tensor(losses).mean()
    return avg_loss

def train(epochs, lr, l, train_loader, pretrain, device):
    model = Tempo(pretrain=pretrain).to(device)
    criterion = BarlowTwinsLoss(lambda_param=l)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-5},
        {"params": model.projection_head.parameters(), "lr": 1e-5}
    ])

    for epoch in range(epochs):
        print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(epoch, loss)

    return model.backbone.state_dict()

def main(args):
    # Parse commandline-arguments
    epochs = args.epochs if args.epochs else 1
    lr = args.lr if args.lr else 1e-3
    l = args.l if args.l else 1e-3
    evaluation = args.eval
    baseline = args.baseline if args.baseline else False
    proximity = args.proximity if args.proximity else 30
    save_model = args.save_model

    # Load datasets
    train_loader = video_dataset(proximity=proximity, batch_size=200)
    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=10)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # Parameters for finetuning
    num_runs = 3
    iterations = 3_000

    # Choose model
    if baseline:
        model = Baseline(out_features=24, pretrain=True).to(device)
    else:
        weights = train(epochs, lr, l, train_loader, pretrain=True, device=device)
        model = TempoLinear(weights, out_features=24).to(device)

    # Train model 
    if evaluation:
        e = []
        for i in tqdm(range(num_runs)):
            _, errors, iters = linear_eval(iterations, model, train_loader_ft, test_loader_ft, device)
            e.append(errors.reshape(1,-1))
        e = np.concatenate(e, axis=0)
        e_mean = e.mean(axis=0)
    
    else:
        e = []

    # Write to tensorboard
    writer = SummaryWriter()
    for i in np.arange(len(e_mean)):
        writer.add_scalar('accuracy', e_mean[i], iters[i])
    writer.close()

    # Save model weights
    if save_model:
        torch.save(model.state_dict(), f"model_zoo/{save_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--l', type=float, required=False)
    parser.add_argument('--eval', type=bool, required=False)
    parser.add_argument('--baseline', type=bool, required=False)
    parser.add_argument('--proximity', type=int, required=False)
    parser.add_argument('--save_model', type=str, required=False)

    args = parser.parse_args()

    main(args)