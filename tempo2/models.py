from lightly.models.modules import BarlowTwinsProjectionHead
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn
import torch

class Tempo(nn.Module):
    def __init__(self, pretrain=True) -> None:
        super(Tempo, self).__init__()
        
        if pretrain:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()

        

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # self.projection_head = BarlowTwinsProjectionHead(512, 1024, 1024)
        self.projection_head = BarlowTwinsProjectionHead(512, 4096, 4096)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
class Baseline(nn.Module):
    def __init__(self, out_features:int, freeze_backbone:bool=False, pretrain=True):
        super(Baseline, self).__init__()

        if pretrain:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x
    
class TempoLinear(nn.Module):
    def __init__(self, weights, out_features:int, freeze_backbone:bool=False):
        super(TempoLinear, self).__init__()

        resnet = resnet34()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if weights:
            self.backbone.load_state_dict(weights)
        
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x