from lightly.models.modules import BarlowTwinsProjectionHead
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights
from torch import nn
import torch

class Tempo(nn.Module):
    """
    Tempo module consisting of a ResNet-34 feature extractor and a projection 
    head. Used for self supervised training.
    """

    def __init__(self, pretrain:bool=True, embedding_dim:int=1024) -> None:
        super(Tempo, self).__init__()
        
        if pretrain:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BarlowTwinsProjectionHead(512, embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
class Baseline(nn.Module):
    """
    Baseline module used for evaluation (ResNet-34 FE + linear head). 
    """

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
    """
    Tempo  module used for evaluation (ResNet-34 FE + linear head).
    Can take pretrained weights. 
    """

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