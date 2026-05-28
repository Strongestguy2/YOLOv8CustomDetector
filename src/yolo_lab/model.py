from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

class ConvNormAct (nn.Sequential):
    def __init__ (self, in_channels, out_channels, kernel_size = 3):
        padding = kernel_size // 2
        groups = min (32, out_channels)
        
        while out_channels % groups != 0:
            groups -= 1
            
        super ().__init__ (
            nn.Conv2d (in_channels, out_channels, kernel_size, padding = padding, bias = False),
            nn.GroupNorm (groups, out_channels),
            nn.SiLU (inplace = True),
        )

class ResNetFeatureBackbone (nn.Module):
    def __init__ (self, name, pretrained):
        super ().__init__ ()
        
        if name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34 (weights = weights)
            channels = [128, 256, 512]
        elif name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50 (weights = weights)
            channels = [512, 1024, 2048]
        else:
            raise ValueError (f"Unsupported backbone: {name}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
            
        self.body = create_feature_extractor (backbone, return_nodes = {"layer2": "p3", "layer3": "p4", "layer4": "p5"})
        self.out_channels = channels
        
    def forward (self, x):
        out = self.body (x)
        return [out ["p3"], out ["p4"], out ["p5"]]
    


        