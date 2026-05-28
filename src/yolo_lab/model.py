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
        

        