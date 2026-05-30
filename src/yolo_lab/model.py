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
    
class DetectionHead (nn.Module):
    def __init__  (self, channels, num_classes, head_channels, levels = 3):
        super().__init__ ()
        self.cls_towers = nn.ModuleList ()
        self.box_towers = nn.ModuleList ()
        self.cls_preds = nn.ModuleList ()
        self.obj_preds = nn.ModuleList ()
        
        for _ in range (levels):
            self.cls_towers.append (nn.Sequential (ConvNormAct (channels, head_channels), ConvNormAct (head_channels, head_channels)))
            self.box_towers.append (nn.Sequential (ConvNormAct (channels, head_channels), ConvNormAct (head_channels, head_channels)))
            self.cls_preds.append (nn.Conv2d (head_channels, 1, 1))
            self.obj_preds.append (nn.Conv2d (head_channels, 4, 1))
            
        self._init_biases ()
        
    def _init_biases (self):
        prior = 0.01
        bias = -torch.log (torch.tensor ((1 - prior) / prior)).item ()
        
        for pred in [*self.cls_preds, *self.obj_preds]:
            nn.init.constant_ (pred.bias, bias)
            
    def forward (self, features):
        cls_out = []
        obj_out = []
        box_out = []
        
        for i, feature in enumerate (features):
            cls_feat = self.cls_towers [i] (feature)
            box_feat = self.box_towers [i] (feature)
            cls_out.append (self.cls_preds [i] (cls_feat))
            obj_out.append (self.obj_preds [i] (cls_feat))
            box_out.append (self.box_preds [i] (box_feat))
            
        return {"cls": cls_out, "obj": obj_out, "box": box_out}
    
class YoloStyleDetector (nn.Module):
    def __init__ (self, num_classes, backbone_name = "resnet34", pretrained_backbone = True, fpn_channels = 128, head_channels = 128, freeze_backbone_epochs = 0,):
        super ().__init__ ()
        self.backbone = ResNetFeatureBackbone (backbone_name, pretrained_backbone)
        self.fpn = FeaturePyramidNetwork (self.backbone.out_channels, fpn_channels)
        self.head = DetectionHead (fpn_channels, num_classes, head_channels)
        self.strides = [8, 16, 32]
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
    def Set_Backbone_Trainable (self, trainable):
        for param in self.backbone.parameters ():
            param.requires_grad_ (trainable)
            
    def Apply_Freeze_Schedule (self, epoch):
        if self.freeze_backbone_epochs > 0:
            self.Set_Backbone_Trainable (epoch >= self.freeze_backbone_epochs)
            
    def forward (self, x):
        c3, c4, c5 = self.backbone (x)
        fpn_out = self.fpn ({"p3": c3, "p4": c4, "p5": c5})
        features = [fpn_out ["p3"], fpn_out ["p4"], fpn_out ["p5"]]
        out = self.head (features)
        out ["strides"] = self.strides
        return out
    
def Build_Model (cfg, pretrained = None):
    model_cfg = cfg ["model"]
    pretrained_backbone = bool (model_cfg.get ("pretrained_backbone", True) if pretrained is None else pretrained)
    
    return YoloStyleDetector (
        num_classes = int (model_cfg ["num_classes"]),
        backbone_name = str (model_cfg.get ("backbone", "resnet34")),
        pretrained_backbone = pretrained_backbone,
        fpn_channels = int (model_cfg.get ("fpn_channels", 128)),
        head_channels = int (model_cfg.get ("head_channels", 128)),
        freeze_backbone_epochs = int (model_cfg.get ("freeze_backbone_epochs", 0)),
    )