from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

def Focal_BCE_With_Logits (logits, targets, alpha, gamma):
    bce = F.binary_cross_entropy_with_logits (logits, targets, reduction = "none")
    prob = torch.sigmoid (logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    
    return alpha_factor * (1 - p_t).pow (gamma) * bce

def Make_Centers (h, w, stride, device):
    ys, xs = torch.meshgrid (torch.arange (h, device = device), torch.arange (w, device = device), indexing = "ij")
    
    return torch.stack (((xs + 0.5) * stride, (ys + 0.5) * stride), dim = -1)

def Decode_LetterBox (distances, stride):
    b, _, h, w = distances.shape
    centers = Make_Centers (h, w, stride, distances.device).view (1, h, w, 2)
    d = distances.permute (0, 2, 3, 1) * stride
    x1 = centers [..., 0] - d [..., 0]
    y1 = centers [..., 1] - d [..., 1]
    x2 = centers [..., 0] + d [..., 2]
    y2 = centers [..., 1] + d [..., 3]
    
    return torch.stack ((x1, y1, x2, y2), dim = -1).view (b, h, w, 4)

class YoloLoss (nn.Module):
    def __init__ (self, num_classes, image_size, strides, obj_weight = 1.0, cls_weight = 1.0, box_weight = 5.0, l1_weight = 0.25, focal_alpha = 0.25, focal_gamma = 2.0, small_box = 64, medium_box = 192, center_radius = 0):
        super().__init__ ()
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.l1_weight = l1_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.small_box = small_box
        self.medium_box = medium_box
        self.center_radius = max (0, int (center_radius))
        
    def forward (self, outputs, targets):
        cls_logits = outputs ["cls"]
        obj_logits = outputs ["obj"]
        box_preds = outputs ["box"]
        target_maps = self._build_targets (outputs, targets)
        
        obj_loss = torch.zeros ((), device = obj_logits [0].device)
        cls_loss = torch.zeros ((), device = cls_logits [0].device)
        box_loss = torch.zeros ((), device = box_preds [0].device)
        l1_loss = torch.zeros ((), device = box_preds [0].device)
        pos_total = 0
        
        for level, stride in enumerate (self.strides):
            obj_t = target_maps [level] ["obj"]
            cls_t = target_maps [level] ["cls"]
            box_t = target_maps [level] ["box"]
            pos = obj_t [:, 0] > 0.5
            pos_count = int (pos.sum ().item ())
            pos_total += pos_count
            
            obj_loss = obj_loss + Focal_BCE_With_Logits (obj_logits [level], obj_t, self.focal_alpha, self.focal_gamma).mean ()
            
            if pos_count == 0:
                continue
            
            cls_p = cls_logits [level].permute (0, 2, 3, 1) [pos]
            cls_target = cls_t.permute (0, 2, 3, 1) [pos]
            cls_loss = cls_loss + Focal_BCE_With_Logits (cls_p, cls_target, self.focal_alpha, self.focal_gamma).mean ()
            
            pred_dist = box_preds [level].permute (0, 2, 3, 1) [pos]
            target_dist = box_t.permute (0, 2, 3, 1) [pos]
            l1_loss = l1_loss + F.smooth_l1_loss (pred_dist, target_dist, reduction = "mean")
            
            pred_boxes = Decode_LetterBox (box_preds [level], stride) [pos]
            target_boxes = target_maps [level] ["xyxy"].permute (0, 2, 3, 1) [pos]
            box_loss = box_loss + generalized_box_iou_loss (pred_boxes, target_boxes, reduction = "mean")
            
        total = self.obj_weight * obj_loss + self.cls_weight * cls_loss + self.box_weight * box_loss + self.l1_weight * l1_loss
        stats = {
            "loss" : float (total.detach ().item ()),
            "loss_obj" : float (obj_loss.detach ().item ()),
            "loss_cls" : float (cls_loss.detach ().item ()),
            "loss_box" : float (box_loss.detach ().item ()),
            "loss_l1" : float (l1_loss.detach ().item ()),
            "num_pos" : float (pos_total),
            "num_gt" : float (len (targets ["labels"])),
        }
        
    def _target_level (self, box):
        max_side = float (torch.max (box [2:] - box [:2]).item ())
        
        if max_side < self.small_box:
            return 0
        if max_side < self.medium_box:
            return 1
        return 2
    