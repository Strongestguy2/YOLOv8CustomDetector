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
    
    def _build_targets (self, outputs, targets):
        batch_size = outputs ["obj"][0].shape [0]
        device = outputs ["obj"][0].device
        target_maps = []
        
        for level, obj in enumerate (outputs ["obj"]):
            _, _, h, w = obj.shape
            target_maps.append ({
                "obj" : torch.zeros ((batch_size, 1, h, w), device = device),
                "cls" : torch.zeros ((batch_size, self.num_classes, h, w), device = device),
                "box" : torch.zeros ((batch_size, 4, h, w), device = device),
                "xyxy" : torch.zeros ((batch_size, 4, h, w), device = device),
                "area" : torch.full ((batch_size, 1, h, w), float ("inf"), device = device),
            })
            
        boxes = targets ["boxes"].to (device)
        labels = targets ["labels"].to (device)
        batch_idx = targets ["batch_idx"].to (device)
        
        for i in range (len (labels)):
            box = boxes [i].clamp (0, self.image_size)
            
            if box [2] <= box [0] or box [3] <= box [1]:
                continue
            
            label = int (labels [i].item ())
            level = self._target_level (box)
            stride = self.strides [level]
            _, _, h, w = outputs ["obj"][level].shape
            cx = (box [0] + box [2]) * 0.5
            cy = (box [1] + box [3]) * 0.5
            gx = int (torch.clamp (torch.floor (cx / stride), 0, w - 1).item ())
            gy = int (torch.clamp (torch.floor (cy / stride), 0, h - 1).item ())
            b = int (batch_idx [i].item ())
            area = (box [2] - box [0]) * (box [3] - box [1])
            
            for cell_x, cell_y in self._candidate_cells (box, gx, gy, w, h, stride):
                if area >= target_maps [level] ["area"][b, 0, cell_y, cell_x]:
                    continue
                
                center_x = (cell_x + 0.5) * stride
                center_y = (cell_y + 0.5) * stride
                letterbox = torch.tensor (
                    [center_x - box [0], center_y - box [1], box [2] - center_x, box [3] - center_y], 
                    device = device, 
                    dtype = torch.float32
                    ).clamp (min = 0.01) / stride
                
                target_maps [level]["obj"][b, 0, cell_y, cell_x] = 1.0
                target_maps [level]["cls"][b, :, cell_y, cell_x] = 0.0
                target_maps [level]["cls"][b, label, cell_y, cell_x] = 1.0
                target_maps [level]["box"][b, :, cell_y, cell_x] = letterbox
                target_maps [level]["xyxy"][b, :, cell_y, cell_x] = box
                target_maps [level]["area"][b, 0, cell_y, cell_x] = area
        
        return target_maps
    
    def _candidate_cells (self, box, gx, gy, w, h, stride):
        if self.center_radius <= 0:
            return [(gx, gy)]
        
        cells = []
        
        for cell_y in range (max (0, gy - self.center_radius), min (h - 1, gy + self.center_radius) + 1):
            center_y = (cell_y + 0.5) * stride
            
            for cell_x in range (max (0, gx - self.center_radius), min (w - 1, gx + self.center_radius) + 1):
                center_x = (cell_x + 0.5) * stride
                
                if box [0] <= center_x <= box [2] and box [1] <= center_y <= box [3]:
                    cells.append ((cell_x, cell_y))
                
        return cells or [(gx, gy)]