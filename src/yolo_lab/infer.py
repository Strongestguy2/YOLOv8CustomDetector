from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from.constants import COCO_CLASSES
from .data import IMAGENET_MEAN, IMAGENET_STD, LetterboxInfo, Invert_Letterbox_Boxes, Letterbox_Image
from .loss import Decode_LetterBox

def Decode_Predictions (outputs, conf_threshold = 0.25, iou_threshold = 0.45, max_detections = 100, image_size = 640,):
    predictions = []
    batch_size = outputs ["obj"][0].shape [0]
    
    for b in range (batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for level, stride in enumerate (outputs ["strides"]):
            obj = outputs ["obj"][level][b].sigmoid ().flatten ()
            cls_prob = outputs ["cls"][level][b].sigmoid ().permute (1, 2, 0).reshape (-1, outputs ["cls"][level].shape [1])
            cls_score, cls_label = cls_prob.max (dim = 1)
            score = obj * cls_score
            keep = score > conf_threshold
            
            if not keep.any ():
                continue
            
            boxes = Decode_LetterBox (outputs ["box"][level], stride) [b].reshape (-1, 4)
            all_boxes.append (boxes [keep].clamp (0, image_size))
            all_scores.append (score [keep])
            all_labels.append (cls_label [keep])
            
        if not all_boxes:
            predictions.append (torch.zeros ((0, 6), device = outputs ["obj"][0].device))
            continue
        
        boxes = torch.cat (all_boxes)
        scores = torch.cat (all_scores)
        labels = torch.cat (all_labels)
        keep_idx = batched_nms (boxes, scores, labels, iou_threshold)[:max_detections]
        predictions.append (torch.cat ((boxes [keep_idx], scores [keep_idx, None], labels [keep_idx, None].float ()), dim = 1))
        
    return predictions


        
            