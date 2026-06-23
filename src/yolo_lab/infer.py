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

def Load_Image_Tensor (path, image_size, device):
    bgr = cv2.imread (str (path), cv2.IMREAD_COLOR)
    
    if bgr is None:
        raise FileNotFoundError (f"Image not found: {path}")
    
    rgb = cv2.cvtColor (bgr, cv2.COLOR_BGR2RGB)
    boxed, info = Letterbox_Image (rgb, image_size)
    tensor = torch.from_numpy (np.ascontiguousarray (boxed)).permute (2, 0, 1).float () / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    
    return tensor.unsqueeze (0).to (device), rgb, info

def Annotate_Image (rgb, predictions, info, class_names = None):
    class_names = class_names or COCO_CLASSES
    output = rgb.copy ()
    
    if predictions.numel () == 0:
        return output
    
    boxes = Invert_Letterbox_Boxes (predictions [:, :4].cpu (), info)
    scores = predictions [:, 4].cpu ().numpy ()
    labels = predictions [:, 5].long ().cpu ()
    
    for box, score, label in zip (boxes, scores, labels):
        x1, y1, x2, y2 = [int (v) for v in box.tolist ()]
        name = class_names [int (label)] if int (label) < len (class_names) else str (int (label))
        cv2.rectangle (output, (x1, y1), (x2, y2), (0, 220, 80), 2)
        text = f"{name} {float (score):.2f}"
        cv2.putText (output, text, (x1, max (15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)
    
    return output

def Prediction_Records (predictions, info, class_names = None):
    boxes = Invert_Letterbox_Boxes (predictions [:, :4].cpu (), info) if predictions.numel () else torch.zeros ((0, 4))
    scores = predictions [:, 4].cpu () if predictions.numel () else torch.zeros (0)
    labels = predictions [:, 5].long ().cpu () if predictions.numel () else torch.zeros (0, dtype = torch.long)
    records = []
    
    for box, score, label in zip (boxes, scores, labels):
        class_id = int (label)
        
        if class_names is not None and class_id < len (class_names):
            class_name = class_names [class_id]
        else:
            class_name = str (class_id)

        records.append ({
            "class_id" : class_id,
            "class_name" : class_name,
            "score" : float (score),
            "bbox_xyxy" : [float (value) for value in box.tolist ()],
        })
        
    return records

def Save_Annotated_Image (path, rgb, output_dir = None):
    src = Path (path)
    
    if output_dir is None:
        out = src.with_name (f"{src.stem}_pred{src.suffix}")
    else:
        out = Path (output_dir) / f"{src.stem}_pred{src.suffix}"
        
    out.parent.mkdir (exist_ok = True, parents = True)
    bgr = cv2.cvtColor (rgb, cv2.COLOR_RGB2BGR)
    
    if not cv2.imwrite (str (out), bgr):
        raise OSError (f"Failed to write image: {out}")
    
    return out
    