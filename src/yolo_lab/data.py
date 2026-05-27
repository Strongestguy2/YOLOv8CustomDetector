from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .constants import COCO_CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor ([0.485, 0.456, 0.406]).view (3, 1, 1)
IMAGENET_STD = torch.tensor ([0.229, 0.224, 0.225]).view (3, 1, 1)

@dataclass
class LetterboxInfo:
    ratio : float
    pad_x : int
    pad_y : int
    original_width : int
    original_height : int
    image_size : int

def Letterbox_Image (image, image_size, colour = 114):
    h, w = image.shape [:2]
    ratio = min (image_size / h, image_size / w)
    new_w = int (round (w * ratio))
    new_h = int (round (h * ratio))
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2
    resized = cv2.resize (image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
    canvas = np.full ((image_size, image_size, 3), colour, dtype = np.uint8)
    canvas [pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    
    return canvas, LetterboxInfo (ratio, pad_x, pad_y, w, h, image_size)

def Transform_Boxes_To_Letterbox (boxes, info):
    if boxes.numel () == 0:
        return boxes.reshape (0, 4)
    
    out = boxes.clone ().float ()
    out [:, [0, 2]] = out [:, [0, 2]] * info.ratio + info.pad_x
    out [:, [1, 3]] = out [:, [1, 3]] * info.ratio + info.pad_y
    
    return out.clamp_ (0, info.image_size)

def Invert_Letterbox_Boxes (boxes, info):
    if boxes.numel () == 0:
        return boxes.reshape (0, 4)
    
    out = boxes.clone ().float ()
    out [:, [0, 2]] = (out [:, [0, 2]] - info.pad_x) / info.ratio
    out [:, [1, 3]] = (out [:, [1, 3]] - info.pad_y) / info.ratio
    out [:, [0, 2]] = out [:, [0, 2]].clamp_ (0, info.original_width)
    out [:, [1, 3]] = out [:, [1, 3]].clamp_ (0, info.original_height)
    
    return out

def Augment_HSV_RGB (image, gain_h, gain_s, gain_v):
    hsv = cv2.cvtColor (image, cv2.COLOR_RGB2HSV).astype (np.float32)
    hsv [..., 0] = (hsv [..., 0] + (random.random () * 2 - 1) * gain_h * 179) % 179
    hsv [..., 1] *= 1 + (random.random () * 2 - 1) * gain_s
    hsv [..., 2] *= 1 + (random.random () * 2 - 1) * gain_v
    hsv [..., 1] = np.clip (hsv [..., 1:], 0, 255)
    
    return cv2.cvtColor (hsv.astype (np.uint8), cv2.COLOR_HSV2RGB)

def Parse_Yolo_Label_File (label_path, original_width, original_height, num_classes, strict = False,):
    path = Path (label_path)
    labels = []
    boxes = []
    
    if not path.exists ():
        return torch.zereos (0, dtype = torch.long), torch.zeros ((0, 4), dtype = torch.float32)
    
    for line_number, raw in enumerate (path.read_text (encoding = "utf-8", errors = "ignore").splitlines (), start = 1):
        line = raw.strip ()
        
        if not line:
            continue
        
        parts = line.split ()
        
        try:
            if len (parts) != 5:
                raise ValueError ("Expected 5 fields")
                
            cls_f, cx, cy, bw, bh = map (float, parts)
            cls = int (cls_f)
            
            if cls != cls_f or cls < 0 or cls >= num_classes:
                if strict:
                    raise ValueError (f"class {cls_f} out of range")
                
                if any (not math.isfinite (v) or v < 0.0 or v > 1.0 for v in (cx, cy, bw, bh)):
                    raise ValueError ("normalised coor must be in [0, 1]")
                
                x1 = (cx - bw / 2) * original_width
                y1 = (cy - bh / 2) * original_height
                x2 = (cx + bw / 2) * original_width
                y2 = (cy + bh / 2) * original_height
                
                if x2 <= x1 or y2 <= y1:
                    raise ValueError ("Boxes has negative area")
                
        except ValueError:
            if strict:
                raise ValueError (f"Invalid label at {path} : {line_number} : {raw}") from None
            continue
        
        labels.append (cls)
        boxes.append ([x1, y1, x2, y2])
        
    if not labels:
        return torch.zeros (0, dtype = torch.long), torch.zeros ((0, 4), dtype = torch.float32)
    
    return torch.tensor (labels, dtype = torch.long), torch.tensor (boxes, dtype = torch.float32)

class YoloDetectionDataset (Dataset):
    def __init__ (self, image_dir, label_dir, image_size, num_classes, augment = False, normalise = True, hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4, hflip_p = 0.5, max_images = None, subset_seed = None):
        self.image_dir = Path (image_dir)
        self.label_dir = Path (label_dir)
        self.image_size = int (image_size)
        self.num_classes = int (num_classes)
        self.augment = augment
        self.normalise = normalise
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.hflip_p = hflip_p
        
        if not self.image_dir.exists ():
            raise FileNotFoundError (f"Image directory {self.image_dir} not found")
        
        self.image_files = sorted (p for p in self.image_dir.iterdir () if p.suffix.lower () in IMAGE_EXTENSIONS)
        
        if max_images is not None and max_images > 0 and len (self.image_files) > max_images:
            if subset_seed is not None:
                if subset_seed is None:
                    self.image_files = self.image_files [:max_images]
                else:
                    rng = random.Random (subset_seed)
                    self.image_files = sorted (rng.sample (self.image_files, max_images))
                    
    def __len__ (self):
        return len (self.image_files)
    
    def __getitem__ (self, index):
        image_path = self.image_files [index]
        bgr = cv2.imread (str (image_path), cv2.IMREAD_COLOR)
        
        if bgr is None:
            image = np.full ((self.image_size, self.image_size, 3), 114, dtype = np.uint8)
            tensor = torch.from_numpy (image).permute (2, 0, 1).float () / 255.0
            
            return self._normalise (tensor), self._empty_target ()
        
        image = cv2.cvtColor (bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape [:2]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        labels, boxes = Parse_Yolo_Label_File (label_path, w, h, self.num_classes)
        
        if self.augment:
            image = Augment_HSV_RGB (image, self.hsv_h, self.hsv_s, self.hsv_v)
            
            if random.random () < self.hflip_p:
                image = np.ascontiguousarray (image [:, ::-1])
                
                if boxes.numel () > 0:
                    x1 = boxes [:, 0].clone ()
                    x2 = boxes [:, 2].clone ()
                    boxes [:, 0] = w - x2
                    boxes [:, 2] = w - x1
                    
        image, info = Letterbox_Image (image, self.image_size)
        boxes = Transform_Boxes_To_Letterbox (boxes, info)
        valid = (boxes [:, 2] > boxes [:, 0]) & (boxes [:, 3] > boxes [:, 1]) if boxes.numel () > 0 else torch.zeros (0, dtype = torch.bool)
        boxes = boxes [valid]
        labels = labels [valid]
        
        tensor = torch.from_numpy (np.ascontiguousarray (image)).permute (2, 0, 1).float () / 255.0
        
        return self._normalise (tensor), {"boxes" : boxes, "labels" : labels}
    
    def _empty_target (self):
        return {"boxes" : torch.zeros ((0, 4), dtype = torch.float32), "labels" : torch.zeros (0, dtype = torch.long)}
    
    def _normalise (self, tensor):
        if not self.normalise:
            return tensor
        
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD
    
def Collate_Fn (batch):
    images, targets = zip (*batch)
    batch_indices = []
    boxes = []
    labels = []
    
    for i, target in enumerate (targets):
        n = len (target ["labels"])
        boxes.append (target ["boxes"])
        labels.append (target ["labels"])
        batch_indices.append (torch.full ((n,), i, dtype = torch.long))
        
    return torch.stack (list (images), 0), {
        "boxes" : torch.cat (boxes, 0) if boxes else torch.zeros ((0, 4), dtype = torch.float32),
        "labels" : torch.cat (labels, 0) if labels else torch.zeros (0, dtype = torch.long),
        "batch_indices" : torch.cat (batch_indices, 0) if batch_indices else torch.zeros (0, dtype = torch.long),
    }
    
def Validate_Yolo_Dataset (root, num_classes):
    data_root = Path (root)
    result = {"root" : str (data_root), "splits" : {}, "errors" : []}
    
    for split in ("train", "val"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        split_result = {"images" : 0, "label_files" : 0, "boxes" : 0, "missing_labels" : 0}
        
        if not image_dir.exists ():
            result ["errors"].append (f"Image directory {image_dir} not found")
            result ["splits"] [split] = split_result
            continue
        
        for image_path in sorted (p for p in image_dir.iterdir () if p.suffix.lower () in IMAGE_EXTENSIONS):
            split_result ["images"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists ():
                split_result ["missing_labels"] += 1
                continue
            
            split_result ["label_files"] += 1
            raw = cv2.imread (str (image_path), cv2.IMREAD_COLOR)
            
            if raw is None:
                result ["errors"].append (f"Failed to read image {image_path}")
                continue
            
            labels, boxes = Parse_Yolo_Label_File (label_path, raw.shape [1], raw.shape [0], num_classes, strict = True)
            split_result ["boxes"] += len (labels)
            
            if boxes.numel () and ((boxes [:, 2] <= boxes [:, 0]) | (boxes [:, 3] <= boxes [:, 1])).any ():
                result ["errors"].append (f"Invalid box area in {label_path}")
        
        result ["splits"][split] = split_result
        
    return result


