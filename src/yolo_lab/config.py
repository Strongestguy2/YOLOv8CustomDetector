from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import yaml
from .constants import COCO_CLASSES

def Deep_Merge (base, override):
    merged = copy.deepcopy (base)
    
    for key, value in override.items ():
        if isinstance (value, dict) and isinstance (merged.get (key), dict):
            merged [key] = Deep_Merge (merged [key], value)
        else:
            merged [key] = copy.deepcopy (value)
        
    return merged

def Load_Config (path):
    config_path = Path (path)
    
    with config_path.open ("r", encoding = "utf-8") as f:
        cfg = yaml.safe_load (f) or {}
        
    Normalise_Config (cfg)
    cfg ["_config_path"] = str (config_path.resolve ())
    return cfg

def Save_Config (cfg, path):
    out = Path (path)
    out.parent.mkdir (exist_ok = True, parents = True)
    serialisable = {k: v for k, v in cfg.items () if not k.startswith ("_")}
    
    with out.open ("w", encoding = "utf-8") as f:
        yaml.safe_dump (serialisable, f, sort_keys = False)
        
def Get_Config (cfg, dotted, default = None):
    current = cfg
    
    for part in dotted.split ("."):
        if not isinstance (current, dict) or part not in current:
            return default
        
        current = current [part]
        
    return current

def Set_Config (cfg, dotted, value):
    current = cfg
    parts = dotted.split (".")
    
    for part in parts [:-1]:
        current [part] = current.setdefault (part, {})
                
    current [parts [-1]] = value
    
def Normalise_Config (cfg):
    model_cfg = cfg.setdefault ("model", {})
    class_names = Class_Names_From_Config (cfg)
    
    if class_names is not None:
        configured_num_classes = model_cfg.get ("num_classes")
        
        if configured_num_classes is not None and int (configured_num_classes) != len (class_names):
            raise ValueError (f"Configured num_classes ({configured_num_classes}) does not match the number of class names ({len (class_names)})!")
        
        model_cfg ["num_classes"] = len (class_names)
        model_cfg.setdefault ("class_names", class_names)
    elif "num_classes" in model_cfg:
        model_cfg ["num_classes"] = int (model_cfg ["num_classes"])
        
    return cfg

def Get_Class_Names (cfg):
    class_names = Class_Names_From_Config (cfg)
    
    if class_names is not None:
        return class_names
    
    num_classes = int (cfg.get ("model", {}).get ("num_classes", 0) or 0)
    
    if num_classes == len (COCO_CLASSES):
        return list (COCO_CLASSES)
    
    return [f"class_{idx}" for idx in range (num_classes)]
    
    if class_names is not None:
        return class_names
def Class_Names_From_Config (cfg):
    model_cfg = cfg.get ("model", {})
    raw = cfg.get ("classes", model_cfg.get ("class_names", cfg.get ("names")))
    
    if raw is None:
        return None
    
    return Parse_Class_Names (raw)

def Parse_Class_Names (raw):
    if isinstance (raw, Mapping):
        try:
            ordered = [raw [key] for key in sorted (raw, key = lambda item: int (item))]
        except (ValueError, TypeError):
            ordered = [raw [key] for key in sorted (raw)]
            
        return [str (name) for name in ordered]
    
    if isinstance (raw, Sequence) and not isinstance (raw, (str, bytes, bytearray)):
        return [str (name) for name in raw]
    
    raise TypeError ("Class names must be a lsit or mapping of cls id to class name!!!")