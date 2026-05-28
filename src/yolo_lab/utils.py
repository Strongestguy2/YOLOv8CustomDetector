from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

def Set_Seed (seed):
    random.seed (seed)
    np.random.seed (seed)
    torch.manual_seed (seed)
    
    if torch.cuda.is_available ():
        torch.cuda.manual_seed_all (seed)
        
    torch.backends.cudnn.benchmark = True

def Get_Device (requested = "auto"):
    if requested == "auto":
        return torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    
    return torch.device (requested)

def Atomic_Torch_Save (payload, path):
    destination = Path (path)
    destination.parent.mkdir (exist_ok = True, parents = True)
    tmp = destination.with_suffix (destination.suffix + ".tmp")
    torch.save (payload, tmp)
    os.replace (tmp, destination)
    
def Atomic_Json_Dump (payload, path):
    destination = Path (path)
    destination.parent.mkdir (exist_ok = True, parents = True)
    tmp = destination.with_suffix (destination.suffix + ".tmp")
    
    with tmp.open ("w", encoding = "utf-8") as f:
        json.dump (payload, f, indent = 2)
        
    os.replace (tmp, destination)
    
def Json_Load (path, default):
    p = Path (path)
    
    if not p.exists ():
        return default
    
    with p.open ("r", encoding = "utf-8") as f:
        return json.load (f)
    
class Stopwatch:
    def __init__ (self):
        self.start = time.time ()
    
    @property
    def Elapsed (self):
        return time.time () - self.start
    
    def Exceeded_Hours (self, hours):
        return hours is not None and self.Elapsed >= hours * 3600