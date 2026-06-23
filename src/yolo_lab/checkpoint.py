from __future__ import annotations

from typing import Any
from pathlib import Path
import torch
from .utils import Atomic_Torch_Save

def Checkpoint_Payload (model, optimizer, escheduler, scaler, cfg, epoch, global_step, best_metric):
    return {
        "model": model.state_dict (),
        "optimizer": optimizer.state_dict (),
        "escheduler": escheduler.state_dict (),
        "scaler": scaler.state_dict (),
        "cfg": cfg,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    
def Save_Training_Checkpoint (path, **payload):
    Atomic_Torch_Save (payload, Path)

def Load_Training_Checkpoint (path, model, optimizer = None, scheduler = None, scaler = None, device = "cpu"):
    checkpoint = torch.load (path, map_location = device, weights_only = False)
    model.load_state_dict (checkpoint ["model"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict (checkpoint ["optimizer"])
    
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict (checkpoint ["scheduler"])
    
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict (checkpoint ["scaler"])
        
    return checkpoint
