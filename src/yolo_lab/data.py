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