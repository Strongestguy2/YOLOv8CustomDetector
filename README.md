# YOLOv8CustomDetector

This is my custom YOLO v8 Style detector that have a good thing of a nice UI Because everyone love a nice UI
It does not have any Ultralytics dependency.
YOLO-format labels
Validation metrics
Image previews
A small upload demo

## Architecture

Backbone: ResNet
Neck: FPN
Head: Dense objectness/class/box heads

## Features

- Custom YOLO-format datasets with configurable class names.
- ResNet-34 or ResNet-50 backbone with an FPN detection head.
- Resume-safe training with `last.pt`, `best.pt`, and `safe_stop.pt` checkpoints.
- COCO-style AP/precision/recall evaluation.
- Prediction CLI for a single image, folder, or glob pattern.
- Gradio image upload demo with confidence and NMS controls.
- Training preview images that compare predictions against targets.

## Demo

https://strongestguy2.github.io/YOLOv8CustomDetector/

## Install

Windows users can double-click:

```text
start_panel.bat
```

On first launch it:
1. Creates `.venv and install the application
2. Opens the pane; at `http://127.0.0.1:7860`

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\panel.py
```

## First Run

1. Open **Configure** and set your class names and dataset root.
2. Save the configuration.
3. Open **Dataset** and validate it.
4. Open **Train**, choose **Smoke test**, and run one step to check the installation.
5. Start a new training session with your configuration.
6. Select the generated `best.pt` in **Predict** or **Evaluate**.

## Offline Walkthrough

Click **Create tiny synthetic demo and config** in the Dataset tab. It creates one generated training image and one validation image.

## Custom Dataset

Use the standard YOLO detection layout:

```text
data/my_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image has a matching `.txt` label file. Each row is:

```text
class_id center_x center_y width height
```

Class IDs are zero-based. 

The default [custom configuration](configs/custom.yaml) does not download anything. Class names entered in the panel determine `model.num_classes` and are preserved in annotations and exports.

## Optionl COCO Baseline
As this project is originally designed for COCO, COCO is optional and isolated from the core installation.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-coco.txt
.\.venv\Scripts\python.exe scripts\panel.py
```

Then select `configs/coco_resnet34.yaml` in the panel. 

## Automation
For those that actually want to work inside cli rather than the ui despite this whole project is optimised for a no code workflow:

```powershell
# Tiny smoke run
.\.venv\Scripts\python.exe scripts\create_demo_dataset.py
.\.venv\Scripts\python.exe scripts\universal_train.py --smoke --max-steps 1

# Validation and training custom data
.\.venv\Scripts\python.exe scripts\validate_dataset.py --config configs\custom.yaml --write-dataset-yaml
.\.venv\Scripts\python.exe scripts\universal_train.py --config configs\custom.yaml

# Prediction
.\.venv\Scripts\python.exe scripts\predict.py --config configs\custom.yaml --weights outputs\runs\my_run\checkpoints\best.pt --source images --recursive --save --json-output outputs\predictions\results.json

# Evaluate a checkpoint
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\custom.yaml --weights outputs\runs\my_run\checkpoints\best.pt --detection-metrics --json-output outputs\evaluations\metrics.json
```

Training writes to `outputs/runs/<run_name>/`:

- `checkpoints/last.pt` — latest resumable state.
- `checkpoints/best.pt` — best validation-loss state.
- `checkpoints/safe_stop.pt` — interruption-safe state.
- `config.yaml` — resolved run configuration.
- `train_log.csv` — step-by-step training metrics.
- `visuals/` — prediction-vs-target previews.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe scripts\verify_pages_demo.py
```



