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

https://strongestguy2.github.io/Yolo-v8-With-COCO-2017-Dataset/

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

Use standard YOLO detection layout:

```text
data/custom_yolo/
  images/
    train/
      image_001.jpg
    val/
      image_101.jpg
  labels/
    train/
      image_001.txt
    val/
      image_101.txt
```

Each label row must be:

```text
class_id center_x center_y width height
```

All coordinates are normalized from `0` to `1`, and `class_id` is zero-based.

Copy or edit [configs/custom.yaml](configs/custom.yaml). Replace the `classes` list and point `data.root` at your dataset:

```yaml
classes:
  - scratch
  - dent
  - missing_part

data:
  root: data/custom_yolo
  auto_prepare: false
```

Validate the dataset before training:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py --config configs\custom.yaml --write-dataset-yaml
```

## Training

For custom data, keep COCO auto-preparation disabled and train from the custom config:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py --config configs\custom.yaml
```

Useful overrides:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py --config configs\custom.yaml --max-steps 100
.\.venv\Scripts\python.exe scripts\universal_train.py --config configs\custom.yaml --set train.batch_size=2 --set train.total_steps=20000
```

Outputs are written to `outputs/runs/<run_name>/`:

- `checkpoints/last.pt` latest resumable checkpoint.
- `checkpoints/best.pt` best validation-loss checkpoint.
- `train_log.csv` step-by-step training logs.
- `visuals/` prediction-vs-target preview images.
- `config.yaml` resolved run configuration.

## COCO Baseline

The default config stages a subset of COCO automatically through FiftyOne:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py --config configs\coco_resnet34.yaml --download-increment 1000
```

Increase `data.train_samples`, `data.val_samples`, or `--download-increment` as you scale the baseline.

## Inference

Run prediction on one image:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\custom.yaml --weights outputs\runs\custom_resnet34\checkpoints\best.pt --source path\to\image.jpg --save
```

Run prediction on a folder and save JSON:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\custom.yaml --weights outputs\runs\custom_resnet34\checkpoints\best.pt --source path\to\images --recursive --save --json-output outputs\predictions\predictions.json
```

Annotated images are saved under `outputs/predictions/` by default.

## Demo App

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --config configs\custom.yaml --weights outputs\runs\custom_resnet34\checkpoints\best.pt
```

Open `http://127.0.0.1:7860`, upload an image, and tune the confidence/NMS sliders.

## Evaluation

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\custom.yaml --weights outputs\runs\custom_resnet34\checkpoints\best.pt --detection-metrics --json-output outputs\eval\metrics.json
```

This reports validation loss, prediction counts, AP50, AP75, mAP50-95, precision, recall, and F1.

## Control Panel

On Windows, double-click:

```text
start_panel.bat
```

The panel opens in the browser and covers the main workflow without typing training commands:

- Edit, create, save, and delete YAML configs.
- Set classes, dataset root, run name, batch size, steps, image size, backbone, AMP, workers, and preview frequency.
- Validate YOLO-format datasets from the UI.
- Start, resume, stop, and monitor training.
- Hide branch checkpoint fields unless Branch mode is selected.
- Run predictions from a checkpoint against a configured image or folder source.
- Compare checkpoints with validation metrics.

You can also launch it manually:

```powershell
.\.venv\Scripts\python.exe scripts\panel.py
```

## Project Layout

```text
configs/          training configs and custom dataset template
scripts/          train, validate, predict, demo, evaluate, panel
src/yolo_lab/     model, loss, data, inference, metrics, checkpoints
tests/            smoke and regression tests
```

## Prototype Notes

This is a prototype detector. It is designed for quick iteration, readable code, and custom YOLO-format data. For serious benchmarking, train long enough on a real validation split, inspect `visuals/`, and use `scripts/evaluate.py --detection-metrics` before trusting the model.





