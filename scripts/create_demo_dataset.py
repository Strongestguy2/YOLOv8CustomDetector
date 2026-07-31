from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path ()

from yolo_lab.data import write_dataset_yaml
from yolo_lab.smoke import ensure_tiny_detection_dataset

def main ():
    parser = argparse.ArgumentParser (description = "Create a demo dataset for the YOLOv8CustomDetector")
    parser.add_argument ("--root", default = "data/demo_yolo")
    parser.add_argument ("--image-size", type = int, default = 128)
    args = parser.parse_args ()
    
    root = ensure_tiny_detection_dataset (args.root, image_size = args.image_size)
    dataset_yaml = write_dataset_yaml (root, ["shape"])
    
    print (f"Demo dataset : {root.resolve ()}")
    print (f"Dataset YAML : {dataset_yaml.resolve ()}")
    
if __name__ == "__main__":
    main ()