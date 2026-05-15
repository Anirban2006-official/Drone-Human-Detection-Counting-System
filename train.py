"""
Task 02: Model Training
========================
Fine-tune YOLOv8 on the VisDrone dataset for person & car detection.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime


# ── Training configuration ───────────────────────────────────────────────────
DEFAULT_TRAIN_CONFIG = {
    # Model
    "model": "yolov8m.pt",          # nano/small/medium/large/xlarge
    "data": "configs/visdrone.yaml",

    # Training
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,                    # reduce to 8 if GPU OOM
    "workers": 8,

    # Optimiser
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss weights
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Augmentations (tuned for drone imagery)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,

    # Output
    "project": "outputs/runs",
    "name": "visdrone_yolov8m",
    "exist_ok": True,
    "save": True,
    "save_period": 10,
    "val": True,
    "plots": True,

    # Other
    "device": "",        # '' = auto-select (cuda if available, else cpu)
    "patience": 30,      # early stopping patience
    "amp": True,         # automatic mixed precision
    "seed": 42,
    "verbose": True,
    "pretrained": True,
    "resume": False,
}


def build_data_yaml(dataset_root: str,
                    output_path: str = "configs/visdrone.yaml") -> str:
    """
    Write / update the dataset YAML so absolute paths are correct
    on the current machine.
    """
    cfg = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    2,
        "names": {0: "person", 1: "car"},
    }
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Dataset config written → {output_path}")
    return output_path


def train(config: dict | None = None) -> str:
    """
    Launch YOLOv8 training.

    Returns:
        Path to the best weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    cfg = {**DEFAULT_TRAIN_CONFIG, **(config or {})}

    print("\n" + "=" * 60)
    print("  Drone Detection – Model Training")
    print("=" * 60)
    print(f"  Model    : {cfg['model']}")
    print(f"  Epochs   : {cfg['epochs']}")
    print(f"  Image sz : {cfg['imgsz']}")
    print(f"  Batch    : {cfg['batch']}")
    print(f"  Output   : {cfg['project']}/{cfg['name']}")
    print("=" * 60 + "\n")

    model = YOLO(cfg.pop("model"))

    results = model.train(**cfg)

    best_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best_weights}")
    return str(best_weights)


def validate(weights: str, data_yaml: str = "configs/visdrone.yaml",
             imgsz: int = 640, split: str = "val") -> dict:
    """
    Validate a trained model and return metrics.

    Returns:
        dict: mAP50, mAP50-95, precision, recall per class
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split,
                        plots=True, save_json=True)

    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "per_class": {
            "person": {
                "AP50":      float(metrics.box.ap50[0]) if len(metrics.box.ap50) > 0 else 0,
                "AP50_95":   float(metrics.box.ap[0])   if len(metrics.box.ap) > 0 else 0,
            },
            "car": {
                "AP50":      float(metrics.box.ap50[1]) if len(metrics.box.ap50) > 1 else 0,
                "AP50_95":   float(metrics.box.ap[1])   if len(metrics.box.ap) > 1 else 0,
            },
        },
    }

    print("\n── Validation Results ──────────────────────────────")
    print(f"  mAP@50      : {results['mAP50']:.4f}")
    print(f"  mAP@50-95   : {results['mAP50_95']:.4f}")
    print(f"  Precision   : {results['precision']:.4f}")
    print(f"  Recall      : {results['recall']:.4f}")
    print("  Per-class:")
    for cls, m in results["per_class"].items():
        print(f"    {cls:8s}  AP@50={m['AP50']:.4f}  AP@50-95={m['AP50_95']:.4f}")
    print("────────────────────────────────────────────────────\n")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-02: Train YOLOv8 on VisDrone")
    parser.add_argument("--dataset",  required=True, help="VisDrone dataset root")
    parser.add_argument("--model",    default="yolov8m.pt")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--device",   default="")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")
    parser.add_argument("--weights",  default=None,
                        help="Existing weights to validate (skip training)")
    args = parser.parse_args()

    # Build/update data YAML
    data_yaml = build_data_yaml(args.dataset)

    if args.weights:
        # Validation only
        validate(args.weights, data_yaml)
    else:
        # Train
        cfg = {
            "model":   args.model,
            "data":    data_yaml,
            "epochs":  args.epochs,
            "imgsz":   args.imgsz,
            "batch":   args.batch,
            "device":  args.device,
        }
        best = train(cfg)

        if args.validate:
            validate(best, data_yaml)
