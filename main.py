"""
Drone Vision Pipeline – Main Entry Point
==========================================
Orchestrates all tasks end-to-end:

  Task 01 – Dataset understanding & preprocessing
  Task 02 – Model training (YOLOv8)
  Task 03 – Detection + human counting
  Task 04 – Object tracking (ByteTrack / SORT)
  Task 05 – Evaluation & visualization

Usage examples
--------------
# Full pipeline from scratch
python main.py --dataset /path/to/VisDrone --mode all

# Dataset analysis only
python main.py --dataset /path/to/VisDrone --mode dataset

# Train only
python main.py --dataset /path/to/VisDrone --mode train --epochs 50

# Detect & count on images (use existing weights)
python main.py --weights outputs/runs/visdrone_yolov8m/weights/best.pt \
               --source  /path/to/images --mode detect

# Detect + track on a video
python main.py --weights outputs/runs/visdrone_yolov8m/weights/best.pt \
               --source  /path/to/video.mp4 --mode track

# Evaluate (with ground truth)
python main.py --weights outputs/runs/visdrone_yolov8m/weights/best.pt \
               --source  /path/to/images --ann_dir /path/to/annotations \
               --mode evaluate
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Drone Vision Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    p.add_argument("--mode", default="all",
                   choices=["all", "dataset", "train", "detect", "track", "evaluate"],
                   help="Pipeline stage to run")

    # Paths
    p.add_argument("--dataset",  default=None,
                   help="VisDrone dataset root (required for dataset/train modes)")
    p.add_argument("--weights",  default=None,
                   help="Model weights (.pt). If None, downloads yolov8m.pt")
    p.add_argument("--source",   default=None,
                   help="Input: image file / folder / video file")
    p.add_argument("--ann_dir",  default=None,
                   help="Ground truth annotation dir (for evaluation)")
    p.add_argument("--output",   default="outputs",
                   help="Output directory")

    # Model / training
    p.add_argument("--model",    default="yolov8m.pt",
                   help="Base model for training (e.g. yolov8n.pt / yolov8m.pt)")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--device",   default="",
                   help="Training device: '' (auto), 'cpu', '0', '0,1'")

    # Inference
    p.add_argument("--conf",     type=float, default=0.30)
    p.add_argument("--iou",      type=float, default=0.45)

    # Tracking
    p.add_argument("--max_age",  type=int,   default=30)
    p.add_argument("--min_hits", type=int,   default=3)

    # Misc
    p.add_argument("--visualize_n", type=int, default=3,
                   help="Number of sample images to show in dataset stage")
    p.add_argument("--skip_convert", action="store_true",
                   help="Skip annotation conversion (already done)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(args):
    """Task 01 – Dataset understanding & preprocessing."""
    print("\n" + "=" * 60)
    print("  TASK 01 – Dataset Understanding & Preprocessing")
    print("=" * 60)

    if not args.dataset:
        print("[ERROR] --dataset is required for 'dataset' mode.")
        sys.exit(1)

    from src.dataset import (analyze_dataset, plot_dataset_stats,
                              visualize_sample, convert_to_yolo,
                              print_dataset_challenges)

    print_dataset_challenges()

    for split in ["train", "val"]:
        try:
            stats = analyze_dataset(args.dataset, split)
            plot_dataset_stats(stats, save_dir=f"{args.output}/visualizations")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    # Sample images
    img_dir = Path(args.dataset) / "images" / "train"
    ann_dir = Path(args.dataset) / "annotations" / "train"
    imgs = sorted(img_dir.glob("*.jpg"))[: args.visualize_n]
    for i, ip in enumerate(imgs):
        ap = ann_dir / (ip.stem + ".txt")
        if ap.exists():
            visualize_sample(str(ip), str(ap),
                             save_path=f"{args.output}/visualizations/sample_{i+1}.png")

    # Convert
    if not args.skip_convert:
        yolo_data = os.path.join(args.dataset, "yolo_format")
        convert_to_yolo(args.dataset, yolo_data, splits=["train", "val", "test"])
        return yolo_data
    return args.dataset


def run_train(args, data_root=None):
    """Task 02 – Model training."""
    print("\n" + "=" * 60)
    print("  TASK 02 – Model Training")
    print("=" * 60)

    dataset = data_root or args.dataset
    if not dataset:
        print("[ERROR] --dataset is required for 'train' mode.")
        sys.exit(1)

    from src.train import build_data_yaml, train, validate

    data_yaml = build_data_yaml(dataset)
    best_weights = train({
        "model":   args.model,
        "data":    data_yaml,
        "epochs":  args.epochs,
        "imgsz":   args.imgsz,
        "batch":   args.batch,
        "device":  args.device,
        "project": f"{args.output}/runs",
        "name":    "visdrone_yolov8m",
    })
    validate(best_weights, data_yaml)
    return best_weights


def run_detect(args, weights=None):
    """Task 03 – Detection + counting."""
    print("\n" + "=" * 60)
    print("  TASK 03 – Human & Car Detection with Counting")
    print("=" * 60)

    w = weights or args.weights
    if not w:
        print("[WARN] No weights specified. Using pretrained yolov8m.pt "
              "(not fine-tuned on VisDrone).")
        w = "yolov8m.pt"

    if not args.source:
        print("[ERROR] --source is required for 'detect' mode.")
        sys.exit(1)

    from src.detect import DroneDetector, run_on_folder, run_on_video

    detector = DroneDetector(weights=w, conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, device=args.device)

    src = Path(args.source)
    out_det = f"{args.output}/detections"
    os.makedirs(out_det, exist_ok=True)

    if src.is_dir():
        results = run_on_folder(detector, str(src), out_det)
    elif src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        out_vid = os.path.join(out_det, src.stem + "_detected.mp4")
        results = run_on_video(detector, str(src), out_vid)
    else:
        import cv2
        from src.detect import Visualizer
        img, result = detector.detect_file(str(src))
        annotated = Visualizer.draw(img, result)
        out_path = os.path.join(out_det, src.name)
        cv2.imwrite(out_path, annotated)
        print(f"Persons: {result.person_count}  Cars: {result.car_count}")
        results = [result]

    return results, detector


def run_track(args, weights=None):
    """Task 04 – Object tracking."""
    print("\n" + "=" * 60)
    print("  TASK 04 – Object Tracking (Bonus)")
    print("=" * 60)

    w = weights or args.weights or "yolov8m.pt"
    if not args.source:
        print("[ERROR] --source is required for 'track' mode.")
        sys.exit(1)

    from src.detect import DroneDetector, run_on_video, run_on_folder
    from src.tracking import ByteTrackWrapper, compute_tracking_stats

    detector = DroneDetector(weights=w, conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, device=args.device)
    tracker  = ByteTrackWrapper(max_age=args.max_age, min_hits=args.min_hits)

    src = Path(args.source)
    out_trk = f"{args.output}/tracking"
    os.makedirs(out_trk, exist_ok=True)

    if src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        out_vid = os.path.join(out_trk, src.stem + "_tracked.mp4")
        results = run_on_video(detector, str(src), out_vid, tracker=tracker)
    else:
        # Image folder with tracking
        results = run_on_folder(detector, str(src), out_trk)

    trk_stats = compute_tracking_stats(results)
    return results, trk_stats


def run_evaluate(args, results=None, weights=None):
    """Task 05 – Evaluation & visualization."""
    print("\n" + "=" * 60)
    print("  TASK 05 – Evaluation & Visualization")
    print("=" * 60)

    from src.evaluate import (plot_count_timeline, plot_pr_curves,
                               evaluate_predictions, generate_summary_report,
                               plot_evaluation_dashboard)

    os.makedirs(f"{args.output}/visualizations", exist_ok=True)

    # If we have results, plot timeline
    if results:
        plot_count_timeline(results,
                            f"{args.output}/visualizations/count_timeline.png")

    # Ground truth evaluation
    if args.ann_dir and args.source:
        import cv2
        import numpy as np
        from src.detect import DroneDetector, run_on_folder
        from src.dataset import parse_visdrone_annotation, TARGET_CLASS_MAP

        w = weights or args.weights or "yolov8m.pt"
        detector = DroneDetector(weights=w, conf=args.conf)

        if results is None:
            results = run_on_folder(detector, args.source,
                                     f"{args.output}/detections")

        gt_anns, pred_anns = [], []
        img_paths = sorted(Path(args.source).glob("*.jpg"))

        for fid, (ip, res) in enumerate(zip(img_paths, results)):
            ap = Path(args.ann_dir) / (ip.stem + ".txt")
            if not ap.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            for obj in parse_visdrone_annotation(str(ap)):
                if obj["category"] not in TARGET_CLASS_MAP:
                    continue
                x, y, bw, bh = obj["bbox"]
                gt_anns.append({"image_id": fid,
                                 "class_id": TARGET_CLASS_MAP[obj["category"]],
                                 "bbox": [x, y, x+bw, y+bh]})
            for det in res.detections:
                pred_anns.append({"image_id": fid, "class_id": det.class_id,
                                   "bbox": det.bbox, "score": det.confidence})

        eval_results = evaluate_predictions(gt_anns, pred_anns)
        plot_pr_curves(eval_results,
                       f"{args.output}/visualizations/pr_curves.png")
        plot_evaluation_dashboard(eval_results, results,
                                   f"{args.output}/visualizations/dashboard.png")
        generate_summary_report(eval_results,
                                 save_path=f"{args.output}/evaluation_report.json")
    else:
        import numpy as np
        dummy = {"person": {"AP": 0, "precision": 0, "recall": 0, "F1": 0,
                             "n_gt": 0, "n_pred": 0,
                             "recalls": np.array([]), "precisions": np.array([])},
                 "car":    {"AP": 0, "precision": 0, "recall": 0, "F1": 0,
                             "n_gt": 0, "n_pred": 0,
                             "recalls": np.array([]), "precisions": np.array([])},
                 "mAP": 0.0}
        generate_summary_report(dummy,
                                  save_path=f"{args.output}/evaluation_report.json")
        print("[INFO] No --ann_dir provided. Skipping metric computation.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/visualizations", exist_ok=True)

    weights = args.weights
    results = None

    if args.mode in ("all", "dataset"):
        data_root = run_dataset(args)
    else:
        data_root = args.dataset

    if args.mode in ("all", "train"):
        weights = run_train(args, data_root)

    if args.mode in ("all", "detect") and args.source:
        results, _ = run_detect(args, weights)

    if args.mode == "track" and args.source:
        results, _ = run_track(args, weights)

    if args.mode in ("all", "evaluate"):
        run_evaluate(args, results, weights)

    print("\n✓ Pipeline complete. Outputs saved to:", args.output)


if __name__ == "__main__":
    main()
