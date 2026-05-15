"""
Task 05: Evaluation & Visualization
=====================================
Metrics computation, results visualization, and summary report generation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using the 101-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        ap += (np.max(p) if p.size else 0) / 101
    return float(ap)


def evaluate_predictions(gt_annotations: list[dict],
                          pred_annotations: list[dict],
                          iou_threshold: float = 0.5,
                          class_names: dict | None = None) -> dict:
    """
    Compute precision, recall, F1, and AP per class.

    Args:
        gt_annotations  : list of {image_id, class_id, bbox:[x1,y1,x2,y2]}
        pred_annotations: list of {image_id, class_id, bbox, score}
        iou_threshold   : IoU threshold for a TP (default 0.5)
        class_names     : {class_id: name}

    Returns:
        dict with per-class and overall metrics
    """
    if class_names is None:
        class_names = {0: "person", 1: "car"}

    # Group by class
    gt_by_cls   = defaultdict(lambda: defaultdict(list))
    pred_by_cls = defaultdict(list)

    for g in gt_annotations:
        gt_by_cls[g["class_id"]][g["image_id"]].append(g["bbox"])
    for p in pred_annotations:
        pred_by_cls[p["class_id"]].append(p)

    results = {}

    for cls_id, cls_name in class_names.items():
        preds = sorted(pred_by_cls[cls_id], key=lambda x: -x["score"])
        gt_cls = gt_by_cls[cls_id]

        n_gt = sum(len(v) for v in gt_cls.values())
        detected = defaultdict(lambda: [False] * 10000)  # matched flags

        tp_arr, fp_arr = [], []
        for pred in preds:
            img_id = pred["image_id"]
            gt_boxes = gt_cls.get(img_id, [])
            best_iou, best_j = 0, -1

            for j, gtb in enumerate(gt_boxes):
                iou = compute_iou(pred["bbox"], gtb)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0:
                if not detected[img_id][best_j]:
                    detected[img_id][best_j] = True
                    tp_arr.append(1); fp_arr.append(0)
                else:
                    tp_arr.append(0); fp_arr.append(1)
            else:
                tp_arr.append(0); fp_arr.append(1)

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        recalls    = tp_cum / max(n_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

        ap = compute_ap(recalls, precisions)
        final_p = float(precisions[-1]) if len(precisions) else 0.0
        final_r = float(recalls[-1])    if len(recalls)    else 0.0
        f1 = 2 * final_p * final_r / max(final_p + final_r, 1e-6)

        results[cls_name] = {
            "AP":        ap,
            "precision": final_p,
            "recall":    final_r,
            "F1":        f1,
            "n_gt":      n_gt,
            "n_pred":    len(preds),
            "recalls":   recalls,
            "precisions": precisions,
        }

    # Overall mAP
    all_aps = [v["AP"] for v in results.values()]
    results["mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

    # Print
    print("\n── Evaluation Results ──────────────────────────────────")
    print(f"  IoU threshold : {iou_threshold}")
    for cls, m in results.items():
        if cls == "mAP":
            continue
        print(f"\n  [{cls}]")
        print(f"    GT / Pred : {m['n_gt']} / {m['n_pred']}")
        print(f"    AP@{iou_threshold:.2f}    : {m['AP']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1        : {m['F1']:.4f}")
    print(f"\n  mAP@{iou_threshold:.2f}      : {results['mAP']:.4f}")
    print("────────────────────────────────────────────────────────\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(eval_results: dict, save_path: str = "outputs/visualizations/pr_curves.png"):
    """Plot Precision-Recall curves per class."""
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, axes = plt.subplots(1, len(eval_results) - 1, figsize=(14, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    colors = {"person": "#2ecc71", "car": "#e74c3c"}
    for ax, (cls, m) in zip(axes, {k: v for k, v in eval_results.items() if k != "mAP"}.items()):
        r = m["recalls"]
        p = m["precisions"]
        ax.step(r, p, color=colors.get(cls, "steelblue"), linewidth=2, where="post")
        ax.fill_between(r, p, alpha=0.2, color=colors.get(cls, "steelblue"), step="post")
        ax.set_title(f"{cls}  (AP={m['AP']:.3f})", fontsize=12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Precision-Recall Curves  (mAP={eval_results.get('mAP', 0):.3f})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"PR curves saved → {save_path}")
    plt.show()


def plot_count_timeline(results: list, save_path: str = "outputs/visualizations/count_timeline.png"):
    """Plot person/car counts over frames."""
    os.makedirs(Path(save_path).parent, exist_ok=True)
    frames   = [r.frame_id for r in results]
    persons  = [r.person_count for r in results]
    cars     = [r.car_count    for r in results]
    fps_vals = [1000 / r.inference_ms if r.inference_ms > 0 else 0 for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.fill_between(frames, persons, alpha=0.4, color="#2ecc71", label="Persons")
    ax1.plot(frames, persons, color="#2ecc71", linewidth=1.5)
    ax1.fill_between(frames, cars, alpha=0.4, color="#e74c3c", label="Cars")
    ax1.plot(frames, cars, color="#e74c3c", linewidth=1.5)
    ax1.set_ylabel("Count")
    ax1.set_title("Detection Counts Over Frames")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, fps_vals, color="#3498db", linewidth=1.5)
    ax2.fill_between(frames, fps_vals, alpha=0.3, color="#3498db")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("FPS")
    ax2.set_title("Inference Speed")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Count timeline saved → {save_path}")
    plt.show()


def create_prediction_mosaic(image_paths: list[str],
                              results: list,
                              detector,
                              n: int = 6,
                              save_path: str = "outputs/visualizations/prediction_mosaic.png"):
    """
    Create a mosaic of N annotated prediction images for visual inspection.
    """
    from detect import Visualizer
    os.makedirs(Path(save_path).parent, exist_ok=True)

    n = min(n, len(image_paths))
    cols = 3
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 640, 480

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx in range(n):
        img = cv2.imread(image_paths[idx])
        result = results[idx]
        annotated = Visualizer.draw(img, result)
        resized   = cv2.resize(annotated, (cell_w, cell_h))
        r, c = divmod(idx, cols)
        canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

    cv2.imwrite(save_path, canvas)
    print(f"Prediction mosaic saved → {save_path}")

    # Display with matplotlib
    plt.figure(figsize=(18, rows * 4))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Prediction Mosaic", fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_summary_report(eval_results: dict,
                             tracking_stats: dict | None = None,
                             save_path: str = "outputs/evaluation_report.json"):
    """
    Save a JSON summary report of evaluation + tracking results.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    report = {
        "evaluation": {k: {ek: ev for ek, ev in v.items()
                           if not isinstance(ev, np.ndarray)}
                       for k, v in eval_results.items()
                       if isinstance(v, dict)},
        "mAP":       eval_results.get("mAP", 0),
        "tracking":  tracking_stats or {},
        "strengths": [
            "YOLOv8 real-time inference suitable for drone applications",
            "Per-class tracking allows robust crowd counting",
            "Mosaic + copy-paste augmentation addresses small objects",
            "ByteTrack handles occlusion with 2-stage matching",
        ],
        "limitations": [
            "Small objects (< 10 px) remain challenging even after tiling",
            "Model struggles under heavy occlusion in dense crowds",
            "Fixed confidence threshold may miss low-visibility objects",
            "FPS may drop on CPU-only systems for high-res images",
        ],
        "challenges_faced": [
            "VisDrone class imbalance (pedestrians >> other classes)",
            "Tiny bounding boxes require carefully tuned NMS and anchors",
            "Aerial perspective causes unusual aspect ratios",
            "Converting VisDrone annotations to YOLO format required custom logic",
        ],
    }

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Summary report saved → {save_path}")
    return report


def plot_evaluation_dashboard(eval_results: dict,
                               count_results: list | None = None,
                               save_path: str = "outputs/visualizations/evaluation_dashboard.png"):
    """
    One-page evaluation dashboard combining key metrics.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Drone Detection – Evaluation Dashboard", fontsize=16, fontweight="bold")

    # ── Metric bars ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    classes = [k for k in eval_results if k != "mAP"]
    metrics = ["AP", "precision", "recall", "F1"]
    x = np.arange(len(classes))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    for i, metric in enumerate(metrics):
        vals = [eval_results[c].get(metric, 0) for c in classes]
        ax1.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Per-Class Metrics")
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── mAP gauge ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    mAP = eval_results.get("mAP", 0)
    theta = np.linspace(0, np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), "lightgrey", linewidth=20)
    ax2.plot(np.cos(theta[:int(mAP * 200)]),
             np.sin(theta[:int(mAP * 200)]),
             "#2ecc71", linewidth=20)
    ax2.text(0, -0.3, f"mAP\n{mAP:.3f}", ha="center", va="center",
             fontsize=18, fontweight="bold")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-0.5, 1.3)
    ax2.axis("off")
    ax2.set_title("mAP@0.50")

    # ── PR curves ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    cls_colors = {"person": "#2ecc71", "car": "#e74c3c"}
    for cls in classes:
        r = eval_results[cls].get("recalls", np.array([]))
        p = eval_results[cls].get("precisions", np.array([]))
        if len(r):
            ax3.step(r, p, color=cls_colors.get(cls, "grey"),
                     linewidth=2, label=f"{cls} (AP={eval_results[cls]['AP']:.3f})",
                     where="post")
    ax3.set_title("Precision-Recall")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ── Count timeline (if available) ────────────────────────────────────────
    if count_results:
        ax4 = fig.add_subplot(2, 3, (4, 6))
        frames  = [r.frame_id for r in count_results]
        persons = [r.person_count for r in count_results]
        cars    = [r.car_count    for r in count_results]
        ax4.fill_between(frames, persons, alpha=0.4, color="#2ecc71", label="Persons")
        ax4.plot(frames, persons, "#2ecc71", lw=1.5)
        ax4.fill_between(frames, cars, alpha=0.4, color="#e74c3c", label="Cars")
        ax4.plot(frames, cars, "#e74c3c", lw=1.5)
        ax4.set_title("Detection Counts Over Sequence")
        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Count")
        ax4.legend()
        ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from detect import DroneDetector, run_on_folder

    parser = argparse.ArgumentParser(description="Task-05: Evaluate & Visualize")
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--source",    required=True, help="Image folder to run inference on")
    parser.add_argument("--ann_dir",   default=None,  help="Ground truth annotation folder")
    parser.add_argument("--output",    default="outputs")
    parser.add_argument("--conf",      type=float, default=0.30)
    parser.add_argument("--iou_thr",   type=float, default=0.50,
                        help="IoU threshold for evaluation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/visualizations", exist_ok=True)

    detector = DroneDetector(weights=args.weights, conf=args.conf)
    results  = run_on_folder(detector, args.source,
                              f"{args.output}/detections")

    # Count timeline
    plot_count_timeline(results, f"{args.output}/visualizations/count_timeline.png")

    # Evaluate against ground truth (if provided)
    if args.ann_dir:
        from dataset import parse_visdrone_annotation, TARGET_CLASS_MAP
        from pathlib import Path as P

        gt_anns, pred_anns = [], []
        img_paths = sorted(P(args.source).glob("*.jpg"))

        for fid, (img_path, res) in enumerate(zip(img_paths, results)):
            ann_path = P(args.ann_dir) / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            objs = parse_visdrone_annotation(str(ann_path))
            for obj in objs:
                if obj["category"] not in TARGET_CLASS_MAP:
                    continue
                x, y, bw, bh = obj["bbox"]
                gt_anns.append({
                    "image_id": fid, "class_id": TARGET_CLASS_MAP[obj["category"]],
                    "bbox": [x, y, x+bw, y+bh]
                })

            for det in res.detections:
                pred_anns.append({
                    "image_id": fid, "class_id": det.class_id,
                    "bbox": det.bbox, "score": det.confidence
                })

        eval_results = evaluate_predictions(gt_anns, pred_anns,
                                            iou_threshold=args.iou_thr)
        plot_pr_curves(eval_results,
                       f"{args.output}/visualizations/pr_curves.png")
        generate_summary_report(eval_results,
                                 save_path=f"{args.output}/evaluation_report.json")
        plot_evaluation_dashboard(eval_results, results,
                                   f"{args.output}/visualizations/evaluation_dashboard.png")
    else:
        # No GT – just generate the report with dummy metrics
        dummy = {"person": {"AP":0,"precision":0,"recall":0,"F1":0,"n_gt":0,"n_pred":0,
                             "recalls":np.array([]),"precisions":np.array([])},
                 "car":    {"AP":0,"precision":0,"recall":0,"F1":0,"n_gt":0,"n_pred":0,
                             "recalls":np.array([]),"precisions":np.array([])},
                 "mAP": 0.0}
        generate_summary_report(dummy,
                                 save_path=f"{args.output}/evaluation_report.json")

    print("Evaluation complete.")
