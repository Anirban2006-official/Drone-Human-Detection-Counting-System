"""
Task 01: Dataset Understanding & Preprocessing
===============================================
VisDrone dataset analysis, preprocessing, and augmentation pipeline.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict, Counter
import json
import random
from tqdm import tqdm


# ── VisDrone class mapping ──────────────────────────────────────────────────
VISDRONE_CLASSES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Map VisDrone classes → our 2 target classes
TARGET_CLASS_MAP = {
    1: 0,   # pedestrian  → person
    2: 0,   # people      → person
    4: 1,   # car         → car
    5: 1,   # van         → car
    6: 1,   # truck       → car
    9: 1,   # bus         → car
}

CLASS_NAMES = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}   # BGR: green=person, red=car


# ── Annotation parser ───────────────────────────────────────────────────────
def parse_visdrone_annotation(ann_path: str) -> list[dict]:
    """
    Parse a VisDrone .txt annotation file.

    Format per line:
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
        <score>,<category>,<truncation>,<occlusion>
    """
    objects = []
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score    = int(parts[4])   # 0=ignored, 1=visible
            category = int(parts[5])
            truncation = int(parts[6]) if len(parts) > 6 else 0
            occlusion  = int(parts[7]) if len(parts) > 7 else 0

            objects.append({
                "bbox": [x, y, w, h],
                "score": score,
                "category": category,
                "truncation": truncation,
                "occlusion": occlusion,
                "class_name": VISDRONE_CLASSES.get(category, "unknown"),
            })
    return objects


# ── Dataset statistics ──────────────────────────────────────────────────────
def analyze_dataset(dataset_root: str, split: str = "train") -> dict:
    """
    Compute dataset statistics for a given split.

    Args:
        dataset_root: root path of the VisDrone dataset
        split: one of 'train', 'val', 'test'

    Returns:
        dict with counts, bbox stats, occlusion stats, etc.
    """
    img_dir = Path(dataset_root) / "images" / split
    ann_dir = Path(dataset_root) / "annotations" / split

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    stats = {
        "split": split,
        "n_images": 0,
        "class_counts": defaultdict(int),
        "target_class_counts": defaultdict(int),
        "bbox_sizes": [],
        "occlusion_counts": defaultdict(int),
        "truncation_counts": defaultdict(int),
        "objects_per_image": [],
        "image_sizes": [],
    }

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    stats["n_images"] = len(img_paths)

    for img_path in tqdm(img_paths, desc=f"Analysing {split}"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            stats["image_sizes"].append((w, h))

        objects = parse_visdrone_annotation(str(ann_path))
        stats["objects_per_image"].append(len(objects))

        for obj in objects:
            cat = obj["category"]
            stats["class_counts"][VISDRONE_CLASSES.get(cat, "unknown")] += 1
            if cat in TARGET_CLASS_MAP:
                target = CLASS_NAMES[TARGET_CLASS_MAP[cat]]
                stats["target_class_counts"][target] += 1

            bw, bh = obj["bbox"][2], obj["bbox"][3]
            stats["bbox_sizes"].append((bw, bh))
            stats["occlusion_counts"][obj["occlusion"]] += 1
            stats["truncation_counts"][obj["truncation"]] += 1

    return stats


# ── Visualisation helpers ───────────────────────────────────────────────────
def visualize_sample(image_path: str, ann_path: str,
                     save_path: str | None = None,
                     show_ignored: bool = False) -> np.ndarray:
    """Draw bounding boxes on a sample image and optionally save/show it."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    objects = parse_visdrone_annotation(ann_path)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img_rgb)

    for obj in objects:
        cat = obj["category"]
        if cat == 0 and not show_ignored:
            continue
        x, y, w, h = obj["bbox"]
        target_id = TARGET_CLASS_MAP.get(cat)
        if target_id is not None:
            color = [c / 255 for c in CLASS_COLORS[target_id][::-1]]  # BGR→RGB
            label = CLASS_NAMES[target_id]
        else:
            if not show_ignored:
                continue
            color = [0.5, 0.5, 0.5]
            label = obj["class_name"]

        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 2, label, color=color, fontsize=7,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))

    ax.axis("off")
    ax.set_title(f"Sample: {Path(image_path).name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample visualisation → {save_path}")
    plt.show()
    return img_rgb


def plot_dataset_stats(stats: dict, save_dir: str = "outputs/visualizations"):
    """Generate statistical charts for the dataset."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"VisDrone Dataset Statistics ({stats['split']} split)", fontsize=14)

    # 1. Class distribution (all original classes)
    ax = axes[0, 0]
    cls_counts = dict(stats["class_counts"])
    cls_counts = {k: v for k, v in sorted(cls_counts.items(), key=lambda x: -x[1])}
    ax.bar(cls_counts.keys(), cls_counts.values(), color="steelblue")
    ax.set_title("Original Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    # 2. Target class distribution
    ax = axes[0, 1]
    tgt = dict(stats["target_class_counts"])
    bars = ax.bar(tgt.keys(), tgt.values(),
                  color=["#2ecc71", "#e74c3c"])
    ax.set_title("Target Class Distribution")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, tgt.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")

    # 3. Objects per image histogram
    ax = axes[0, 2]
    ax.hist(stats["objects_per_image"], bins=40, color="coral", edgecolor="black")
    ax.set_title("Objects per Image")
    ax.set_xlabel("Count")
    ax.set_ylabel("Frequency")
    ax.axvline(np.mean(stats["objects_per_image"]), color="navy",
               linestyle="--", label=f"Mean={np.mean(stats['objects_per_image']):.1f}")
    ax.legend()

    # 4. BBox size scatter
    ax = axes[1, 0]
    if stats["bbox_sizes"]:
        bws = [s[0] for s in stats["bbox_sizes"][:5000]]
        bhs = [s[1] for s in stats["bbox_sizes"][:5000]]
        ax.scatter(bws, bhs, alpha=0.3, s=5, color="purple")
        ax.set_title("Bounding Box Sizes (sample)")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)

    # 5. Occlusion distribution
    ax = axes[1, 1]
    occ_labels = {0: "No info", 1: "Visible", 2: "Partial", 3: "Heavy"}
    occ = {occ_labels.get(k, str(k)): v for k, v in stats["occlusion_counts"].items()}
    ax.pie(occ.values(), labels=occ.keys(), autopct="%1.1f%%",
           colors=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_title("Occlusion Distribution")

    # 6. Image count summary
    ax = axes[1, 2]
    summary = {
        "Images": stats["n_images"],
        "Total Objects": sum(stats["class_counts"].values()),
        "Persons": stats["target_class_counts"].get("person", 0),
        "Cars": stats["target_class_counts"].get("car", 0),
    }
    bars = ax.bar(summary.keys(), summary.values(),
                  color=["#34495e", "#9b59b6", "#2ecc71", "#e74c3c"])
    ax.set_title("Dataset Summary")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, summary.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(save_dir, f"dataset_stats_{stats['split']}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Stats chart saved → {out}")
    plt.show()


# ── YOLO label converter ────────────────────────────────────────────────────
def convert_to_yolo(dataset_root: str, output_root: str, splits: list[str] = None):
    """
    Convert VisDrone annotations to YOLO format.

    YOLO format per line:
        <class_id> <cx> <cy> <w> <h>   (all normalised 0-1)
    """
    if splits is None:
        splits = ["train", "val", "test"]

    for split in splits:
        img_dir = Path(dataset_root) / "images" / split
        ann_dir = Path(dataset_root) / "annotations" / split
        out_lbl = Path(output_root) / "labels" / split
        out_img = Path(output_root) / "images" / split
        out_lbl.mkdir(parents=True, exist_ok=True)
        out_img.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        skipped = 0

        for img_path in tqdm(img_paths, desc=f"Converting {split}"):
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            img_h, img_w = img.shape[:2]

            objects = parse_visdrone_annotation(str(ann_path))
            yolo_lines = []

            for obj in objects:
                cat = obj["category"]
                if cat not in TARGET_CLASS_MAP:
                    continue
                if obj["score"] == 0:   # ignored region
                    continue

                x, y, w, h = obj["bbox"]

                # Skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue
                # Skip too-small objects (< 5px) — common VisDrone challenge
                if w < 5 or h < 5:
                    continue

                # Clip to image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                class_id = TARGET_CLASS_MAP[cat]
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Write label file (even if empty — YOLO requires it)
            lbl_out = out_lbl / (img_path.stem + ".txt")
            lbl_out.write_text("\n".join(yolo_lines))

            # Symlink or copy image
            dst_img = out_img / img_path.name
            if not dst_img.exists():
                import shutil
                shutil.copy2(str(img_path), str(dst_img))

        print(f"[{split}] Converted {len(img_paths) - skipped} images. "
              f"Skipped {skipped}.")


# ── Augmentation (applied during training via Albumentations/YOLO) ──────────
AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,     # Hue shift
    "hsv_s": 0.7,       # Saturation
    "hsv_v": 0.4,       # Value/brightness
    "degrees": 5.0,     # Rotation (small – aerial images)
    "translate": 0.1,   # Translation fraction
    "scale": 0.5,       # Scale variance
    "shear": 2.0,       # Shear degrees
    "flipud": 0.3,      # Vertical flip probability
    "fliplr": 0.5,      # Horizontal flip probability
    "mosaic": 1.0,      # Mosaic augmentation (4-image)
    "mixup": 0.15,      # MixUp blending
    "copy_paste": 0.1,  # Copy-paste augmentation
}


def print_dataset_challenges():
    """Print a structured summary of known VisDrone challenges."""
    challenges = [
        ("Small object density",
         "Hundreds of tiny objects (< 20×20 px) per image; standard anchors miss them."),
        ("Heavy occlusion",
         "Crowded scenes lead to significant overlap; IoU-based NMS degrades recall."),
        ("Scale variance",
         "Objects range from 5 px to 300+ px depending on altitude & camera angle."),
        ("Class imbalance",
         "Person >> car >> other classes; requires weighted loss or oversampling."),
        ("Background clutter",
         "Rooftops, roads, vegetation create false-positive hotspots."),
        ("Lighting & weather",
         "Dataset includes dawn, dusk, foggy, and overcast conditions."),
        ("Camera motion blur",
         "High-speed drone footage introduces motion blur on small objects."),
        ("Truncation at edges",
         "Objects near frame borders are partially cut off."),
    ]
    print("\n" + "=" * 60)
    print("  VisDrone Dataset Challenges")
    print("=" * 60)
    for title, desc in challenges:
        print(f"\n  ▸ {title}")
        print(f"    {desc}")
    print("=" * 60 + "\n")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task-01: Dataset Understanding")
    parser.add_argument("--dataset", required=True,
                        help="Path to VisDrone dataset root")
    parser.add_argument("--output", default="data/yolo_visdrone",
                        help="Output path for converted YOLO labels")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--convert", action="store_true",
                        help="Convert annotations to YOLO format")
    parser.add_argument("--visualize_n", type=int, default=3,
                        help="Number of sample images to visualise")
    args = parser.parse_args()

    print_dataset_challenges()

    # Analyse
    stats = analyze_dataset(args.dataset, args.split)
    plot_dataset_stats(stats)

    print(f"\nDataset Summary ({args.split}):")
    print(f"  Images  : {stats['n_images']:,}")
    print(f"  Persons : {stats['target_class_counts'].get('person', 0):,}")
    print(f"  Cars    : {stats['target_class_counts'].get('car', 0):,}")
    print(f"  Avg obj/img: {np.mean(stats['objects_per_image']):.1f}")

    # Sample visualisations
    img_dir = Path(args.dataset) / "images" / args.split
    ann_dir = Path(args.dataset) / "annotations" / args.split
    imgs = sorted(img_dir.glob("*.jpg"))[:args.visualize_n]
    for i, ip in enumerate(imgs):
        ap = ann_dir / (ip.stem + ".txt")
        if ap.exists():
            visualize_sample(str(ip), str(ap),
                             save_path=f"outputs/visualizations/sample_{i+1}.png")

    # Convert
    if args.convert:
        print("\nConverting to YOLO format …")
        convert_to_yolo(args.dataset, args.output, splits=["train", "val", "test"])
        print("Conversion complete.")
