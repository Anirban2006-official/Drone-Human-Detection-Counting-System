"""
Task 03: Human & Car Detection with Human Counting
====================================================
Inference pipeline: load weights → detect → draw boxes → count humans.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time


# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 220, 0), 1: (0, 80, 255)}   # BGR: green=person, blue=car

DEFAULT_CONF  = 0.30
DEFAULT_IOU   = 0.45
DEFAULT_IMGSZ = 640


# ── Detection result dataclass ───────────────────────────────────────────────
@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2]  absolute pixels
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    person_count: int = 0
    car_count: int = 0
    inference_ms: float = 0.0
    frame_id: int = 0

    @property
    def total_count(self) -> int:
        return self.person_count + self.car_count


# ── Detector ─────────────────────────────────────────────────────────────────
class DroneDetector:
    """
    YOLOv8-based detector for person & car classes in aerial imagery.
    """

    def __init__(self,
                 weights: str = "yolov8m.pt",
                 conf: float = DEFAULT_CONF,
                 iou: float  = DEFAULT_IOU,
                 imgsz: int  = DEFAULT_IMGSZ,
                 device: str = ""):
        """
        Args:
            weights : path to fine-tuned weights (or pretrained hub weights)
            conf    : minimum confidence threshold
            iou     : NMS IoU threshold
            imgsz   : inference image size (px)
            device  : '' = auto, 'cpu', '0', '0,1', etc.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self.model  = YOLO(weights)
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device
        print(f"[DroneDetector] Loaded weights: {weights}")

    # ── Core inference ────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray) -> FrameResult:
        """
        Run detection on a single BGR frame.

        Args:
            image: BGR numpy array (H, W, 3)

        Returns:
            FrameResult with all detections and counts
        """
        t0 = time.perf_counter()
        raw = self.model(image,
                         conf=self.conf,
                         iou=self.iou,
                         imgsz=self.imgsz,
                         device=self.device,
                         verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = FrameResult(inference_ms=elapsed_ms)
        boxes  = raw.boxes

        if boxes is None or len(boxes) == 0:
            return result

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score      = float(box.conf[0])
            cls_id          = int(box.cls[0])

            if cls_id not in CLASS_NAMES:
                continue

            det = Detection(
                bbox        = [x1, y1, x2, y2],
                confidence  = conf_score,
                class_id    = cls_id,
                class_name  = CLASS_NAMES[cls_id],
            )
            result.detections.append(det)

            if cls_id == 0:
                result.person_count += 1
            elif cls_id == 1:
                result.car_count += 1

        return result

    def detect_file(self, path: str) -> tuple[np.ndarray, FrameResult]:
        """Detect on an image file. Returns (image, result)."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img, self.detect(img)


# ── Visualisation ─────────────────────────────────────────────────────────────
class Visualizer:
    """Draw detections, counts, and stats onto frames."""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICKNESS  = 2

    @staticmethod
    def draw(frame: np.ndarray, result: FrameResult,
             show_conf: bool = True, show_count: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and the human/car count overlay.

        Args:
            frame      : BGR image
            result     : FrameResult from DroneDetector
            show_conf  : show confidence score on each box
            show_count : draw the summary counter panel

        Returns:
            Annotated BGR image
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Bounding boxes ────────────────────────────────────────────────────
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS[det.class_id]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label_parts = [det.class_name]
            if det.track_id is not None:
                label_parts.append(f"#{det.track_id}")
            if show_conf:
                label_parts.append(f"{det.confidence:.2f}")
            label = " ".join(label_parts)

            (lw, lh), baseline = cv2.getTextSize(
                label, Visualizer.FONT, Visualizer.FONT_SCALE, 1)
            y_label = max(y1 - 4, lh + 4)
            cv2.rectangle(out,
                          (x1, y_label - lh - baseline - 2),
                          (x1 + lw + 4, y_label + 2),
                          color, -1)
            cv2.putText(out, label,
                        (x1 + 2, y_label - baseline),
                        Visualizer.FONT, Visualizer.FONT_SCALE,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # ── Count panel ───────────────────────────────────────────────────────
        if show_count:
            panel_lines = [
                f" Persons : {result.person_count:>4d}",
                f" Cars    : {result.car_count:>4d}",
                f" Total   : {result.total_count:>4d}",
                f" FPS     : {1000/result.inference_ms:.1f}" if result.inference_ms > 0 else "",
            ]
            panel_lines = [l for l in panel_lines if l]

            font_scale = 0.65
            pad = 8
            line_h = 26
            panel_h = len(panel_lines) * line_h + pad * 2
            panel_w = 200

            # Semi-transparent dark background
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, panel_h),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

            for i, line in enumerate(panel_lines):
                y_pos = pad + (i + 1) * line_h - 4
                # Choose colour per line
                if "Person" in line:
                    col = CLASS_COLORS[0]
                elif "Car" in line:
                    col = CLASS_COLORS[1]
                else:
                    col = (220, 220, 220)
                cv2.putText(out, line, (4, y_pos),
                            Visualizer.FONT, font_scale, col, 1, cv2.LINE_AA)

        return out

    @staticmethod
    def draw_summary_grid(images: list[np.ndarray],
                          results: list[FrameResult],
                          grid_size: tuple[int, int] = (2, 3),
                          cell_size: tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Compose multiple annotated frames into a summary grid image.
        """
        rows, cols = grid_size
        cw, ch = cell_size
        grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

        for idx, (img, res) in enumerate(zip(images, results)):
            if idx >= rows * cols:
                break
            annotated = Visualizer.draw(img, res)
            resized   = cv2.resize(annotated, (cw, ch))
            r, c = divmod(idx, cols)
            grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = resized

        return grid


# ── Batch inference ───────────────────────────────────────────────────────────
def run_on_folder(detector: "DroneDetector",
                  input_dir: str,
                  output_dir: str,
                  extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
                  save_txt: bool = True) -> list[FrameResult]:
    """
    Run detection on every image in a folder and save annotated outputs.

    Args:
        detector   : DroneDetector instance
        input_dir  : folder with input images
        output_dir : folder for annotated images (+ optional txt results)
        extensions : image file extensions to process
        save_txt   : write per-image detection CSV files

    Returns:
        list of FrameResult (one per image)
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_txt:
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    paths = [p for p in sorted(Path(input_dir).iterdir())
             if p.suffix.lower() in extensions]
    print(f"Processing {len(paths)} images in '{input_dir}' …")

    all_results = []
    for fid, img_path in enumerate(paths):
        img, result = detector.detect_file(str(img_path))
        result.frame_id = fid
        annotated = Visualizer.draw(img, result)

        out_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(out_path), annotated)

        if save_txt:
            lines = [f"{d.class_id},{d.class_name},"
                     f"{d.bbox[0]:.1f},{d.bbox[1]:.1f},"
                     f"{d.bbox[2]:.1f},{d.bbox[3]:.1f},"
                     f"{d.confidence:.4f}"
                     for d in result.detections]
            lbl_path = Path(output_dir) / "labels" / (img_path.stem + ".csv")
            lbl_path.write_text("\n".join(lines))

        all_results.append(result)
        print(f"  [{fid+1:>4d}/{len(paths)}] {img_path.name:40s} "
              f"persons={result.person_count:3d}  cars={result.car_count:3d}  "
              f"ms={result.inference_ms:.1f}")

    # Summary
    total_persons = sum(r.person_count for r in all_results)
    total_cars    = sum(r.car_count    for r in all_results)
    avg_fps       = 1000 / (sum(r.inference_ms for r in all_results) / len(all_results)) \
                    if all_results else 0

    print(f"\n── Batch Summary ──────────────────────────────────")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Total persons    : {total_persons:,}")
    print(f"  Total cars       : {total_cars:,}")
    print(f"  Avg FPS          : {avg_fps:.1f}")
    print(f"  Saved to         : {output_dir}")
    print(f"──────────────────────────────────────────────────\n")

    return all_results


# ── Video inference ───────────────────────────────────────────────────────────
def run_on_video(detector: "DroneDetector",
                 video_path: str,
                 output_path: str,
                 tracker=None) -> list[FrameResult]:
    """
    Run detection (and optionally tracking) on a video file.

    Args:
        detector   : DroneDetector instance
        video_path : path to input video
        output_path: path for annotated output video
        tracker    : optional Tracker instance (from tracking.py)

    Returns:
        list of FrameResult (one per frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_results, fid = [], 0
    print(f"Processing video: {video_path}  ({total} frames @ {fps} FPS)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        result.frame_id = fid

        if tracker is not None:
            result = tracker.update(frame, result)

        annotated = Visualizer.draw(frame, result)
        writer.write(annotated)
        all_results.append(result)
        fid += 1

        if fid % 50 == 0:
            print(f"  frame {fid}/{total}")

    cap.release()
    writer.release()
    print(f"Video saved → {output_path}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task-03: Detect & Count")
    parser.add_argument("--weights",  required=True, help="Model weights (.pt)")
    parser.add_argument("--source",   required=True,
                        help="Image file, image folder, or video file")
    parser.add_argument("--output",   default="outputs/detections")
    parser.add_argument("--conf",     type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou",      type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument("--device",   default="")
    parser.add_argument("--track",    action="store_true",
                        help="Enable ByteTrack tracking (requires tracking.py)")
    args = parser.parse_args()

    detector = DroneDetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    tracker = None
    if args.track:
        from tracking import ByteTrackWrapper
        tracker = ByteTrackWrapper()

    src = Path(args.source)
    if src.is_dir():
        run_on_folder(detector, str(src), args.output)
    elif src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        os.makedirs(args.output, exist_ok=True)
        out_vid = os.path.join(args.output, src.stem + "_detected.mp4")
        run_on_video(detector, str(src), out_vid, tracker=tracker)
    else:
        img, result = detector.detect_file(str(src))
        annotated = Visualizer.draw(img, result)
        out_path = os.path.join(args.output, src.name)
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"Persons: {result.person_count}  Cars: {result.car_count}")
        print(f"Saved → {out_path}")
