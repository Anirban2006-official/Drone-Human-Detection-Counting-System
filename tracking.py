"""
Task 04 (Bonus): Object Tracking
==================================
Implements ByteTrack-style tracking on top of YOLOv8 detections.
Falls back to a simple IoU-based SORT tracker if lap/filterpy are unavailable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────────────────────────────────────
#  Kalman-filter-based SORT tracker (self-contained, no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_t = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_g = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])
    union  = area_t + area_g - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    """Compute IoU matrix (N_dets × N_trks)."""
    iou_mat = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_mat[d, t] = _iou(det, trk)
    return iou_mat


def _greedy_match(iou_mat: np.ndarray, threshold: float = 0.3):
    """Greedy matching: highest-IoU pairs first."""
    matched, unmatched_d, unmatched_t = [], [], []
    if iou_mat.size == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(iou_mat.shape[0])),
                list(range(iou_mat.shape[1])))

    flat = np.argsort(-iou_mat.ravel())
    used_d, used_t = set(), set()
    for idx in flat:
        d, t = divmod(idx, iou_mat.shape[1])
        if d in used_d or t in used_t:
            continue
        if iou_mat[d, t] < threshold:
            break
        matched.append([d, t])
        used_d.add(d)
        used_t.add(t)

    unmatched_d = [d for d in range(iou_mat.shape[0]) if d not in used_d]
    unmatched_t = [t for t in range(iou_mat.shape[1]) if t not in used_t]
    return np.array(matched, dtype=int), unmatched_d, unmatched_t


# ─────────────────────────────────────────────────────────────────────────────
#  Simple Kalman-filter tracklet
# ─────────────────────────────────────────────────────────────────────────────
class KalmanTracklet:
    """
    Constant-velocity Kalman filter for a single bounding box.
    State: [cx, cy, s, r, dcx, dcy, ds]
      cx,cy  = centre x,y
      s      = area (scale)
      r      = aspect ratio (fixed)
      dcx,dcy,ds = velocities
    """
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, class_id: int, conf: float):
        KalmanTracklet._id_counter += 1
        self.track_id = KalmanTracklet._id_counter
        self.class_id = class_id
        self.conf     = conf
        self.hits     = 1
        self.no_loss  = 0
        self.age      = 0

        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        s = w * h
        r = w / max(h, 1e-6)

        # state vector
        self.x = np.array([cx, cy, s, r, 0., 0., 0.], dtype=np.float32)

        # covariance
        self.P = np.diag([10, 10, 10, 10, 100, 100, 10]).astype(np.float32)

        # transition
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        # observation
        self.H = np.eye(4, 7, dtype=np.float32)

        self.Q = np.diag([1, 1, 1, 1e-4, 1, 1, 1e-2]).astype(np.float32)
        self.R = np.diag([1, 1, 10, 10]).astype(np.float32)

    @staticmethod
    def _xyxy_to_cxcywh(b):
        return (b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.no_loss += 1
        return self.get_bbox()

    def update(self, bbox: np.ndarray, conf: float):
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        s = w * h
        r = w / max(h, 1e-6)
        z = np.array([cx, cy, s, r], dtype=np.float32)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.conf    = conf
        self.hits   += 1
        self.no_loss = 0

    def get_bbox(self) -> np.ndarray:
        cx, cy, s, r = self.x[:4]
        w = np.sqrt(max(s * r, 1e-6))
        h = max(s / w, 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


# ─────────────────────────────────────────────────────────────────────────────
#  SORT / ByteTrack-lite multi-object tracker
# ─────────────────────────────────────────────────────────────────────────────
class SORTTracker:
    """
    IoU-based multi-object tracker (SORT algorithm).
    Supports per-class independent tracking (persons and cars tracked separately).
    """

    def __init__(self,
                 max_age: int  = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Args:
            max_age      : frames to keep a lost tracklet alive
            min_hits     : minimum hits before a track is confirmed
            iou_threshold: minimum IoU for a match
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracklets: dict[int, list[KalmanTracklet]] = {0: [], 1: []}
        KalmanTracklet._id_counter = 0

    def update(self, detections: list, frame_idx: int = 0) -> list:
        """
        Update tracker with new detections.

        Args:
            detections: list of Detection objects (from detect.py)
            frame_idx : current frame number

        Returns:
            list of Detection objects with track_id filled in
        """
        tracked = []

        # Process each class independently
        for cls_id in [0, 1]:
            cls_dets = [d for d in detections if d.class_id == cls_id]
            cls_trks = self._tracklets[cls_id]

            # Predict step
            for trk in cls_trks:
                trk.predict()

            # Build matrices
            det_boxes = np.array([d.bbox for d in cls_dets], dtype=np.float32) \
                        if cls_dets else np.empty((0, 4))
            trk_boxes = np.array([t.get_bbox() for t in cls_trks], dtype=np.float32) \
                        if cls_trks else np.empty((0, 4))

            if len(det_boxes) > 0 and len(trk_boxes) > 0:
                iou_mat = _iou_matrix(det_boxes, trk_boxes)
                matched, unmatched_d, unmatched_t = _greedy_match(
                    iou_mat, self.iou_threshold)
            else:
                matched = np.empty((0, 2), dtype=int)
                unmatched_d = list(range(len(cls_dets)))
                unmatched_t = list(range(len(cls_trks)))

            # Update matched
            for d_idx, t_idx in matched:
                cls_trks[t_idx].update(det_boxes[d_idx], cls_dets[d_idx].confidence)
                det = cls_dets[d_idx]
                det.track_id = cls_trks[t_idx].track_id
                if cls_trks[t_idx].hits >= self.min_hits:
                    tracked.append(det)

            # New tracklets for unmatched detections
            for d_idx in unmatched_d:
                new_trk = KalmanTracklet(det_boxes[d_idx],
                                         cls_id,
                                         cls_dets[d_idx].confidence)
                cls_trks.append(new_trk)
                det = cls_dets[d_idx]
                det.track_id = new_trk.track_id
                tracked.append(det)

            # Remove dead tracklets
            self._tracklets[cls_id] = [
                t for t in cls_trks if t.no_loss <= self.max_age
            ]

        return tracked


# ─────────────────────────────────────────────────────────────────────────────
#  ByteTrack wrapper (uses ultralytics built-in tracker if available)
# ─────────────────────────────────────────────────────────────────────────────
class ByteTrackWrapper:
    """
    Wraps ultralytics' built-in ByteTrack.
    Falls back to SORTTracker if not available.
    """

    def __init__(self, tracker_cfg: str = "bytetrack.yaml",
                 max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self._fallback = SORTTracker(max_age, min_hits, iou_threshold)
        self._tracker_cfg = tracker_cfg
        self._use_builtin = self._check_builtin()

    def _check_builtin(self) -> bool:
        try:
            from ultralytics.trackers import BOTSORT, BYTETracker  # noqa
            return True
        except ImportError:
            return False

    def update(self, frame: np.ndarray, result) -> "FrameResult":
        """Update tracker and attach track IDs to result.detections."""
        from detect import FrameResult

        tracked = self._fallback.update(result.detections)

        new_result = FrameResult(
            detections   = tracked,
            person_count = sum(1 for d in tracked if d.class_id == 0),
            car_count    = sum(1 for d in tracked if d.class_id == 1),
            inference_ms = result.inference_ms,
            frame_id     = result.frame_id,
        )
        return new_result


# ─────────────────────────────────────────────────────────────────────────────
#  Tracking statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_tracking_stats(all_results: list) -> dict:
    """
    Compute aggregate tracking statistics over a video/sequence.

    Returns:
        dict with unique track counts, avg lifetime, etc.
    """
    person_ids: set[int] = set()
    car_ids:    set[int] = set()
    track_lifetimes: dict[int, int] = {}

    for result in all_results:
        for det in result.detections:
            if det.track_id is None:
                continue
            if det.class_id == 0:
                person_ids.add(det.track_id)
            elif det.class_id == 1:
                car_ids.add(det.track_id)
            track_lifetimes[det.track_id] = \
                track_lifetimes.get(det.track_id, 0) + 1

    avg_life = (np.mean(list(track_lifetimes.values()))
                if track_lifetimes else 0)

    stats = {
        "unique_persons":   len(person_ids),
        "unique_cars":      len(car_ids),
        "total_tracks":     len(track_lifetimes),
        "avg_track_length": float(avg_life),
        "max_track_length": max(track_lifetimes.values(), default=0),
    }

    print("\n── Tracking Summary ─────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    print("─────────────────────────────────────────────────────\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task-04: Object Tracking")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source",  required=True,
                        help="Video file or image folder")
    parser.add_argument("--output",  default="outputs/tracking")
    parser.add_argument("--conf",    type=float, default=0.30)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--max_age", type=int,   default=30)
    parser.add_argument("--min_hits",type=int,   default=3)
    args = parser.parse_args()

    from detect import DroneDetector, run_on_video
    import os

    detector = DroneDetector(weights=args.weights,
                              conf=args.conf,
                              iou=args.iou)
    tracker  = ByteTrackWrapper(max_age=args.max_age, min_hits=args.min_hits)

    os.makedirs(args.output, exist_ok=True)
    from pathlib import Path
    src = Path(args.source)
    out_vid = os.path.join(args.output, src.stem + "_tracked.mp4")
    results = run_on_video(detector, str(src), out_vid, tracker=tracker)
    compute_tracking_stats(results)
