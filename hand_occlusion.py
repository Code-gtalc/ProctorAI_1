from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
    area_a = float(max(1, (ax1 - ax0) * (ay1 - ay0)))
    area_b = float(max(1, (bx1 - bx0) * (by1 - by0)))
    return inter_area / max(1e-6, (area_a + area_b - inter_area))


def _bbox_from_points(points: np.ndarray, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if points.size == 0:
        return None
    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def mouth_bbox_from_facemesh(face_landmarks: Any, frame_shape: tuple[int, int, int]) -> Optional[tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    lip_ids = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
    ]
    pts = np.array(
        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in lip_ids],
        dtype=np.int32,
    )
    return _bbox_from_points(pts, w, h)


def hand_bboxes_from_mediapipe(hand_landmarks: Any, frame_shape: tuple[int, int, int], pad_ratio: float) -> list[tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if not hand_landmarks:
        return boxes
    for hand in hand_landmarks:
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in hand.landmark], dtype=np.int32)
        box = _bbox_from_points(pts, w, h)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        padx = int((x1 - x0) * pad_ratio) + 2
        pady = int((y1 - y0) * pad_ratio) + 2
        boxes.append((max(0, x0 - padx), max(0, y0 - pady), min(w - 1, x1 + padx), min(h - 1, y1 + pady)))
    return boxes


@dataclass
class HandMouthOcclusionResult:
    mouth_visible: bool
    mouth_occluded: bool
    overlap_iou: float
    consecutive_frames: int


class HandMouthOcclusionDetector:
    def __init__(self, iou_threshold: float = 0.03, consecutive_frames: int = 3, pad_ratio: float = 0.2) -> None:
        self.iou_threshold = iou_threshold
        self.consecutive_frames = consecutive_frames
        self.pad_ratio = pad_ratio
        self._streak = 0

    def update(
        self,
        mouth_box: Optional[tuple[int, int, int, int]],
        hand_landmarks: Any,
        frame_shape: tuple[int, int, int],
    ) -> HandMouthOcclusionResult:
        if mouth_box is None:
            self._streak = 0
            return HandMouthOcclusionResult(True, False, 0.0, 0)

        hand_boxes = hand_bboxes_from_mediapipe(hand_landmarks, frame_shape, self.pad_ratio)
        max_iou = 0.0
        for hand_box in hand_boxes:
            max_iou = max(max_iou, bbox_iou(hand_box, mouth_box))

        if max_iou > self.iou_threshold:
            self._streak += 1
        else:
            self._streak = 0

        occluded = self._streak >= self.consecutive_frames
        return HandMouthOcclusionResult(
            mouth_visible=not occluded,
            mouth_occluded=occluded,
            overlap_iou=max_iou,
            consecutive_frames=self._streak,
        )
