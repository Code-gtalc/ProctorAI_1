import asyncio
from collections import deque
from threading import Lock
import time
from typing import Literal, Optional
import os

import cv2
import mediapipe as mp
import moviepy.editor as moviepy
import numpy as np
import pandas as pd
import yaml
from typing import Any

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = Any

try:
    import sounddevice as sd
except ImportError:
    sd = None


# ---- Load Config ----
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CONFIG_PATH: str = cfg["config_path"]
VIDEO_PATH: str = cfg["video_path"]
WINDOW_SIZE: int = int(cfg["window_size"])
SILENCE_DELAY: float = float(cfg["silence_delay"])
MAHAL_THRESHOLD: float = float(cfg["mahal_threshold"])
NUM_CORES: int = int(cfg["num_cores"])
ONLY_VAD: Literal[True, False] = bool(cfg["only_vad"])
LIVE_MODE: Literal[True, False] = bool(cfg.get("live_mode", True))
CAMERA_INDEX: int = int(cfg.get("camera_index", 0))
MIC_DEVICE_INDEX: Optional[int] = cfg.get("mic_device_index")
MIC_SAMPLE_RATE: int = int(cfg.get("mic_sample_rate", 16000))
MIC_BLOCK_SIZE: int = int(cfg.get("mic_block_size", 1024))
MIC_THRESHOLD: float = float(cfg.get("mic_threshold", 0.015))
LIP_MOTION_THRESHOLD: float = float(cfg.get("lip_motion_threshold", 6.0))
EMA_ALPHA: float = float(cfg.get("ema_alpha", 0.35))
NERVOUS_LIP_MULTIPLIER: float = float(cfg.get("nervous_lip_multiplier", 1.4))
MOUTH_HIDDEN_TEXTURE_THRESHOLD: float = float(cfg.get("mouth_hidden_texture_threshold", 10.0))
RULE_MAR_THRESHOLD: float = float(cfg.get("rule_mar_threshold", 0.20))
RULE_MAR_DELTA_THRESHOLD: float = float(cfg.get("rule_mar_delta_threshold", 0.015))
OPTICAL_FLOW_THRESHOLD: float = float(cfg.get("optical_flow_threshold", 0.90))
CORR_WINDOW_FRAMES: int = int(cfg.get("corr_window_frames", 24))
CORR_THRESHOLD: float = float(cfg.get("corr_threshold", 0.30))
CORR_MAX_LAG_FRAMES: int = int(cfg.get("corr_max_lag_frames", 4))
GATE_FLOW_EPSILON: float = float(cfg.get("gate_flow_epsilon", 0.35))
GATE_FLOW_WINDOW_FRAMES: int = int(cfg.get("gate_flow_window_frames", 12))
GATE_FLOW_MIN_COUNT: int = int(cfg.get("gate_flow_min_count", 2))
GATE_MAR_MIN: float = float(cfg.get("gate_mar_min", 0.02))
GATE_MAR_MAX: float = float(cfg.get("gate_mar_max", 1.20))
GATE_STABILITY_FRAMES: int = int(cfg.get("gate_stability_frames", 4))
HAND_MOUTH_IOU_THRESHOLD: float = float(cfg.get("hand_mouth_iou_threshold", 0.03))
HAND_BOX_PADDING_RATIO: float = float(cfg.get("hand_box_padding_ratio", 0.20))
MOUTH_OCCLUSION_COVERAGE_THRESHOLD: float = float(cfg.get("mouth_occlusion_coverage_threshold", 0.20))


# ---- Audio Extraction ----
def extract_audio(video_path: str, audio_output: str = "temp_audio.wav") -> tuple[str, int]:
    clip = moviepy.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output, codec='pcm_s16le')
    return audio_output, int(clip.duration)

# ---- VAD Segments ----
def get_vad_segments(pipeline: Pipeline, audio_path: str) -> list[tuple[float, float]]:
    vad = pipeline(audio_path)
    return [(seg.start, seg.end) for seg in vad.get_timeline().support()]

def is_vad_speaking(time_s: float, vad_segments: list[tuple[float, float]]) -> bool:
    return any(start <= time_s <= end for start, end in vad_segments)

# ---- CSV Export ----
# def export_combined_csv(
#     results: list[dict],
#     csv_path: str = "lipsync_detection_results.csv",
#     only_vad: bool = False
# ) -> pd.DataFrame:
#     rows = []
#     for r in results:
#         ts = r["Time (s)"] + 1
#         m, s = divmod(ts, 60)
#         formatted = f"{int(m)}.{int(s):02d}"
#         row = {"Timestamp": formatted, "VAD Speaking": r.get("VAD Speaking", "N/A")}
#         if not only_vad:
#             mah = r.get("Mahalanobis Status", "N/A")
#             if mah == "No Lips Detected":
#                 lip = "No Lips Detected"
#             elif isinstance(row["VAD Speaking"], str) or isinstance(mah, str):
#                 lip = "No"
#             else:
#                 lip = "Yes" if row["VAD Speaking"] != mah else "No"
#             row.update({
#                 "Mahalanobis Status": mah,
#                 "Lipsync": lip
#             })
#         rows.append(row)
#     df = pd.DataFrame(rows)
#     df.to_csv(csv_path, index=False)
#     return df

# ---- Lip Feature Computation ----
def calculate_mahalanobis(feature_window: deque[np.ndarray], features: np.ndarray) -> float:
    if len(feature_window) < 2:
        return 0.0
    arr = np.array(feature_window)
    mean = arr.mean(axis=0)
    diff = features - mean
    cov = np.cov(arr.T) + np.eye(arr.shape[1]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(diff.dot(inv_cov).dot(diff))

def extract_lip_features(
    landmarks: Any,
    shape: tuple[int, int, int],
    height_window: deque[float],
    movement_window: deque[float],
    feature_window: deque[np.ndarray]
) -> np.ndarray:
    ih, iw = shape[:2]
    upper_ids = [40, 0, 270]
    lower_ids = [181, 17, 405]
    upper_pts = np.array([(int(landmarks.landmark[i].x * iw), int(landmarks.landmark[i].y * ih)) for i in upper_ids])
    lower_pts = np.array([(int(landmarks.landmark[i].x * iw), int(landmarks.landmark[i].y * ih)) for i in lower_ids])
    dists = [abs(upper_pts[i][1] - lower_pts[i][1]) for i in range(min(len(upper_pts), len(lower_pts)))]
    inner_h = np.mean(dists) / ih
    lip_w = (upper_pts[:, 0].max() - upper_pts[:, 0].min()) / iw
    lip_area = inner_h * lip_w

    height_window.append(inner_h)
    mov = 0.0
    if len(height_window) >= 2:
        mov = abs(height_window[-1] - height_window[-2])
        movement_window.append(mov)

    features = np.array([inner_h, movement_window[-1] if movement_window else 0.0, lip_area])
    feature_window.append(features)
    return features

# ---- Speaking Detection ----
def detect_speaking(
    features: np.ndarray,
    time_s: float,
    feature_window: deque[np.ndarray],
    last_time: Optional[float]
) -> tuple[bool, Optional[float]]:
    dist = calculate_mahalanobis(feature_window, features)
    if dist > MAHAL_THRESHOLD:
        return True, time_s
    if last_time is None:
        return False, last_time
    return (time_s - last_time) < SILENCE_DELAY, last_time


class MicrophoneVoiceMonitor:
    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        threshold: float,
        device_index: Optional[int] = None
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.device_index = device_index
        self._rms: float = 0.0
        self._lock = Lock()
        self._stream = None

    def _audio_callback(self, indata, frames, callback_time, status) -> None:
        del frames, callback_time, status
        rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        with self._lock:
            self._rms = rms

    def start(self) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for live microphone voice status.")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            device=self.device_index,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def current_rms(self) -> float:
        with self._lock:
            return self._rms

    def is_speaking(self) -> bool:
        return self.current_rms() >= self.threshold


def draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    h, w = frame.shape[:2]
    box_width = min(620, max(340, w - 30))
    x = max(10, w - box_width - 10)
    y = 25
    line_h = 20
    box_h = line_h * len(lines) + 12
    cv2.rectangle(frame, (x - 8, 6), (x + box_width, box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 8, 6), (x + box_width, box_h), (0, 255, 0), 1)
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (x, y + (idx * line_h)), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def ema(previous: float, current: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.01), 1.0)
    return (alpha * current) + ((1.0 - alpha) * previous)


def compute_mar_mesh(landmarks: Any) -> float:
    points = landmarks.landmark
    upper_inner = points[13]
    lower_inner = points[14]
    left_corner = points[78]
    right_corner = points[308]
    vertical = abs(lower_inner.y - upper_inner.y)
    horizontal = max(abs(right_corner.x - left_corner.x), 1e-6)
    return float(vertical / horizontal)


def classify_rule_based(mouth_open: bool, audio_active: bool) -> str:
    if mouth_open and audio_active:
        return "VALID"
    if mouth_open and (not audio_active):
        return "SUSPICIOUS_MOUTH_OPEN_NO_AUDIO"
    if audio_active and (not mouth_open):
        return "SUSPICIOUS_AUDIO_NO_MOUTH_MOVEMENT"
    return "QUIET"


def extract_mouth_roi_gray(frame: np.ndarray, lip_box: tuple[int, int, int, int], pad: int = 2) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = lip_box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 48))


def compute_optical_flow_intensity(prev_roi: Optional[np.ndarray], curr_roi: Optional[np.ndarray]) -> float:
    if prev_roi is None or curr_roi is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        curr_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def classify_optical_flow(motion_high: bool, audio_high: bool) -> str:
    if motion_high and audio_high:
        return "SYNCED"
    if (not motion_high) and (not audio_high):
        return "QUIET"
    return "NOT_SYNCED"


def compute_cross_correlation_score(
    audio_series: deque[float],
    lip_series: deque[float],
    max_lag: int
) -> tuple[Optional[float], int]:
    if len(audio_series) != len(lip_series) or len(audio_series) < 6:
        return None, 0

    audio = np.array(audio_series, dtype=np.float32)
    lips = np.array(lip_series, dtype=np.float32)
    if np.std(audio) < 1e-6 or np.std(lips) < 1e-6:
        return None, 0

    best_score = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = audio[-lag:]
            l = lips[:len(lips) + lag]
        elif lag > 0:
            a = audio[:-lag]
            l = lips[lag:]
        else:
            a = audio
            l = lips
        if len(a) < 4 or len(l) < 4:
            continue
        corr = np.corrcoef(a, l)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_score:
            best_score = float(corr)
            best_lag = lag

    if best_score < -0.99:
        return None, 0
    return best_score, best_lag


def classify_av_correlation(score: Optional[float], threshold: float) -> str:
    if score is None:
        return "INSUFFICIENT_DATA"
    return "SYNCED" if score >= threshold else "NOT_SYNCED"


def is_mar_non_degenerate(mar_value: float, mar_min: float, mar_max: float) -> bool:
    return bool(np.isfinite(mar_value) and mar_min <= mar_value <= mar_max)


def evaluate_multi_signal_gate(
    face_detected: bool,
    texture_std: Optional[float],
    flow_value: float,
    mar_value: float,
    flow_window: deque[bool],
    stability_streak: int
) -> tuple[bool, int, dict[str, bool]]:
    if not face_detected:
        flow_window.clear()
        return False, 0, {
            "face_ok": False,
            "texture_ok": False,
            "flow_ok": False,
            "mar_ok": False,
            "stability_ok": False,
        }

    texture_ok = (texture_std is not None) and (texture_std > MOUTH_HIDDEN_TEXTURE_THRESHOLD)
    flow_window.append(flow_value > GATE_FLOW_EPSILON)
    flow_ok = sum(flow_window) >= GATE_FLOW_MIN_COUNT
    mar_ok = is_mar_non_degenerate(mar_value, GATE_MAR_MIN, GATE_MAR_MAX)
    candidate = texture_ok and flow_ok and mar_ok
    stability_streak = (stability_streak + 1) if candidate else 0
    stability_ok = stability_streak >= GATE_STABILITY_FRAMES
    gate_ok = candidate and stability_ok

    return gate_ok, stability_streak, {
        "face_ok": True,
        "texture_ok": texture_ok,
        "flow_ok": flow_ok,
        "mar_ok": mar_ok,
        "stability_ok": stability_ok,
    }


def safe_nanmean(values: list[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def majority_sync_status(rule_status: str, optical_status: str, corr_status: str) -> str:
    positive = 0
    negative = 0
    if rule_status == "VALID":
        positive += 1
    elif rule_status.startswith("SUSPICIOUS"):
        negative += 1
    if optical_status == "SYNCED":
        positive += 1
    elif optical_status == "NOT_SYNCED":
        negative += 1
    if corr_status == "SYNCED":
        positive += 1
    elif corr_status == "NOT_SYNCED":
        negative += 1

    if positive == 0 and negative == 0:
        return "QUIET"
    if positive > negative:
        return "SYNCED"
    if negative > positive:
        return "NOT_SYNCED"
    return "UNCERTAIN"


def classify_expression_mesh(landmarks: Any) -> str:
    points = landmarks.landmark
    left_corner = points[61]
    right_corner = points[291]
    upper_lip = points[13]
    lower_lip = points[14]
    brow_left = points[70]
    brow_right = points[300]
    eye_left_top = points[159]
    eye_right_top = points[386]
    forehead = points[10]
    chin = points[152]

    face_height = max(chin.y - forehead.y, 1e-6)
    mouth_open = abs(lower_lip.y - upper_lip.y) / face_height
    corner_avg_y = (left_corner.y + right_corner.y) / 2.0
    smile_curve = (upper_lip.y - corner_avg_y) / face_height
    brow_eye = ((eye_left_top.y - brow_left.y) + (eye_right_top.y - brow_right.y)) / (2.0 * face_height)

    if smile_curve > 0.012 and mouth_open > 0.010:
        return "HAPPY"
    if smile_curve < -0.006 and mouth_open < 0.03:
        return "SAD"
    if brow_eye > 0.16 and mouth_open > 0.018:
        return "NERVOUS"
    return "NORMAL"


def get_lip_contour_mesh(frame: np.ndarray, landmarks: Any) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    lip_ids = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
    ]
    points = []
    for idx in lip_ids:
        point = landmarks.landmark[idx]
        x = int(point.x * w)
        y = int(point.y * h)
        points.append([x, y])
    if len(points) < 3:
        return None, None
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    x, y, bw, bh = cv2.boundingRect(contour)
    return contour, (x, y, x + bw, y + bh)


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
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    denom = area_a + area_b - inter_area
    if denom <= 1e-6:
        return 0.0
    return inter_area / denom


def bbox_intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    return float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))


def get_hand_boxes(frame_shape: tuple[int, int, int], hands_result: Any) -> list[tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if not hands_result or not getattr(hands_result, "multi_hand_landmarks", None):
        return boxes
    for hand_lm in hands_result.multi_hand_landmarks:
        xs = [int(pt.x * w) for pt in hand_lm.landmark]
        ys = [int(pt.y * h) for pt in hand_lm.landmark]
        if not xs or not ys:
            continue
        x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
        y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
        pad_x = int((x1 - x0) * HAND_BOX_PADDING_RATIO) + 2
        pad_y = int((y1 - y0) * HAND_BOX_PADDING_RATIO) + 2
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w - 1, x1 + pad_x)
        y1 = min(h - 1, y1 + pad_y)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def is_mouth_occluded_by_hand(
    mouth_box: Optional[tuple[int, int, int, int]],
    hand_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float
) -> bool:
    if mouth_box is None:
        return False
    mx0, my0, mx1, my1 = mouth_box
    mouth_area = float(max(1, (mx1 - mx0) * (my1 - my0)))
    for hand_box in hand_boxes:
        iou = bbox_iou(mouth_box, hand_box)
        coverage = bbox_intersection_area(mouth_box, hand_box) / mouth_area
        if iou > iou_threshold or coverage >= MOUTH_OCCLUSION_COVERAGE_THRESHOLD:
            return True
    return False


def create_face_mesh_backend():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        ), "mediapipe_face_mesh"
    return None, "opencv_fallback"


def create_hands_backend():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ), "mediapipe_hands"
    return None, "hands_unavailable"


def run_live_voice_overlay() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    mic = MicrophoneVoiceMonitor(
        sample_rate=MIC_SAMPLE_RATE,
        block_size=MIC_BLOCK_SIZE,
        threshold=MIC_THRESHOLD,
        device_index=MIC_DEVICE_INDEX
    )
    mic_error: Optional[str] = None
    try:
        mic.start()
    except Exception as error:
        mic_error = str(error)

    face_mesh, vision_backend = create_face_mesh_backend()
    hands_detector, hands_backend = create_hands_backend()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    previous_mouth_roi: Optional[np.ndarray] = None
    smoothed_audio_rms = 0.0
    smoothed_lip_metric = 0.0
    smoothed_flow_metric = 0.0
    previous_mar = 0.0
    gate_stability_streak = 0
    flow_gate_window: deque[bool] = deque(maxlen=GATE_FLOW_WINDOW_FRAMES)
    audio_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    lip_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    last_face_bbox: Optional[tuple[int, int, int, int]] = None
    last_face_time = 0.0
    last_lip_box: Optional[tuple[int, int, int, int]] = None
    last_lip_time = 0.0

    height_w = deque(maxlen=WINDOW_SIZE)
    movement_w = deque(maxlen=WINDOW_SIZE)
    feature_w = deque(maxlen=WINDOW_SIZE)
    last_speak = None
    session_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            t = time.time() - session_start

            lips_detected = False
            lip_activity = False
            mahal_value: Optional[float] = None
            mar_value = 0.0
            mar_delta = 0.0
            optical_flow_value = 0.0
            corr_score: Optional[float] = None
            corr_lag = 0
            texture_std: Optional[float] = None
            gate_components = {
                "face_ok": False,
                "texture_ok": False,
                "flow_ok": False,
                "mar_ok": False,
                "stability_ok": False,
            }
            expression = "NORMAL"
            active_lip_threshold = MAHAL_THRESHOLD if face_mesh is not None else LIP_MOTION_THRESHOLD
            lip_box: Optional[tuple[int, int, int, int]] = None
            lip_contour: Optional[np.ndarray] = None
            hand_boxes: list[tuple[int, int, int, int]] = []
            mouth_occluded = False
            mouth_hidden = False
            face_mesh_detected = False
            hands_detected = False
            if face_mesh is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if hands_detector is not None:
                    hands_result = hands_detector.process(image_rgb)
                    hand_boxes = get_hand_boxes(frame.shape, hands_result)
                    hands_detected = len(hand_boxes) > 0
                res = face_mesh.process(image_rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    p = lm.landmark[200]
                    if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                        face_mesh_detected = True
                    if face_mesh_detected:
                        features = extract_lip_features(lm, image_rgb.shape, height_w, movement_w, feature_w)
                        mahal_value = float(calculate_mahalanobis(feature_w, features))
                        lip_activity, last_speak = detect_speaking(features, t, feature_w, last_speak)
                        mar_value = compute_mar_mesh(lm)
                        mar_delta = abs(mar_value - previous_mar)
                        previous_mar = mar_value
                        expression = classify_expression_mesh(lm)
                        lip_contour, lip_box = get_lip_contour_mesh(frame, lm)
                        if lip_box is not None:
                            x0, y0, x1, y1 = lip_box
                            roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                            if roi.size > 0:
                                texture_std = float(np.std(roi))
                            curr_mouth_roi = extract_mouth_roi_gray(frame, lip_box)
                            optical_flow_value = compute_optical_flow_intensity(previous_mouth_roi, curr_mouth_roi)
                            previous_mouth_roi = curr_mouth_roi
                lips_detected, gate_stability_streak, gate_components = evaluate_multi_signal_gate(
                    face_mesh_detected,
                    texture_std,
                    optical_flow_value,
                    mar_value,
                    flow_gate_window,
                    gate_stability_streak
                )
                mouth_hidden = face_mesh_detected and (not gate_components["texture_ok"])
                if not face_mesh_detected:
                    previous_mouth_roi = None
                    previous_mar = 0.0
            else:
                if hands_detector is not None:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hands_result = hands_detector.process(image_rgb)
                    hand_boxes = get_hand_boxes(frame.shape, hands_result)
                    hands_detected = len(hand_boxes) > 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    faces = face_cascade_alt.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    faces = face_cascade_profile.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

                selected_face: Optional[tuple[int, int, int, int]] = None
                if len(faces) > 0:
                    selected_face = max(faces, key=lambda item: item[2] * item[3])
                    last_face_bbox = selected_face
                    last_face_time = t
                elif last_face_bbox is not None and (t - last_face_time) < 0.6:
                    selected_face = last_face_bbox

                if selected_face is not None:
                    lips_detected = True
                    x, y, w, h = selected_face
                    mouth_y0 = y + int(h * 0.55)
                    mouth_y1 = y + h
                    mouth_x0 = x + int(w * 0.20)
                    mouth_x1 = x + int(w * 0.80)
                    lip_box = (mouth_x0, mouth_y0, mouth_x1, mouth_y1)
                    mouth_roi = gray[mouth_y0:mouth_y1, mouth_x0:mouth_x1]

                    motion_value = 0.0
                    if mouth_roi.size > 0:
                        if float(np.std(mouth_roi)) < MOUTH_HIDDEN_TEXTURE_THRESHOLD:
                            mouth_hidden = True
                        mouth_roi = cv2.resize(mouth_roi, (64, 32))
                        if previous_mouth_roi is not None:
                            delta = cv2.absdiff(mouth_roi, previous_mouth_roi)
                            motion_value = float(np.mean(delta))
                        previous_mouth_roi = mouth_roi

                    mahal_value = motion_value
                    optical_flow_value = motion_value
                    lip_activity = motion_value >= LIP_MOTION_THRESHOLD
                    face_roi = gray[y:y + h, x:x + w]
                    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
                    expression = "HAPPY" if len(smiles) > 0 else "NORMAL"

            mouth_occluded = is_mouth_occluded_by_hand(lip_box, hand_boxes, HAND_MOUTH_IOU_THRESHOLD)
            if mouth_occluded:
                lips_detected = False
                mouth_hidden = False

            if lips_detected and lip_box is not None:
                last_lip_box = lip_box
                last_lip_time = t
            elif (not lips_detected) and last_lip_box is not None and (t - last_lip_time) < 0.9:
                lip_box = last_lip_box
                mouth_hidden = True

            lip_color = (0, 0, 255) if (mouth_hidden or not lips_detected) else (0, 255, 0)
            if lip_contour is not None:
                cv2.polylines(frame, [lip_contour], True, lip_color, 1, cv2.LINE_AA)
            if lip_box is not None:
                x0, y0, x1, y1 = lip_box
                cv2.rectangle(frame, (x0, y0), (x1, y1), lip_color, 1)
            for hand_box in hand_boxes:
                hx0, hy0, hx1, hy1 = hand_box
                cv2.rectangle(frame, (hx0, hy0), (hx1, hy1), (255, 255, 0), 1)

            audio_rms = mic.current_rms() if mic_error is None else 0.0
            smoothed_audio_rms = ema(smoothed_audio_rms, audio_rms, EMA_ALPHA)
            audio_activity = smoothed_audio_rms >= MIC_THRESHOLD if mic_error is None else False
            if mahal_value is not None:
                smoothed_lip_metric = ema(smoothed_lip_metric, mahal_value, EMA_ALPHA)
            else:
                smoothed_lip_metric = ema(smoothed_lip_metric, 0.0, EMA_ALPHA)
            smoothed_flow_metric = ema(smoothed_flow_metric, optical_flow_value, EMA_ALPHA)
            lip_activity = smoothed_lip_metric >= active_lip_threshold if lips_detected else False

            rule_mouth_open = (mar_value >= RULE_MAR_THRESHOLD) or (mar_delta >= RULE_MAR_DELTA_THRESHOLD)
            rule_status = classify_rule_based(rule_mouth_open, audio_activity)

            optical_motion_high = smoothed_flow_metric >= OPTICAL_FLOW_THRESHOLD
            optical_status = classify_optical_flow(optical_motion_high, audio_activity)

            audio_energy_window.append(smoothed_audio_rms)
            lip_energy_window.append(smoothed_flow_metric)
            corr_score, corr_lag = compute_cross_correlation_score(
                audio_energy_window,
                lip_energy_window,
                CORR_MAX_LAG_FRAMES
            )
            corr_status = classify_av_correlation(corr_score, CORR_THRESHOLD)

            nervous_signal = lips_detected and lip_activity and (not audio_activity) and (
                smoothed_lip_metric >= (active_lip_threshold * NERVOUS_LIP_MULTIPLIER)
            )
            if nervous_signal:
                expression = "NERVOUS"

            if not lips_detected:
                if mouth_occluded:
                    sync_status = "MOUTH_OCCLUDED"
                    expression = "MOUTH_OCCLUDED"
                elif face_mesh is not None and gate_components["face_ok"]:
                    sync_status = "GATED_OUT"
                    expression = "GATED_OUT"
                else:
                    sync_status = "NO_FACE"
                    expression = "NO_FACE"
            elif mouth_hidden:
                sync_status = "MOUTH_HIDDEN"
                expression = "MOUTH_HIDDEN"
            else:
                sync_status = majority_sync_status(rule_status, optical_status, corr_status)

            lines = [
                f"Audio Voice: {'SPEAKING' if audio_activity else 'SILENT'}",
                f"Audio RMS: {smoothed_audio_rms:.5f} (thr={MIC_THRESHOLD:.5f})",
                "Mouth Occluded: YES" if mouth_occluded else f"Lips Detected: {'YES' if lips_detected else 'NO'}",
                f"Hands Detected: {'YES' if hands_detected else 'NO'}",
                f"Occlusion Thr: IoU>{HAND_MOUTH_IOU_THRESHOLD:.2f} or Coverage>{MOUTH_OCCLUSION_COVERAGE_THRESHOLD:.2f}",
                f"Gate Face/Tex/Flow/MAR/Stable: {int(gate_components['face_ok'])}/{int(gate_components['texture_ok'])}/{int(gate_components['flow_ok'])}/{int(gate_components['mar_ok'])}/{int(gate_components['stability_ok'])}",
                f"Lip Activity: {'SPEAKING' if lip_activity else 'SILENT'}",
                f"Lip Metric: {smoothed_lip_metric:.4f} (thr={active_lip_threshold:.4f})",
                f"Rule-Based: {rule_status} (MAR={mar_value:.3f}, d={mar_delta:.3f})",
                f"Optical Flow: {optical_status} (flow={smoothed_flow_metric:.3f}, thr={OPTICAL_FLOW_THRESHOLD:.3f})",
                f"AV Corr: {corr_status} (score={corr_score if corr_score is not None else float('nan'):.3f}, lag={corr_lag})",
                f"LipSync Status: {sync_status} (ensemble)",
                f"Expression: {expression}",
                f"Vision Backend: {vision_backend}",
                f"Hands Backend: {hands_backend}",
                "Mic: ERROR (check device/index)" if mic_error else "Mic: OK",
                "Press Q to quit",
            ]
            draw_overlay(frame, lines)
            cv2.imshow("Live Voice + LipSync Overlay", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        mic.stop()
        cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if hands_detector is not None:
            hands_detector.close()
        cv2.destroyAllWindows()

# ---- Async Chunk Processing ----
async def process_chunk_async(
    start_f: int,
    end_f: int,
    video_path: str,
    audio_path: str,
    fps: float,
    vad_segments: list[tuple[float, float]]
) -> list[dict]:
    return await asyncio.to_thread(process_chunk, (start_f, end_f, video_path, audio_path, fps, vad_segments))

# ---- Chunk Processing ----
def process_chunk(
    args: tuple[int, int, str, str, float, list[tuple[float, float]]]
) -> list[dict]:
    start_f, end_f, video_path, audio_path, fps, vad_segments = args
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    height_w = deque(maxlen=WINDOW_SIZE)
    movement_w = deque(maxlen=WINDOW_SIZE)
    feature_w = deque(maxlen=WINDOW_SIZE)
    last_speak = None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    curr_sec = int(start_f / fps)

    lip_seen = []
    rule_sync = []
    optical_sync = []
    corr_sync = []
    corr_scores = []
    mar_values = []
    flow_values = []
    mahal_values = []
    audio_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    lip_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    previous_mouth_roi: Optional[np.ndarray] = None
    previous_mar = 0.0
    gate_stability_streak = 0
    flow_gate_window: deque[bool] = deque(maxlen=GATE_FLOW_WINDOW_FRAMES)
    out = []

    while cap.isOpened() and frame_idx < end_f:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        t = frame_idx / fps
        sec = int(t)

        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img)

        seen = False
        lip_box: Optional[tuple[int, int, int, int]] = None
        hand_boxes: list[tuple[int, int, int, int]] = []
        texture_std: Optional[float] = None
        mar_value = 0.0
        mar_delta = 0.0
        motion_value = 0.0
        mouth_occluded = False
        hands_result = hands_detector.process(img)
        hand_boxes = get_hand_boxes(frame.shape, hands_result)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            p = lm.landmark[200]
            if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                seen = True
                _, lip_box = get_lip_contour_mesh(frame, lm)
                mar_value = compute_mar_mesh(lm)
                mar_delta = abs(mar_value - previous_mar)
                previous_mar = mar_value
                if lip_box is not None:
                    x0, y0, x1, y1 = lip_box
                    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                    if roi.size > 0:
                        texture_std = float(np.std(roi))
                    curr_mouth_roi = extract_mouth_roi_gray(frame, lip_box)
                    motion_value = compute_optical_flow_intensity(previous_mouth_roi, curr_mouth_roi)
                    previous_mouth_roi = curr_mouth_roi

        gated_seen, gate_stability_streak, _ = evaluate_multi_signal_gate(
            seen,
            texture_std,
            motion_value,
            mar_value,
            flow_gate_window,
            gate_stability_streak
        )
        mouth_occluded = is_mouth_occluded_by_hand(lip_box, hand_boxes, HAND_MOUTH_IOU_THRESHOLD)
        if mouth_occluded:
            gated_seen = False
        if not seen:
            previous_mouth_roi = None
            previous_mar = 0.0
        lip_seen.append(gated_seen)

        if seen and gated_seen:
            features = extract_lip_features(lm, img.shape, height_w, movement_w, feature_w)
            mahal_val = calculate_mahalanobis(feature_w, features)
            vad_b = is_vad_speaking(t, vad_segments)
            mouth_open = (mar_value >= RULE_MAR_THRESHOLD) or (mar_delta >= RULE_MAR_DELTA_THRESHOLD)
            rb = classify_rule_based(mouth_open, vad_b)

            op = classify_optical_flow(motion_value >= OPTICAL_FLOW_THRESHOLD, vad_b)

            audio_energy_window.append(1.0 if vad_b else 0.0)
            lip_energy_window.append(motion_value)
            corr_score, _ = compute_cross_correlation_score(audio_energy_window, lip_energy_window, CORR_MAX_LAG_FRAMES)
            av = classify_av_correlation(corr_score, CORR_THRESHOLD)

            rule_sync.append(rb == "VALID")
            optical_sync.append(op == "SYNCED")
            corr_sync.append(av == "SYNCED")
            mahal_values.append(float(mahal_val))
            corr_scores.append(corr_score if corr_score is not None else np.nan)
            mar_values.append(mar_value)
            flow_values.append(motion_value)
        else:
            rule_sync.append(False)
            optical_sync.append(False)
            corr_sync.append(False)
            mahal_values.append(np.nan)
            corr_scores.append(np.nan)
            mar_values.append(np.nan)
            flow_values.append(np.nan)

        if sec > curr_sec:
            if not any(lip_seen):
                out.append({
                    "Time (s)": curr_sec,
                    "VAD Speaking": "No Lips Detected",
                    "Mahalanobis Status": "No Lips Detected",
                    "Rule-Based": "No Lips Detected",
                    "Optical Flow": "No Lips Detected",
                    "AV Correlation": "No Lips Detected",
                    "MAR Mean": np.nan,
                    "Flow Mean": np.nan,
                    "Corr Score Mean": np.nan,
                    "Ensemble": "NO_FACE",
                })
            else:
                rb_ratio = float(np.mean(rule_sync)) if rule_sync else 0.0
                op_ratio = float(np.mean(optical_sync)) if optical_sync else 0.0
                av_ratio = float(np.mean(corr_sync)) if corr_sync else 0.0
                rb_state = "SYNCED" if rb_ratio >= 0.5 else "NOT_SYNCED"
                op_state = "SYNCED" if op_ratio >= 0.5 else "NOT_SYNCED"
                av_state = "SYNCED" if av_ratio >= 0.5 else "NOT_SYNCED"
                ensemble = majority_sync_status(
                    "VALID" if rb_state == "SYNCED" else "SUSPICIOUS_AUDIO_NO_MOUTH_MOVEMENT",
                    op_state,
                    av_state
                )
                out.append({
                    "Time (s)": curr_sec,
                    "VAD Speaking": any(rule_sync),
                    "Mahalanobis Status": safe_nanmean(mahal_values),
                    "Rule-Based": rb_state,
                    "Optical Flow": op_state,
                    "AV Correlation": av_state,
                    "MAR Mean": safe_nanmean(mar_values),
                    "Flow Mean": safe_nanmean(flow_values),
                    "Corr Score Mean": safe_nanmean(corr_scores),
                    "Ensemble": ensemble,
                })
            curr_sec = sec
            lip_seen, rule_sync, optical_sync, corr_sync = [], [], [], []
            corr_scores, mar_values, flow_values, mahal_values = [], [], [], []

    cap.release()
    face_mesh.close()
    hands_detector.close()
    return out



# ---- Main Async Entry ----
async def main() -> None:
    if LIVE_MODE:
        run_live_voice_overlay()
        return

    if Pipeline is Any:
        raise ImportError(
            "pyannote.audio is required when live_mode is False. "
            "Use Python 3.10/3.11 and install pyannote.audio, or set live_mode: True."
        )

    audio, duration = extract_audio(VIDEO_PATH)
    pipeline = Pipeline.from_pretrained(CONFIG_PATH)
    vad_segments = get_vad_segments(pipeline, audio)

    if ONLY_VAD:
        results = []
        for t in range(duration + 1):
            status = is_vad_speaking(t, vad_segments)
            results.append({"Time (s)": t, "VAD Speaking": status})
        # export_combined_csv(results, only_vad=True)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        chunk_sz = total_frames // NUM_CORES
        tasks = [
            process_chunk_async(
                i * chunk_sz,
                (i + 1) * chunk_sz if i < NUM_CORES - 1 else total_frames,
                VIDEO_PATH, audio, fps, vad_segments
            ) for i in range(NUM_CORES)
        ]

        chunk_results = await asyncio.gather(*tasks)
        flat = [item for sub in chunk_results for item in sub]
        # export_combined_csv(flat, only_vad=False)

        df = pd.DataFrame(flat)
        df = df.rename(columns={
            'Time (s)': 'timestamp',
            'VAD Speaking': 'vad_status',
            'Mahalanobis Status': 'mahalanobis',
            'Rule-Based': 'rule_based',
            'Optical Flow': 'optical_flow',
            'AV Correlation': 'av_correlation',
            'MAR Mean': 'mar_mean',
            'Flow Mean': 'flow_mean',
            'Corr Score Mean': 'corr_score_mean',
            'Ensemble': 'ensemble',
        })
        df.to_csv('raw_metrics.csv', index=False)
        print('Saved raw_metrics.csv with timestamp, vad_status, mahalanobis, rule_based, optical_flow, av_correlation, mar_mean, flow_mean, corr_score_mean, ensemble columns.')

    if os.path.exists(audio):
        os.remove(audio)

if __name__ == "__main__":
    asyncio.run(main())
