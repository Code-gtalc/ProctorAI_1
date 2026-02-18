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


def create_face_mesh_backend():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        ), "mediapipe_face_mesh"
    return None, "opencv_fallback"


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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    previous_mouth_roi: Optional[np.ndarray] = None
    smoothed_audio_rms = 0.0
    smoothed_lip_metric = 0.0
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
            expression = "NORMAL"
            active_lip_threshold = MAHAL_THRESHOLD if face_mesh is not None else LIP_MOTION_THRESHOLD
            lip_box: Optional[tuple[int, int, int, int]] = None
            lip_contour: Optional[np.ndarray] = None
            mouth_hidden = False
            if face_mesh is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(image_rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    p = lm.landmark[200]
                    if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                        lips_detected = True
                    if lips_detected:
                        features = extract_lip_features(lm, image_rgb.shape, height_w, movement_w, feature_w)
                        mahal_value = float(calculate_mahalanobis(feature_w, features))
                        lip_activity, last_speak = detect_speaking(features, t, feature_w, last_speak)
                        expression = classify_expression_mesh(lm)
                        lip_contour, lip_box = get_lip_contour_mesh(frame, lm)
                        if lip_box is not None:
                            x0, y0, x1, y1 = lip_box
                            roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                            if roi.size > 0 and float(np.std(roi)) < MOUTH_HIDDEN_TEXTURE_THRESHOLD:
                                mouth_hidden = True
            else:
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
                    lip_activity = motion_value >= LIP_MOTION_THRESHOLD
                    face_roi = gray[y:y + h, x:x + w]
                    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
                    expression = "HAPPY" if len(smiles) > 0 else "NORMAL"

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

            audio_rms = mic.current_rms() if mic_error is None else 0.0
            smoothed_audio_rms = ema(smoothed_audio_rms, audio_rms, EMA_ALPHA)
            audio_activity = smoothed_audio_rms >= MIC_THRESHOLD if mic_error is None else False
            if mahal_value is not None:
                smoothed_lip_metric = ema(smoothed_lip_metric, mahal_value, EMA_ALPHA)
            else:
                smoothed_lip_metric = ema(smoothed_lip_metric, 0.0, EMA_ALPHA)
            lip_activity = smoothed_lip_metric >= active_lip_threshold if lips_detected else False

            nervous_signal = lips_detected and lip_activity and (not audio_activity) and (
                smoothed_lip_metric >= (active_lip_threshold * NERVOUS_LIP_MULTIPLIER)
            )
            if nervous_signal:
                expression = "NERVOUS"

            if not lips_detected:
                sync_status = "NO_FACE"
                expression = "NO_FACE"
            elif mouth_hidden:
                sync_status = "MOUTH_HIDDEN"
                expression = "MOUTH_HIDDEN"
            else:
                sync_status = "SYNCED" if lip_activity == audio_activity else "NOT_SYNCED"

            lines = [
                f"Audio Voice: {'SPEAKING' if audio_activity else 'SILENT'}",
                f"Audio RMS: {smoothed_audio_rms:.5f} (thr={MIC_THRESHOLD:.5f})",
                f"Lips Detected: {'YES' if lips_detected else 'NO'}",
                f"Lip Activity: {'SPEAKING' if lip_activity else 'SILENT'}",
                f"Lip Metric: {smoothed_lip_metric:.4f} (thr={active_lip_threshold:.4f})",
                f"LipSync Status: {sync_status}",
                f"Expression: {expression}",
                f"Vision Backend: {vision_backend}",
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
    height_w = deque(maxlen=WINDOW_SIZE)
    movement_w = deque(maxlen=WINDOW_SIZE)
    feature_w = deque(maxlen=WINDOW_SIZE)
    last_speak = None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    curr_sec = int(start_f / fps)

    lip_seen = []
    lip_vad = []
    lip_mah = []
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
        mahal_val = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            p = lm.landmark[200]
            if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                seen = True
        lip_seen.append(seen)

        if seen:
            features = extract_lip_features(lm, img.shape, height_w, movement_w, feature_w)
            vad_b = is_vad_speaking(t, vad_segments)
            if vad_b:
                mahal_val = calculate_mahalanobis(feature_w, features)
                lip_mah.append(mahal_val)
            lip_vad.append(vad_b)

        if sec > curr_sec:
            if not any(lip_seen):
                out.append({"Time (s)": curr_sec, "VAD Speaking": "No Lips Detected", "Mahalanobis Status": "No Lips Detected"})
            else:
                if any(lip_vad):
                    mah_val = float(np.mean([v for v in lip_mah if v is not None])) if lip_mah else None
                    out.append({"Time (s)": curr_sec, "VAD Speaking": True, "Mahalanobis Status": mah_val})
                else:
                    out.append({"Time (s)": curr_sec, "VAD Speaking": False, "Mahalanobis Status": "No Issues"})
            curr_sec = sec
            lip_seen, lip_vad, lip_mah = [], [], []

    cap.release()
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
            'Mahalanobis Status': 'mahalanobis'
        })
        df.to_csv('raw_metrics.csv', index=False)
        print('Saved raw_metrics.csv with timestamp, vad_status, mahalanobis columns.')

    if os.path.exists(audio):
        os.remove(audio)

if __name__ == "__main__":
    asyncio.run(main())
