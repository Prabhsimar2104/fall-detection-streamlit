# inference.py  -- YOLO-only inference (no mediapipe / no cvzone)
import os
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")

# Load YOLO model once (on import)
model = YOLO(MODEL_PATH)

# Load class names
CLASSES_FILE = "classes.txt"
if not os.path.exists(CLASSES_FILE):
    # fallback default (should not happen if classes.txt present)
    classnames = []
else:
    with open(CLASSES_FILE, "r") as f:
        classnames = [c.strip() for c in f.read().splitlines()]

def infer_frame(frame_bgr):
    """
    Run YOLO inference on a single BGR frame (numpy array).
    Uses a simple heuristic: if person bbox width > height => horizontal => fall.
    Returns: dict { fall: bool, type: str|None, confidence: float }
    """
    try:
        # Resize to speed up inference
        small = cv2.resize(frame_bgr, (640, 480))
        results = model(small)  # ultralytics model call

        for info in results:
            for box in info.boxes:
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                name = classnames[cls_idx] if cls_idx < len(classnames) else str(cls_idx)
                # Only consider people with reasonable confidence
                if name == "person" and conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    # Heuristic: width > height implies horizontal person
                    if w > h:
                        return {"fall": True, "type": "horizontal", "confidence": float(conf)}
        return {"fall": False, "type": None, "confidence": 0.0}
    except Exception as e:
        return {"fall": False, "type": None, "confidence": 0.0, "error": str(e)}

def infer_video(path, sample_rate=5):
    """
    Process video file at `path`. Sample every `sample_rate` frames to speed up.
    Returns a summary dict.
    """
    cap = cv2.VideoCapture(path)
    frames = 0
    fall_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if frames % sample_rate != 0:
            continue
        r = infer_frame(frame)
        if r.get("fall"):
            fall_count += 1
    cap.release()
    return {"frames_sampled": frames // sample_rate, "falls": fall_count, "fall_detected": fall_count > 0}
