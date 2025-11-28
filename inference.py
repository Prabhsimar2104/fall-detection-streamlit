# inference.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv
from cloudinary_upload import upload_fall_image
import time
import math
import requests

load_dotenv()

# Config / env
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:4000")
FALL_ALERT_TOKEN = os.getenv("FALL_ALERT_TOKEN", "secret_token_for_fall_detection")
USER_ID = int(os.getenv("USER_ID", "1"))

# Load model once
model = YOLO(MODEL_PATH)

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# load classnames
with open("classes.txt", "r") as f:
    classnames = f.read().splitlines()

# Simple detect_fall logic copied/adapted from your main_enhanced.py
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def detect_fall_advanced(landmarks):
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        body_vertical = abs(shoulder_center_y - hip_center_y)
        hip_height = 1.0 - hip_center_y
        if body_vertical < 0.15 and hip_height < 0.3:
            return True, "horizontal", 0.95
        if hip_height < 0.2:
            return True, "collapsed", 0.90
        return False, None, 0.0
    except Exception:
        return False, None, 0.0

def infer_frame(frame_bgr):
    """
    Input: BGR frame (numpy array)
    Output: dict: {fall: bool, type: str|None, confidence: float}
    """
    try:
        # downscale to speed up
        small = cv2.resize(frame_bgr, (640, 480))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # YOLO detection
        yres = model(small)
        person_detected = False

        for info in yres:
            for box in info.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = classnames[cls]
                if name == "person" and conf > 0.6:
                    person_detected = True

        # Pose-based check if person exists
        if person_detected:
            pose_res = pose.process(rgb)
            if pose_res.pose_landmarks:
                is_fallen, fall_type, conf = detect_fall_advanced(pose_res.pose_landmarks.landmark)
                return {"fall": bool(is_fallen), "type": fall_type, "confidence": float(conf)}
        return {"fall": False, "type": None, "confidence": 0.0}
    except Exception as e:
        return {"fall": False, "type": None, "confidence": 0.0, "error": str(e)}

def infer_video(path, sample_rate=5):
    """
    Process a video file on disk. Sample every `sample_rate` frames.
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
    return {"frames_sampled": frames//sample_rate, "falls": fall_count, "fall_detected": fall_count > 0}
