# FALL-DETECTION/main_enhanced.py
# Enhanced fall detection using MediaPipe Pose Estimation + YOLO

import cv2
import cvzone
import math
from ultralytics import YOLO
import mediapipe as mp
import requests
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from cloudinary_upload import upload_fall_image
import numpy as np

load_dotenv()

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:4000')
FALL_ALERT_TOKEN = os.getenv('FALL_ALERT_TOKEN', 'secret_token_for_fall_detection')
USER_ID = int(os.getenv('USER_ID', '1'))

# Fall detection settings
FALL_CONFIDENCE_THRESHOLD = 80
FALL_COOLDOWN = 30
CONSECUTIVE_FRAMES_THRESHOLD = 5
INACTIVITY_THRESHOLD = 60  # seconds - detect prolonged lying down

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Track last alert time and states
last_alert_time = 0
fall_detected_frames = 0
last_movement_time = time.time()
is_person_down = False
down_start_time = None

# Initialize video capture and YOLO model
cap = cv2.VideoCapture(0)
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def detect_fall_advanced(landmarks, frame_height):
    """
    Advanced fall detection using pose landmarks
    
    Returns: (is_fallen, fall_type, confidence)
    fall_type: 'horizontal', 'sitting', 'crouching', None
    """
    try:
        # Get key landmarks
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate center points
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate body orientation (vertical distance)
        body_vertical = abs(shoulder_center_y - hip_center_y)
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )
        right_knee_angle = calculate_angle(
            [right_hip.x, right_hip.y],
            [right_knee.x, right_knee.y],
            [right_ankle.x, right_ankle.y]
        )
        
        # Calculate hip height relative to frame (lower = closer to ground)
        hip_height = 1.0 - hip_center_y
        
        # Fall Detection Logic
        
        # 1. Horizontal fall (lying down)
        if body_vertical < 0.15 and hip_height < 0.3:
            return True, 'horizontal', 0.95
        
        # 2. Person is very low to ground (collapsed)
        if hip_height < 0.2:
            return True, 'collapsed', 0.90
        
        # 3. Both knees bent and person low (potential fall)
        if (left_knee_angle < 90 and right_knee_angle < 90) and hip_height < 0.35:
            return True, 'crouching_fall', 0.75
        
        # 4. Unusual body angle (tilted)
        shoulder_hip_angle = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_hip_angle > 0.2 and hip_height < 0.4:
            return True, 'tilted', 0.80
        
        return False, None, 0.0
        
    except Exception as e:
        print(f"Error in pose analysis: {e}")
        return False, None, 0.0

def detect_inactivity(is_fallen, fall_type):
    """Detect prolonged inactivity (person hasn't moved for a while)"""
    global is_person_down, down_start_time
    
    if is_fallen:
        if not is_person_down:
            is_person_down = True
            down_start_time = time.time()
        else:
            # Check if person has been down too long
            down_duration = time.time() - down_start_time
            if down_duration > INACTIVITY_THRESHOLD:
                return True, down_duration
    else:
        is_person_down = False
        down_start_time = None
    
    return False, 0

def send_fall_alert(confidence, fall_type, frame=None, inactivity_duration=None):
    """Send fall detection alert to backend API"""
    global last_alert_time
    
    current_time = time.time()
    
    if current_time - last_alert_time < FALL_COOLDOWN:
        return False
    
    try:
        alert_data = {
            'userId': USER_ID,
            'timestamp': datetime.now().isoformat(),
            'confidence': float(confidence)
        }
        
        # Add fall type and inactivity info to alert
        if fall_type:
            alert_data['fallType'] = fall_type
        if inactivity_duration:
            alert_data['inactivityDuration'] = int(inactivity_duration)
        
        # Save and upload frame
        image_url = None
        if frame is not None:
            os.makedirs('alerts', exist_ok=True)
            filename = f"alerts/fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved: {filename}")
            
            if os.getenv('CLOUDINARY_CLOUD_NAME'):
                print("☁️  Uploading to Cloudinary...")
                image_url = upload_fall_image(filename, USER_ID)
                if image_url:
                    alert_data['imageUrl'] = image_url
        
        headers = {
            'X-API-KEY': FALL_ALERT_TOKEN,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/notify/fall-alert",
            json=alert_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 201:
            print("✅ Fall alert sent successfully!")
            print(f"   Type: {fall_type}, Confidence: {confidence*100:.1f}%")
            last_alert_time = current_time
            return True
        else:
            print(f"❌ Failed to send alert. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending fall alert: {str(e)}")
        return False

print("🚀 Enhanced Fall Detection System Started (YOLO + MediaPipe Pose)")
print(f"📡 Backend URL: {BACKEND_URL}")
print(f"👤 Monitoring User ID: {USER_ID}")
print(f"⏱️  Cooldown: {FALL_COOLDOWN}s | Inactivity Threshold: {INACTIVITY_THRESHOLD}s")
print("\n🎥 Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to read frame from camera")
        break
    
    frame = cv2.resize(frame, (980, 740))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Pose
    pose_results = pose.process(frame_rgb)
    
    current_frame_fall = False
    fall_type = None
    fall_confidence = 0.0
    
    # YOLO detection for person
    yolo_results = model(frame)
    person_detected = False
    
    for info in yolo_results:
        parameters = info.boxes
        for box in parameters:
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_name = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            
            if conf > FALL_CONFIDENCE_THRESHOLD and class_name == 'person':
                person_detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cvzone.cornerRect(frame, [x1, y1, x2-x1, y2-y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'Person {conf}%', 
                                 [x1 + 8, y1 - 12], thickness=2, scale=1.5)
    
    # Advanced pose-based fall detection
    if person_detected and pose_results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        
        # Detect fall using pose
        is_fallen, fall_type, fall_confidence = detect_fall_advanced(
            pose_results.pose_landmarks.landmark,
            frame.shape[0]
        )
        
        if is_fallen:
            current_frame_fall = True
            cv2.rectangle(frame, (0, 0), (980, 740), (0, 0, 255), 5)
            cvzone.putTextRect(frame, f'🚨 FALL DETECTED - {fall_type}!', 
                             [50, 50], thickness=3, scale=2.5, colorR=(0, 0, 255))
            
            # Check for prolonged inactivity
            is_inactive, duration = detect_inactivity(True, fall_type)
            if is_inactive:
                cvzone.putTextRect(frame, f'⚠️ INACTIVE: {int(duration)}s', 
                                 [50, 120], thickness=2, scale=2, colorR=(255, 0, 0))
    else:
        detect_inactivity(False, None)
    
    # Track consecutive frames
    if current_frame_fall:
        fall_detected_frames += 1
        
        if fall_detected_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
            print(f"\n⚠️  FALL CONFIRMED ({fall_detected_frames} consecutive frames)")
            print(f"   Type: {fall_type}, Confidence: {fall_confidence*100:.1f}%")
            send_fall_alert(fall_confidence, fall_type, frame)
            fall_detected_frames = 0
    else:
        fall_detected_frames = 0
    
    # Display status
    if fall_detected_frames > 0:
        cvzone.putTextRect(frame, f'Confirming... {fall_detected_frames}/{CONSECUTIVE_FRAMES_THRESHOLD}', 
                         [50, 680], thickness=2, scale=1.8, colorR=(255, 128, 0))
    
    # Display cooldown
    if time.time() - last_alert_time < FALL_COOLDOWN:
        remaining = int(FALL_COOLDOWN - (time.time() - last_alert_time))
        cvzone.putTextRect(frame, f'Cooldown: {remaining}s', 
                         [750, 50], thickness=2, scale=1.5, colorR=(128, 128, 128))
    
    cv2.imshow('Enhanced Fall Detection System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n👋 Shutting down...")
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ Enhanced Fall Detection System stopped")