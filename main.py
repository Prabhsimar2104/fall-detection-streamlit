import cv2
import cvzone
import math
from ultralytics import YOLO
import requests
from datetime import datetime
import time
import os
import logging
from dotenv import load_dotenv
from cloudinary_upload import upload_fall_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:4000')
FALL_ALERT_TOKEN = os.getenv('FALL_ALERT_TOKEN', 'secret_token_for_fall_detection')
USER_ID = int(os.getenv('USER_ID', '1'))

# Fall detection settings
PERSON_CONFIDENCE_THRESHOLD = 70  # Reduced from 80 for better detection
FALL_COOLDOWN = 30
CONSECUTIVE_FRAMES_THRESHOLD = 5
ASPECT_RATIO_THRESHOLD = 0.75  # Width/Height ratio for fall detection

# Track state
last_alert_time = 0
fall_detected_frames = 0
fps_counter = 0
prev_time = time.time()

# Initialize video capture and model
cap = cv2.VideoCapture(0)
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def get_frame_dimensions(frame):
    """Get optimal frame dimensions"""
    height, width = frame.shape[:2]
    # Use power of 2 for better performance
    target_width = 640
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    return target_width, target_height

def send_fall_alert(confidence, frame=None):
    """Send fall detection alert to backend API"""
    global last_alert_time
    
    current_time = time.time()
    
    if current_time - last_alert_time < FALL_COOLDOWN:
        remaining = int(FALL_COOLDOWN - (current_time - last_alert_time))
        logger.debug(f"Cooldown active. {remaining}s remaining")
        return False
    
    try:
        alert_data = {
            'userId': USER_ID,
            'timestamp': datetime.now().isoformat(),
            'confidence': float(confidence) / 100.0,
            'detectionMethod': 'yolo_bbox'
        }
        
        image_url = None
        if frame is not None:
            os.makedirs('alerts', exist_ok=True)
            filename = f"alerts/fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Frame saved: {filename}")
            
            if os.getenv('CLOUDINARY_CLOUD_NAME'):
                logger.info("Uploading to Cloudinary...")
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
            logger.info("Fall alert sent successfully!")
            logger.info(f"Response: {response.json()}")
            last_alert_time = current_time
            return True
        else:
            logger.warning(f"Failed to send alert. Status: {response.status_code}")
            logger.warning(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Backend server not reachable")
        return False
    except requests.exceptions.Timeout:
        logger.error("Request timeout: Backend took too long to respond")
        return False
    except Exception as e:
        logger.error(f"Error sending fall alert: {str(e)}")
        return False

def detect_fall(x1, y1, x2, y2):
    """
    Detect if person has fallen based on aspect ratio
    Returns True if fall detected (person is horizontal)
    """
    height = y2 - y1
    width = x2 - x1
    
    if height <= 0 or width <= 0:
        return False
    
    aspect_ratio = width / height
    
    # Person is horizontal/lying down when aspect ratio > threshold
    is_horizontal = aspect_ratio > ASPECT_RATIO_THRESHOLD
    
    return is_horizontal

logger.info("🚀 Fall Detection System Started")
logger.info(f"🔡 Backend URL: {BACKEND_URL}")
logger.info(f"👤 Monitoring User ID: {USER_ID}")
logger.info(f"⏱️  Cooldown Period: {FALL_COOLDOWN} seconds")
logger.info("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        logger.error("Failed to read frame from camera")
        break
    
    # Adaptive frame resizing
    target_width, target_height = get_frame_dimensions(frame)
    frame = cv2.resize(frame, (target_width, target_height))
    
    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    
    results = model(frame, conf=0.5)
    
    current_frame_fall = False
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            
            height = y2 - y1
            width = x2 - x1
            
            # Only process people with sufficient confidence
            if conf > PERSON_CONFIDENCE_THRESHOLD and class_detect == 'person':
                # Draw bounding box
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect} {conf}%', 
                                 [x1 + 8, y1 - 12], thickness=2, scale=1.5)
                
                # Check for fall
                if detect_fall(x1, y1, x2, y2):
                    current_frame_fall = True
                    
                    # Draw alert
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cvzone.putTextRect(frame, '🚨 FALL DETECTED!', 
                                     [50, 50], thickness=3, scale=3, 
                                     colorR=(0, 0, 255))
                    
                    # Debug dimensions
                    aspect_ratio = width / height if height > 0 else 0
                    cvzone.putTextRect(frame, f'W:{width} H:{height} AR:{aspect_ratio:.2f}', 
                                     [x1 + 8, y2 + 30], thickness=1, scale=1)
    
    # Track consecutive frames
    if current_frame_fall:
        fall_detected_frames += 1
        
        if fall_detected_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
            logger.warning(f"FALL CONFIRMED ({fall_detected_frames} consecutive frames)")
            send_fall_alert(PERSON_CONFIDENCE_THRESHOLD, frame)
            fall_detected_frames = 0
    else:
        fall_detected_frames = 0
    
    # Display frame confirmation status
    if fall_detected_frames > 0:
        cvzone.putTextRect(frame, f'Confirming... {fall_detected_frames}/{CONSECUTIVE_FRAMES_THRESHOLD}', 
                         [50, 120], thickness=2, scale=2, colorR=(255, 128, 0))
    
    # Display cooldown timer
    if time.time() - last_alert_time < FALL_COOLDOWN:
        remaining = int(FALL_COOLDOWN - (time.time() - last_alert_time))
        cvzone.putTextRect(frame, f'Cooldown: {remaining}s', 
                         [target_width - 250, 50], thickness=2, scale=1.5, colorR=(128, 128, 128))
    
    # Display FPS
    cvzone.putTextRect(frame, f'FPS: {fps:.1f}', 
                     [target_width - 250, target_height - 40], thickness=2, scale=1.5, colorR=(0, 255, 0))
    
    cv2.imshow('Fall Detection System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Shutting down...")
        break

cap.release()
cv2.destroyAllWindows()
logger.info("✅ Fall Detection System stopped")