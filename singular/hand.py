import mediapipe as mp
import cv2 as cv
import numpy as np
import os
import urllib.request
from adafruit_servokit import ServoKit

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_filename = "hand_landmarker.task"

kit = ServoKit(channels=16)
if not os.path.exists(model_filename):
    print("Downloading hand landmarker model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(model_url, model_filename)
    print("Model downloaded!")

# Distance calculation function
def Distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def Mapping(num):
    return num * 90 / 0.17

def Angle(num):
    return num/1.3

INDEX_FINGER_TIP = 8
THUMB_TIP = 4

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_filename),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
        
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        annotated_frame = frame.copy()
        
        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                h, w = frame.shape[:2]
                
                if len(hand_landmarks) > max(INDEX_FINGER_TIP, THUMB_TIP):
                    index = [hand_landmarks[INDEX_FINGER_TIP].x, 
                            hand_landmarks[INDEX_FINGER_TIP].y]
                    thumb = [hand_landmarks[THUMB_TIP].x, 
                            hand_landmarks[THUMB_TIP].y]
                    
                    res = Distance(index, thumb)
                    mapped = Mapping(res)

                    kit.servo[1].angle = Angle(mapped)                   
                    print(f"Hand {hand_idx + 1}: Distance = {res:.4f}, Mapped = {mapped:.2f}mm")
                    
                    index_px = (int(index[0] * w), int(index[1] * h))
                    thumb_px = (int(thumb[0] * w), int(thumb[1] * h))
                    
                    cv.line(annotated_frame, thumb_px, index_px, (255, 255, 0), 3)
                    
                    cv.circle(annotated_frame, index_px, 8, (0, 255, 0), -1)
                    cv.circle(annotated_frame, thumb_px, 8, (0, 0, 255), -1)
                    
                    mid_point = ((index_px[0] + thumb_px[0]) // 2, 
                                (index_px[1] + thumb_px[1]) // 2)
                    cv.putText(annotated_frame, f"{mapped:.1f}mm", 
                              mid_point, 
                              cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 3, cv.LINE_AA)
                    cv.putText(annotated_frame, f"{mapped:.1f}mm", 
                              mid_point, 
                              cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 0), 1, cv.LINE_AA)
                
                connections = mp.solutions.hands.HAND_CONNECTIONS
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        
                        cv.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
                
                for idx, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                   
                    if idx == INDEX_FINGER_TIP:
                        cv.circle(annotated_frame, (x, y), 8, (0, 255, 0), -1)
                    elif idx == THUMB_TIP:
                        cv.circle(annotated_frame, (x, y), 8, (0, 0, 255), -1)
                    else:
                        cv.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)
                
              
                hand_text = f"Hand {hand_idx + 1}: {mapped:.1f}mm"
                cv.putText(annotated_frame, hand_text, 
                          (10, 30 + hand_idx * 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2, cv.LINE_AA)
        
        cv.imshow('Hand Landmarker', annotated_frame)

        if cv.waitKey(1) == 27: 
            break

    cap.release()
    cv.destroyAllWindows()