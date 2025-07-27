import os
import urllib.request
import mediapipe as mp
import cv2 as cv
import numpy as np
from adafruit_servokit import ServoKit


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_filename = "pose_landmarker.task"

kit = ServoKit(channels=16)

def Angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # middle
    c = np.array(c)  # end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

angle = 0

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_filename),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

with PoseLandmarker.create_from_options(options) as landmarker:
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
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            h, w = frame.shape[:2]
            
            if len(landmarks) > max(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST):
                shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]
                elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y]
                wrist = [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y]
                
                angle = Angle(shoulder, elbow, wrist)
                kit.servo[0].angle = angle  
                
                elbow_px = (int(elbow[0] * w), int(elbow[1] * h))
                
                cv.putText(annotated_frame, f"{angle:.1f}°", 
                          elbow_px, 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.8, (255, 255, 255), 3, cv.LINE_AA)
                cv.putText(annotated_frame, f"{angle:.1f}°", 
                          elbow_px, 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0, 0, 0), 1, cv.LINE_AA)
            
            for connection in mp.solutions.pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    if (start_idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST] and 
                        end_idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
                        cv.line(annotated_frame, start_point, end_point, (0, 255, 255), 4)
                    else:
                        cv.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
            
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                if idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]:
                    cv.circle(annotated_frame, (x, y), 8, (255, 255, 0), -1)
                    cv.circle(annotated_frame, (x, y), 10, (0, 0, 255), 2)
                else:
                    cv.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
        
        cv.putText(annotated_frame, f"Left Arm Angle: {angle:.1f}°", 
                  (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 
                  1, (0, 255, 0), 2, cv.LINE_AA)
        
        cv.imshow('Pose Landmarker', annotated_frame)
        
        if cv.waitKey(5) & 0xFF == 27: 
            break
    
    cap.release()
    cv.destroyAllWindows()