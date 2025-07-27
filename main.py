import os
import urllib.request
import mediapipe as mp
import cv2 as cv
import numpy as np
from adafruit_servokit import ServoKit
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, servo_channel: int = None):
        self.servo_channel = servo_channel
        self.current_value = 0
        self.kit = ServoKit(channels=16) if servo_channel is not None else None
    
    @abstractmethod
    def process_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> float:
        """Process landmarks and return calculated value"""
        pass
    
    @abstractmethod
    def draw_visualization(self, frame: np.ndarray, landmarks, frame_shape: Tuple[int, int]):
        """Draw visualization on frame"""
        pass
    
    def control_servo(self, value: float):
        """Control servo with calculated value"""
        if self.kit and self.servo_channel is not None:
            clamped_value = max(0, min(180, value))
            self.kit.servo[self.servo_channel].angle = clamped_value
    
    def get_status_text(self) -> str:
        """Return status text for display"""
        return f"Value: {self.current_value:.1f}"

class PoseDetector(BaseDetector):
    """Base class for pose-based detectors"""
    
    def __init__(self, model_path: str = "pose_landmarker.task", servo_channel: int = None):
        super().__init__(servo_channel)
        self.model_path = model_path
        self.landmarker = None
        self._setup_landmarker()
    
    def _setup_landmarker(self):
        """Setup pose landmarker"""
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def detect(self, mp_image, timestamp_ms):
        """Detect pose landmarks"""
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

class HandDetector(BaseDetector):
    """Base class for hand-based detectors"""
    
    def __init__(self, model_path: str = "hand_landmarker.task", servo_channel: int = None):
        super().__init__(servo_channel)
        self.model_path = model_path
        self.landmarker = None
        self._setup_landmarker()
        self._download_model_if_needed()
    
    def _download_model_if_needed(self):
        """Download hand model if it doesn't exist"""
        if not os.path.exists(self.model_path):
            print("Downloading hand landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, self.model_path)
            print("Hand model downloaded!")
    
    def _setup_landmarker(self):
        """Setup hand landmarker"""
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
    
    def detect(self, mp_image, timestamp_ms):
        """Detect hand landmarks"""
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

class LeftArmAngleDetector(PoseDetector):
    """Detects left arm angle (shoulder-elbow-wrist)"""
    
    def __init__(self, servo_channel: int = 0):
        super().__init__(servo_channel=servo_channel)
        self.LEFT_SHOULDER = 11
        self.LEFT_ELBOW = 13
        self.LEFT_WRIST = 15
    
    def _calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180 / np.pi)
        
        if angle > 180:
            angle = 360 - angle
        
        return angle
    
    def process_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> float:
        """Process pose landmarks to get arm angle"""
        if len(landmarks) > max(self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST):
            shoulder = [landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y]
            elbow = [landmarks[self.LEFT_ELBOW].x, landmarks[self.LEFT_ELBOW].y]
            wrist = [landmarks[self.LEFT_WRIST].x, landmarks[self.LEFT_WRIST].y]
            
            angle = self._calculate_angle(shoulder, elbow, wrist)
            self.current_value = angle
            self.control_servo(angle)
            return angle
        return 0
    
    def draw_visualization(self, frame: np.ndarray, landmarks, frame_shape: Tuple[int, int]):
        """Draw arm visualization"""
        h, w = frame_shape
        
        for connection in mp.solutions.pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                

                if (start_idx in [self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST] and 
                    end_idx in [self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST]):
                    cv.line(frame, start_point, end_point, (0, 255, 255), 3)
                else:
                    cv.line(frame, start_point, end_point, (0, 255, 0), 1)
        
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            if idx in [self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST]:
                cv.circle(frame, (x, y), 6, (255, 255, 0), -1)
                cv.circle(frame, (x, y), 8, (0, 0, 255), 2)
            else:
                cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        if len(landmarks) > self.LEFT_ELBOW:
            elbow = landmarks[self.LEFT_ELBOW]
            elbow_px = (int(elbow.x * w), int(elbow.y * h))
            cv.putText(frame, f"{self.current_value:.1f}°", 
                      elbow_px, cv.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 255, 255), 3, cv.LINE_AA)
            cv.putText(frame, f"{self.current_value:.1f}°", 
                      elbow_px, cv.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 0, 0), 1, cv.LINE_AA)
    
    def get_status_text(self) -> str:
        return f"Left Arm Angle: {self.current_value:.1f}°"

class ThumbIndexDistanceDetector(HandDetector):
    """Detects distance between thumb and index finger"""
    
    def __init__(self, servo_channel: int = 1):
        super().__init__(servo_channel=servo_channel)
        self.INDEX_FINGER_TIP = 8
        self.THUMB_TIP = 4
    
    def _calculate_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate distance between two points"""
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)
    
    def _map_distance(self, distance: float) -> float:
        """Map normalized distance to real-world measurement"""
        return distance * 90 / 0.17
    
    def _distance_to_servo_angle(self, distance: float) -> float:
        """Convert distance to servo angle"""
        return distance / 1.3
    
    def process_landmarks(self, hand_landmarks_list, frame_shape: Tuple[int, int]) -> float:
        """Process hand landmarks to get finger distance"""
        if not hand_landmarks_list:
            return 0
        
        # Use first detected hand
        hand_landmarks = hand_landmarks_list[0]
        
        if len(hand_landmarks) > max(self.INDEX_FINGER_TIP, self.THUMB_TIP):
            index = [hand_landmarks[self.INDEX_FINGER_TIP].x, 
                    hand_landmarks[self.INDEX_FINGER_TIP].y]
            thumb = [hand_landmarks[self.THUMB_TIP].x, 
                    hand_landmarks[self.THUMB_TIP].y]
            
            distance = self._calculate_distance(index, thumb)
            mapped_distance = self._map_distance(distance)
            
            self.current_value = mapped_distance
            servo_angle = self._distance_to_servo_angle(mapped_distance)
            self.control_servo(servo_angle)
            
            return mapped_distance
        return 0
    
    def draw_visualization(self, frame: np.ndarray, hand_landmarks_list, frame_shape: Tuple[int, int]):
        """Draw hand visualization"""
        if not hand_landmarks_list:
            return
        
        h, w = frame_shape
        
        for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    cv.line(frame, start_point, end_point, (128, 255, 128), 1)
            
            for idx, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                if idx == self.INDEX_FINGER_TIP:
                    cv.circle(frame, (x, y), 6, (0, 255, 0), -1)
                elif idx == self.THUMB_TIP:
                    cv.circle(frame, (x, y), 6, (0, 0, 255), -1)
                else:
                    cv.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
            if len(hand_landmarks) > max(self.INDEX_FINGER_TIP, self.THUMB_TIP):
                index = hand_landmarks[self.INDEX_FINGER_TIP]
                thumb = hand_landmarks[self.THUMB_TIP]
                
                index_px = (int(index.x * w), int(index.y * h))
                thumb_px = (int(thumb.x * w), int(thumb.y * h))
                
                cv.line(frame, thumb_px, index_px, (255, 255, 0), 2)
                
                mid_point = ((index_px[0] + thumb_px[0]) // 2, 
                            (index_px[1] + thumb_px[1]) // 2)
                cv.putText(frame, f"{self.current_value:.1f}mm", 
                          mid_point, cv.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(frame, f"{self.current_value:.1f}mm", 
                          mid_point, cv.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 0, 0), 1, cv.LINE_AA)
    
    def get_status_text(self) -> str:
        return f"Thumb-Index Distance: {self.current_value:.1f}mm"

class RightArmAngleDetector(PoseDetector):
    """Detects right arm angle (shoulder-elbow-wrist)"""
    
    def __init__(self, servo_channel: int = 2):
        super().__init__(servo_channel=servo_channel)
        self.RIGHT_SHOULDER = 12
        self.RIGHT_ELBOW = 14
        self.RIGHT_WRIST = 16
    
    def _calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180 / np.pi)
        
        if angle > 180:
            angle = 360 - angle
        
        return angle
    
    def process_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> float:
        """Process pose landmarks to get right arm angle"""
        if len(landmarks) > max(self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST):
            shoulder = [landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y]
            elbow = [landmarks[self.RIGHT_ELBOW].x, landmarks[self.RIGHT_ELBOW].y]
            wrist = [landmarks[self.RIGHT_WRIST].x, landmarks[self.RIGHT_WRIST].y]
            
            angle = self._calculate_angle(shoulder, elbow, wrist)
            self.current_value = angle
            self.control_servo(angle)
            return angle
        return 0
    
    def draw_visualization(self, frame: np.ndarray, landmarks, frame_shape: Tuple[int, int]):
        """Draw right arm visualization"""
        h, w = frame_shape
        
        connections = [(self.RIGHT_SHOULDER, self.RIGHT_ELBOW), 
                      (self.RIGHT_ELBOW, self.RIGHT_WRIST)]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv.line(frame, start_point, end_point, (255, 0, 255), 3)  
        
        for idx in [self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST]:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv.circle(frame, (x, y), 6, (255, 0, 255), -1)
                cv.circle(frame, (x, y), 8, (128, 0, 128), 2)
        
        if len(landmarks) > self.RIGHT_ELBOW:
            elbow = landmarks[self.RIGHT_ELBOW]
            elbow_px = (int(elbow.x * w), int(elbow.y * h))
            cv.putText(frame, f"{self.current_value:.1f}°", 
                      (elbow_px[0] + 20, elbow_px[1]), cv.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 255, 255), 3, cv.LINE_AA)
            cv.putText(frame, f"{self.current_value:.1f}°", 
                      (elbow_px[0] + 20, elbow_px[1]), cv.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 0, 255), 1, cv.LINE_AA)
    
    def get_status_text(self) -> str:
        return f"Right Arm Angle: {self.current_value:.1f}°"

class DetectionSystem:
    """Main system that coordinates all detectors"""
    
    def __init__(self):
        self.pose_detectors: List[PoseDetector] = []
        self.hand_detectors: List[HandDetector] = []
        self.pose_landmarker = None
        self.hand_landmarker = None
        self._setup_landmarkers()
    
    def _setup_landmarkers(self):
        """Setup MediaPipe landmarkers"""
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
    
    def add_detector(self, detector: BaseDetector):
        """Add a detector to the system"""
        if isinstance(detector, PoseDetector):
            self.pose_detectors.append(detector)
        elif isinstance(detector, HandDetector):
            self.hand_detectors.append(detector)
    
    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Process a single frame with all detectors"""
        h, w = frame.shape[:2]
        frame_shape = (h, w)
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
        
        annotated_frame = frame.copy()
        
        if self.pose_detectors:
            pose_results = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
                landmarks = pose_results.pose_landmarks[0]
                
                for detector in self.pose_detectors:
                    detector.process_landmarks(landmarks, frame_shape)
                    detector.draw_visualization(annotated_frame, landmarks, frame_shape)
        
        if self.hand_detectors:
            hand_results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_results.hand_landmarks:
                for detector in self.hand_detectors:
                    detector.process_landmarks(hand_results.hand_landmarks, frame_shape)
                    detector.draw_visualization(annotated_frame, hand_results.hand_landmarks, frame_shape)
        
        y_offset = 30
        for detector in self.pose_detectors + self.hand_detectors:
            cv.putText(annotated_frame, detector.get_status_text(), 
                      (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 255, 0), 2, cv.LINE_AA)
            y_offset += 30
        
        return annotated_frame
    
    def run(self):
        """Run the detection system"""
        cap = cv.VideoCapture(0)
        frame_timestamp_ms = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = self.process_frame(frame, frame_timestamp_ms)
            frame_timestamp_ms += 33
            
            cv.imshow('Modular Detection System', annotated_frame)
            
            if cv.waitKey(5) & 0xFF == 27: 
                break
        
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    system = DetectionSystem()
    
    system.add_detector(LeftArmAngleDetector(servo_channel=0))
    system.add_detector(ThumbIndexDistanceDetector(servo_channel=1))
    system.add_detector(RightArmAngleDetector(servo_channel=2)) 
    
    system.run()