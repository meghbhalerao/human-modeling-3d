import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define skeleton connections for pose (based on MediaPipe pose model)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # Body
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
]

# Hand connections (21 landmarks per hand)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

def draw_skeleton(image, landmarks, connections, color=(0, 255, 0), thickness=3):
    """Draw skeleton by connecting landmarks"""
    height, width = image.shape[:2]
    
    print(f"Drawing skeleton with {len(landmarks)} landmarks, color: {color}")
    
    # Draw connections (lines) FIRST
    lines_drawn = 0
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            # Draw thicker lines for visibility
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            lines_drawn += 1
    
    print(f"Drew {lines_drawn} lines")
    
    # Draw landmarks (points) on top
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.circle(image, (x, y), 6, (255, 255, 255), 2)  # White outline

# Configure pose landmarker with lower confidence thresholds
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='../../checkpoints/mediapipe_models/pose_landmarker_heavy.task'),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3)

# Configure hand landmarker with lower confidence thresholds
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='../../checkpoints/mediapipe_models/hand_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("/home/mb230/projects/human-modeling-3d/data/TOY/20260206_114125.bag")  # UPDATE THIS PATH
pipeline.start(config)
playback = config.resolve(pipeline).get_device().as_playback()
playback.set_real_time(False)

# Storage for keypoints
all_keypoints = []

# Statistics
pose_detected_count = 0
hand_detected_count = 0
total_frames = 0

with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    
    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            total_frames += 1
            
            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect pose
            pose_result = pose_landmarker.detect(mp_image)
            
            # Detect hands
            hand_result = hand_landmarker.detect(mp_image)
            
            # Extract keypoints
            frame_data = {
                'frame_idx': frame_idx,
                'pose': None,
                'left_hand': None,
                'right_hand': None
            }
            
            # Start with original image
            annotated_image = color_image.copy()
            
            # Draw pose skeleton
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose_detected_count += 1
                print(f"\nFrame {frame_idx}: Pose detected with {len(pose_result.pose_landmarks[0])} landmarks")
                
                frame_data['pose'] = [
                    [lm.x, lm.y, lm.z, lm.visibility] 
                    for lm in pose_result.pose_landmarks[0]
                ]
                
                draw_skeleton(
                    annotated_image, 
                    pose_result.pose_landmarks[0], 
                    POSE_CONNECTIONS,
                    color=(0, 255, 0),  # Green for body
                    thickness=3
                )
            else:
                print(f"Frame {frame_idx}: No pose detected")
            
            # Draw hand skeletons
            if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                hand_detected_count += 1
                print(f"Frame {frame_idx}: {len(hand_result.hand_landmarks)} hand(s) detected")
                
                for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[idx][0].category_name
                    hand_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    
                    if handedness == 'Left':
                        frame_data['left_hand'] = hand_data
                        color = (255, 0, 0)  # Blue for left hand (BGR format)
                        print(f"  Left hand detected")
                    else:
                        frame_data['right_hand'] = hand_data
                        color = (0, 0, 255)  # Red for right hand (BGR format)
                        print(f"  Right hand detected")
                    
                    draw_skeleton(
                        annotated_image,
                        hand_landmarks,
                        HAND_CONNECTIONS,
                        color=color,
                        thickness=3
                    )
            
            all_keypoints.append(frame_data)
            
            # Add frame info text
            cv2.putText(annotated_image, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_image, f"Pose: {'YES' if pose_result.pose_landmarks else 'NO'}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if pose_result.pose_landmarks else (0, 0, 255), 2)
            cv2.putText(annotated_image, f"Hands: {len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Skeleton', annotated_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar to pause
                cv2.waitKey(0)
            
            frame_idx += 1
            
    except RuntimeError as e:
        print(f"Reached end of video or error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# Save keypoints
import json
with open('keypoints.json', 'w') as f:
    json.dump(all_keypoints, f, indent=2)

print(f"\n=== Summary ===")
print(f"Total frames processed: {total_frames}")
print(f"Frames with pose detected: {pose_detected_count} ({100*pose_detected_count/max(total_frames,1):.1f}%)")
print(f"Frames with hands detected: {hand_detected_count} ({100*hand_detected_count/max(total_frames,1):.1f}%)")
print(f"Keypoints saved to keypoints.json")