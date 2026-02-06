#!/usr/bin/env python3
"""
Extract SMPL body poses and MANO hand poses from RealSense bag files.
Outputs 6D rotation representation for machine learning.

Uses Hydra for configuration management.

SMPL body: 23 joints + 1 global orientation = 24 x 6D = 144 parameters
MANO hands: 15 joints per hand x 6D = 90 parameters per hand

Total per frame: 144 (body) + 90 (left hand) + 90 (right hand) + 3 (root translation) = 327 parameters
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig, OmegaConf


# ============================================================================
# MediaPipe to SMPL/MANO Joint Mapping
# ============================================================================

# MediaPipe Pose has 33 landmarks, SMPL has 24 joints
# We need to map and infer some joints
MEDIAPIPE_TO_SMPL_MAPPING = {
    # SMPL joint index: MediaPipe landmark index
    0: None,  # Pelvis - computed as midpoint of hips
    1: 24,    # Left hip
    2: 23,    # Right hip
    3: None,  # Spine1 - interpolated
    4: 26,    # Left knee
    5: 25,    # Right knee
    6: None,  # Spine2 - interpolated
    7: 28,    # Left ankle
    8: 27,    # Right ankle
    9: None,  # Spine3 - interpolated
    10: 32,   # Left foot
    11: 31,   # Right foot
    12: None, # Neck - computed
    13: None, # Left collar - computed
    14: None, # Right collar - computed
    15: 0,    # Head - nose
    16: 12,   # Left shoulder
    17: 11,   # Right shoulder
    18: 14,   # Left elbow
    19: 13,   # Right elbow
    20: 16,   # Left wrist
    21: 15,   # Right wrist
    22: 20,   # Left hand (approx)
    23: 19,   # Right hand (approx)
}

# MANO hand has 21 landmarks from MediaPipe which we map to 16 joints (wrist + 15 finger joints)
# MANO structure: wrist (0) + 3 joints per finger × 5 fingers
MEDIAPIPE_HAND_TO_MANO_MAPPING = {
    0: 0,     # Wrist
    # Thumb (4 joints in MediaPipe, 3 in MANO - skip base)
    1: 1,     # Thumb CMC
    2: 2,     # Thumb MCP
    3: 3,     # Thumb IP
    4: 3,     # Thumb tip -> use IP
    # Index finger
    5: 4,     # Index MCP
    6: 5,     # Index PIP
    7: 6,     # Index DIP
    8: 6,     # Index tip -> use DIP
    # Middle finger
    9: 7,     # Middle MCP
    10: 8,    # Middle PIP
    11: 9,    # Middle DIP
    12: 9,    # Middle tip -> use DIP
    # Ring finger
    13: 10,   # Ring MCP
    14: 11,   # Ring PIP
    15: 12,   # Ring DIP
    16: 12,   # Ring tip -> use DIP
    # Pinky
    17: 13,   # Pinky MCP
    18: 14,   # Pinky PIP
    19: 15,   # Pinky DIP
    20: 15,   # Pinky tip -> use DIP
}

# SMPL kinematic tree (parent joint for each joint)
SMPL_PARENT = [
    -1,  # 0: Pelvis (root)
    0,   # 1: Left hip
    0,   # 2: Right hip
    0,   # 3: Spine1
    1,   # 4: Left knee
    2,   # 5: Right knee
    3,   # 6: Spine2
    4,   # 7: Left ankle
    5,   # 8: Right ankle
    6,   # 9: Spine3
    7,   # 10: Left foot
    8,   # 11: Right foot
    9,   # 12: Neck
    9,   # 13: Left collar
    9,   # 14: Right collar
    12,  # 15: Head
    13,  # 16: Left shoulder
    14,  # 17: Right shoulder
    16,  # 18: Left elbow
    17,  # 19: Right elbow
    18,  # 20: Left wrist
    19,  # 21: Right wrist
    20,  # 22: Left hand
    21,  # 23: Right hand
]

# MANO kinematic tree (parent for each joint)
MANO_PARENT = [
    -1,  # 0: Wrist (root)
    0,   # 1: Thumb CMC
    1,   # 2: Thumb MCP
    2,   # 3: Thumb IP
    0,   # 4: Index MCP
    4,   # 5: Index PIP
    5,   # 6: Index DIP
    0,   # 7: Middle MCP
    7,   # 8: Middle PIP
    8,   # 9: Middle DIP
    0,   # 10: Ring MCP
    10,  # 11: Ring PIP
    11,  # 12: Ring DIP
    0,   # 13: Pinky MCP
    13,  # 14: Pinky PIP
    14,  # 15: Pinky DIP
]


# ============================================================================
# 6D Rotation Utilities
# ============================================================================

def rotation_matrix_to_6d(R_mat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D representation.
    
    Args:
        R_mat: 3x3 rotation matrix
        
    Returns:
        6D vector [r1, r2] where r1 and r2 are first two columns of R
    """
    return R_mat[:, :2].reshape(-1)  # Shape: (6,)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Convert 6D representation to rotation matrix using Gram-Schmidt.
    
    Args:
        d6: 6D vector [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z]
        
    Returns:
        3x3 rotation matrix
    """
    a1 = d6[:3]
    a2 = d6[3:6]
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    
    return np.stack([b1, b2, b3], axis=1)


def compute_rotation_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates v1 to v2.
    
    Args:
        v1: Source vector (3,)
        v2: Target vector (3,)
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Use scipy's rotation library for robustness
    # Find rotation that aligns v1 to v2
    cross = np.cross(v1_norm, v2_norm)
    dot = np.dot(v1_norm, v2_norm)
    
    if np.linalg.norm(cross) < 1e-6:  # Vectors are parallel
        if dot > 0:
            return np.eye(3)
        else:
            # Vectors are opposite - return 180 degree rotation around perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1_norm[0]) < 0.9 else np.array([0, 1, 0])
            perp = np.cross(v1_norm, perp)
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            return R.from_rotvec(np.pi * perp).as_matrix()
    
    # Rodrigues formula
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    axis = cross / (np.linalg.norm(cross) + 1e-8)
    
    return R.from_rotvec(angle * axis).as_matrix()


# ============================================================================
# MediaPipe Keypoints to SMPL/MANO Conversion
# ============================================================================

def mediapipe_pose_to_smpl_positions(mp_landmarks: List) -> np.ndarray:
    """
    Convert MediaPipe pose landmarks to SMPL joint positions.
    
    Args:
        mp_landmarks: MediaPipe pose landmarks (33 points)
        
    Returns:
        SMPL joint positions (24, 3)
    """
    smpl_joints = np.zeros((24, 3))
    
    for smpl_idx, mp_idx in MEDIAPIPE_TO_SMPL_MAPPING.items():
        if mp_idx is not None:
            smpl_joints[smpl_idx] = [
                mp_landmarks[mp_idx].x,
                mp_landmarks[mp_idx].y,
                mp_landmarks[mp_idx].z
            ]
    
    # Compute derived joints
    # Pelvis (0) = midpoint of hips
    smpl_joints[0] = (smpl_joints[1] + smpl_joints[2]) / 2
    
    # Spine interpolation
    pelvis = smpl_joints[0]
    # Shoulders midpoint
    shoulders_mid = (smpl_joints[16] + smpl_joints[17]) / 2
    
    smpl_joints[3] = pelvis + (shoulders_mid - pelvis) * 0.33  # Spine1
    smpl_joints[6] = pelvis + (shoulders_mid - pelvis) * 0.66  # Spine2
    smpl_joints[9] = shoulders_mid  # Spine3
    
    # Neck (12) = midpoint between spine3 and head
    smpl_joints[12] = (smpl_joints[9] + smpl_joints[15]) / 2
    
    # Collars
    smpl_joints[13] = smpl_joints[9] + (smpl_joints[16] - smpl_joints[9]) * 0.5  # Left collar
    smpl_joints[14] = smpl_joints[9] + (smpl_joints[17] - smpl_joints[9]) * 0.5  # Right collar
    
    return smpl_joints


def mediapipe_hand_to_mano_positions(mp_hand_landmarks: List) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks to MANO joint positions.
    
    Args:
        mp_hand_landmarks: MediaPipe hand landmarks (21 points)
        
    Returns:
        MANO joint positions (16, 3)
    """
    mano_joints = np.zeros((16, 3))
    
    # Count occurrences to average duplicates
    joint_counts = np.zeros(16)
    
    for mp_idx, mano_idx in MEDIAPIPE_HAND_TO_MANO_MAPPING.items():
        pos = np.array([
            mp_hand_landmarks[mp_idx].x,
            mp_hand_landmarks[mp_idx].y,
            mp_hand_landmarks[mp_idx].z
        ])
        mano_joints[mano_idx] += pos
        joint_counts[mano_idx] += 1
    
    # Average duplicates
    for i in range(16):
        if joint_counts[i] > 0:
            mano_joints[i] /= joint_counts[i]
    
    return mano_joints


def positions_to_6d_rotations(positions: np.ndarray, parents: List[int], reference_pose: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert joint positions to 6D rotation representation.
    
    Args:
        positions: Joint positions (N, 3)
        parents: Parent indices for each joint
        reference_pose: Reference T-pose positions (N, 3). If None, use first frame.
        
    Returns:
        6D rotations (N, 6)
    """
    n_joints = len(positions)
    rotations_6d = np.zeros((n_joints, 6))
    
    # Define reference directions (T-pose/rest pose)
    if reference_pose is None:
        # Use canonical T-pose directions
        reference_directions = np.zeros((n_joints, 3))
        for i in range(n_joints):
            if parents[i] >= 0:
                # Default: child is below parent (for legs) or to the side (for arms)
                reference_directions[i] = np.array([0, -1, 0])  # Point down by default
    else:
        reference_directions = np.zeros((n_joints, 3))
        for i in range(n_joints):
            if parents[i] >= 0:
                bone_vec = reference_pose[i] - reference_pose[parents[i]]
                reference_directions[i] = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
    
    # Compute rotations for each joint
    for i in range(n_joints):
        if parents[i] < 0:
            # Root joint - use identity or global orientation
            rotations_6d[i] = rotation_matrix_to_6d(np.eye(3))
        else:
            parent_idx = parents[i]
            
            # Current bone direction
            bone_vec = positions[i] - positions[parent_idx]
            bone_vec = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
            
            # Reference bone direction
            ref_dir = reference_directions[i]
            ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-8)
            
            # Compute rotation from reference to current
            R_mat = compute_rotation_from_vectors(ref_dir, bone_vec)
            
            # Convert to 6D
            rotations_6d[i] = rotation_matrix_to_6d(R_mat)
    
    return rotations_6d


# ============================================================================
# Main Extraction Pipeline with Hydra
# ============================================================================

class SMPLPoseExtractor:
    """Extract SMPL/MANO poses from RealSense bag files."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the pose extractor.
        
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        log_level = getattr(logging, cfg.logging.level)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if cfg.logging.save_log:
            fh = logging.FileHandler(cfg.logging.log_path)
            fh.setLevel(log_level)
            self.logger.addHandler(fh)
        
        # Model paths
        model_path = Path(cfg.models.path)
        pose_model_path = model_path / cfg.models.pose_model
        hand_model_path = model_path / cfg.models.hand_model
        
        self.logger.info(f"Loading pose model from: {pose_model_path}")
        self.logger.info(f"Loading hand model from: {hand_model_path}")
        
        # Configure MediaPipe
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=cfg.detection.pose.min_detection_confidence,
            min_pose_presence_confidence=cfg.detection.pose.min_presence_confidence,
            min_tracking_confidence=cfg.detection.pose.min_tracking_confidence
        )
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=cfg.detection.hand.num_hands,
            min_hand_detection_confidence=cfg.detection.hand.min_detection_confidence,
            min_hand_presence_confidence=cfg.detection.hand.min_presence_confidence,
            min_tracking_confidence=cfg.detection.hand.min_tracking_confidence
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        # Reference poses (will be set from first valid frame)
        self.reference_smpl_pose = None
        self.reference_mano_left = None
        self.reference_mano_right = None
        
        # Load custom reference if provided
        if cfg.reference.custom_reference is not None:
            self._load_custom_reference(cfg.reference.custom_reference)
    
    def _load_custom_reference(self, ref_path: str):
        """Load custom reference pose from JSON."""
        self.logger.info(f"Loading custom reference from: {ref_path}")
        with open(ref_path, 'r') as f:
            ref_data = json.load(f)
        
        if 'smpl_body' in ref_data and ref_data['smpl_body'] is not None:
            self.reference_smpl_pose = np.array(ref_data['smpl_body']['positions'])
        if 'mano_left' in ref_data and ref_data['mano_left'] is not None:
            self.reference_mano_left = np.array(ref_data['mano_left']['positions'])
        if 'mano_right' in ref_data and ref_data['mano_right'] is not None:
            self.reference_mano_right = np.array(ref_data['mano_right']['positions'])
    
    def extract_from_bag(self, bag_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Extract SMPL/MANO poses from a RealSense bag file.
        
        Args:
            bag_path: Path to .bag file
            output_path: Path to save output JSON (default: bagname_smpl.json)
            
        Returns:
            Dictionary with extracted poses
        """
        bag_path = Path(bag_path)
        if output_path is None:
            output_path = bag_path.parent / f"{bag_path.stem}_smpl.json"
        else:
            output_path = Path(output_path)
        
        self.logger.info(f"Processing: {bag_path}")
        self.logger.info(f"Output will be saved to: {output_path}")
        
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(bag_path))
        
        # Start pipeline
        pipeline.start(config)
        
        # Disable looping and set playback speed
        playback = config.resolve(pipeline).get_device().as_playback()
        playback.set_real_time(self.cfg.processing.real_time)
        
        # Storage
        all_frames_data = []
        stats = {
            'total_frames': 0,
            'pose_detected': 0,
            'left_hand_detected': 0,
            'right_hand_detected': 0
        }
        
        try:
            frame_idx = 0
            processed_count = 0
            
            while True:
                # Check max frames limit
                if self.cfg.processing.max_frames is not None:
                    if processed_count >= self.cfg.processing.max_frames:
                        self.logger.info(f"Reached max frames limit: {self.cfg.processing.max_frames}")
                        break
                
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    self.logger.info("End of bag file reached")
                    break
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                stats['total_frames'] += 1
                
                # Frame skipping
                if frame_idx % self.cfg.processing.frame_skip != 0:
                    frame_idx += 1
                    continue
                
                processed_count += 1
                
                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Convert to MediaPipe format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Detect pose and hands
                pose_result = self.pose_landmarker.detect(mp_image)
                hand_result = self.hand_landmarker.detect(mp_image)
                
                # Process results
                frame_data = self._process_frame(
                    frame_idx, pose_result, hand_result, stats
                )
                
                all_frames_data.append(frame_data)
                
                # Visualization
                if self.cfg.processing.visualize:
                    self._visualize_frame(
                        color_image, frame_idx, frame_data, stats
                    )
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("User quit")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Progress logging
                if processed_count % self.cfg.logging.print_progress_every == 0:
                    self.logger.info(f"Processed {processed_count} frames...")
        
        finally:
            pipeline.stop()
            if self.cfg.processing.visualize:
                cv2.destroyAllWindows()
        
        # Prepare output data
        output_data = self._prepare_output(bag_path, all_frames_data, stats)
        
        # Save to JSON
        self.logger.info(f"Saving to {output_path}...")
        with open(output_path, 'w') as f:
            if self.cfg.output.pretty_print:
                json.dump(output_data, f, indent=2)
            else:
                json.dump(output_data, f)
        
        # Print summary
        self._print_summary(stats)
        
        return output_data
    
    def _process_frame(self, frame_idx: int, pose_result, hand_result, 
                       stats: Dict) -> Dict:
        """Process a single frame and extract SMPL/MANO parameters."""
        frame_data = {
            'frame_idx': frame_idx,
            'smpl_body': None,
            'mano_left': None,
            'mano_right': None,
            'root_translation': None
        }
        
        # Process body pose
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            stats['pose_detected'] += 1
            
            # Convert to SMPL positions
            smpl_positions = mediapipe_pose_to_smpl_positions(
                pose_result.pose_landmarks[0]
            )
            
            # Set reference pose from first valid frame
            if self.reference_smpl_pose is None and self.cfg.reference.use_first_frame:
                self.reference_smpl_pose = smpl_positions.copy()
                self.logger.info(f"Set reference SMPL pose from frame {frame_idx}")
            
            # Convert to 6D rotations
            smpl_rotations_6d = positions_to_6d_rotations(
                smpl_positions, SMPL_PARENT, self.reference_smpl_pose
            )
            
            # Root translation (pelvis position)
            root_translation = smpl_positions[0]
            
            frame_data['smpl_body'] = {
                'rotations_6d': smpl_rotations_6d.tolist(),  # (24, 6)
            }
            
            if self.cfg.output.save_positions:
                frame_data['smpl_body']['positions'] = smpl_positions.tolist()
            
            frame_data['root_translation'] = root_translation.tolist()
        
        # Process hands
        if hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[idx][0].category_name
                
                # Convert to MANO positions
                mano_positions = mediapipe_hand_to_mano_positions(hand_landmarks)
                
                # Set reference pose
                if handedness == 'Left':
                    if self.reference_mano_left is None and self.cfg.reference.use_first_frame:
                        self.reference_mano_left = mano_positions.copy()
                        self.logger.info(f"Set reference MANO left hand pose from frame {frame_idx}")
                    reference = self.reference_mano_left
                    stats['left_hand_detected'] += 1
                else:
                    if self.reference_mano_right is None and self.cfg.reference.use_first_frame:
                        self.reference_mano_right = mano_positions.copy()
                        self.logger.info(f"Set reference MANO right hand pose from frame {frame_idx}")
                    reference = self.reference_mano_right
                    stats['right_hand_detected'] += 1
                
                # Convert to 6D rotations
                mano_rotations_6d = positions_to_6d_rotations(
                    mano_positions, MANO_PARENT, reference
                )
                
                hand_data = {
                    'rotations_6d': mano_rotations_6d.tolist(),  # (16, 6)
                }
                
                if self.cfg.output.save_positions:
                    hand_data['positions'] = mano_positions.tolist()
                
                if handedness == 'Left':
                    frame_data['mano_left'] = hand_data
                else:
                    frame_data['mano_right'] = hand_data
        
        return frame_data
    
    def _prepare_output(self, bag_path: Path, frames_data: List[Dict], 
                       stats: Dict) -> Dict:
        """Prepare output dictionary with metadata."""
        output = {'frames': frames_data}
        
        if self.cfg.output.save_metadata:
            output['metadata'] = {
                'source_bag': str(bag_path),
                'total_frames': stats['total_frames'],
                'pose_detected_frames': stats['pose_detected'],
                'left_hand_detected_frames': stats['left_hand_detected'],
                'right_hand_detected_frames': stats['right_hand_detected'],
                'config': OmegaConf.to_container(self.cfg, resolve=True),
                'format': {
                    'body': 'SMPL (24 joints × 6D rotation = 144 params)',
                    'hands': 'MANO (16 joints × 6D rotation = 96 params per hand)',
                    'total_per_frame': '144 + 96 + 96 + 3 (root translation) = 339 params'
                }
            }
        
        return output
    
    def _visualize_frame(self, color_image: np.ndarray, frame_idx: int,
                        frame_data: Dict, stats: Dict):
        """Visualize extracted pose on frame."""
        vis_image = color_image.copy()
        
        # Draw skeleton if available
        if frame_data['smpl_body'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['smpl_body']['positions'])
            self._draw_skeleton(vis_image, positions, SMPL_PARENT, (0, 255, 0))
        
        if frame_data['mano_left'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['mano_left']['positions'])
            self._draw_skeleton(vis_image, positions, MANO_PARENT, (255, 0, 0))
        
        if frame_data['mano_right'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['mano_right']['positions'])
            self._draw_skeleton(vis_image, positions, MANO_PARENT, (0, 0, 255))
        
        # Add info text
        cv2.putText(vis_image, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Body: {'YES' if frame_data['smpl_body'] else 'NO'}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0) if frame_data['smpl_body'] else (0, 0, 255), 2)
        
        cv2.imshow('SMPL/MANO Extraction', vis_image)
    
    def _draw_skeleton(self, image: np.ndarray, positions: np.ndarray,
                      parents: List[int], color: Tuple[int, int, int]):
        """Draw skeleton on image."""
        height, width = image.shape[:2]
        
        # Draw bones
        for i, parent in enumerate(parents):
            if parent >= 0:
                start = positions[parent]
                end = positions[i]
                
                start_x = int(start[0] * width)
                start_y = int(start[1] * height)
                end_x = int(end[0] * width)
                end_y = int(end[1] * height)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Draw joints
        for pos in positions:
            x = int(pos[0] * width)
            y = int(pos[1] * height)
            cv2.circle(image, (x, y), 4, color, -1)
            cv2.circle(image, (x, y), 5, (255, 255, 255), 1)
    
    def _print_summary(self, stats: Dict):
        """Print extraction summary."""
        total = stats['total_frames']
        self.logger.info("="*60)
        self.logger.info("EXTRACTION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total frames: {total}")
        self.logger.info(f"Body pose detected: {stats['pose_detected']} "
                        f"({100*stats['pose_detected']/max(total,1):.1f}%)")
        self.logger.info(f"Left hand detected: {stats['left_hand_detected']} "
                        f"({100*stats['left_hand_detected']/max(total,1):.1f}%)")
        self.logger.info(f"Right hand detected: {stats['right_hand_detected']} "
                        f"({100*stats['right_hand_detected']/max(total,1):.1f}%)")
        self.logger.info("="*60)
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()


# ============================================================================
# Hydra Main Function
# ============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="pose")
def main(cfg: DictConfig) -> None:
    """
    Main function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Check if bag_file is provided
    if cfg.bag_file is None:
        raise ValueError("bag_file must be specified! Use: python script.py bag_file=path/to/file.bag")
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create extractor
    extractor = SMPLPoseExtractor(cfg)
    
    # Extract poses
    output_data = extractor.extract_from_bag(
        bag_path=cfg.bag_file,
        output_path=cfg.output_path
    )
    
    logger.info(f"Done! Output saved.")


if __name__ == '__main__':
    main()
