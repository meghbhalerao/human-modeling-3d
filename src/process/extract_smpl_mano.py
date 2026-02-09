#!/usr/bin/env python3
"""
TWO-PASS VERSION - The actually sensible way:

Pass 1: Count total frames
Pass 2: Process those frames

No estimates, no guessing, no infinite loops!
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


# [All the mapping constants - same as before]
MEDIAPIPE_TO_SMPL_MAPPING = {
    0: None, 1: 24, 2: 23, 3: None, 4: 26, 5: 25, 6: None, 7: 28, 8: 27, 9: None,
    10: 32, 11: 31, 12: None, 13: None, 14: None, 15: 0, 16: 12, 17: 11, 18: 14,
    19: 13, 20: 16, 21: 15, 22: 20, 23: 19,
}

MEDIAPIPE_HAND_TO_MANO_MAPPING = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8, 11: 9, 
    12: 9, 13: 10, 14: 11, 15: 12, 16: 12, 17: 13, 18: 14, 19: 15, 20: 15,
}

SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
MANO_PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]


def rotation_matrix_to_6d(R_mat: np.ndarray) -> np.ndarray:
    return R_mat[:, :2].reshape(-1)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def compute_rotation_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(v1_norm, v2_norm)
    dot = np.dot(v1_norm, v2_norm)
    
    if np.linalg.norm(cross) < 1e-6:
        if dot > 0:
            return np.eye(3)
        else:
            perp = np.array([1, 0, 0]) if abs(v1_norm[0]) < 0.9 else np.array([0, 1, 0])
            perp = np.cross(v1_norm, perp)
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            return R.from_rotvec(np.pi * perp).as_matrix()
    
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    axis = cross / (np.linalg.norm(cross) + 1e-8)
    return R.from_rotvec(angle * axis).as_matrix()


def mediapipe_pose_to_smpl_positions(mp_landmarks: List) -> np.ndarray:
    smpl_joints = np.zeros((24, 3))
    for smpl_idx, mp_idx in MEDIAPIPE_TO_SMPL_MAPPING.items():
        if mp_idx is not None:
            smpl_joints[smpl_idx] = [mp_landmarks[mp_idx].x, mp_landmarks[mp_idx].y, mp_landmarks[mp_idx].z]
    smpl_joints[0] = (smpl_joints[1] + smpl_joints[2]) / 2
    pelvis, shoulders_mid = smpl_joints[0], (smpl_joints[16] + smpl_joints[17]) / 2
    smpl_joints[3] = pelvis + (shoulders_mid - pelvis) * 0.33
    smpl_joints[6] = pelvis + (shoulders_mid - pelvis) * 0.66
    smpl_joints[9] = shoulders_mid
    smpl_joints[12] = (smpl_joints[9] + smpl_joints[15]) / 2
    smpl_joints[13] = smpl_joints[9] + (smpl_joints[16] - smpl_joints[9]) * 0.5
    smpl_joints[14] = smpl_joints[9] + (smpl_joints[17] - smpl_joints[9]) * 0.5
    return smpl_joints


def mediapipe_hand_to_mano_positions(mp_hand_landmarks: List) -> np.ndarray:
    mano_joints = np.zeros((16, 3))
    joint_counts = np.zeros(16)
    for mp_idx, mano_idx in MEDIAPIPE_HAND_TO_MANO_MAPPING.items():
        pos = np.array([mp_hand_landmarks[mp_idx].x, mp_hand_landmarks[mp_idx].y, mp_hand_landmarks[mp_idx].z])
        mano_joints[mano_idx] += pos
        joint_counts[mano_idx] += 1
    for i in range(16):
        if joint_counts[i] > 0:
            mano_joints[i] /= joint_counts[i]
    return mano_joints


def positions_to_6d_rotations(positions: np.ndarray, parents: List[int], 
                               reference_pose: Optional[np.ndarray] = None) -> np.ndarray:
    n_joints = len(positions)
    rotations_6d = np.zeros((n_joints, 6))
    
    if reference_pose is None:
        reference_directions = np.zeros((n_joints, 3))
        for i in range(n_joints):
            if parents[i] >= 0:
                reference_directions[i] = np.array([0, -1, 0])
    else:
        reference_directions = np.zeros((n_joints, 3))
        for i in range(n_joints):
            if parents[i] >= 0:
                bone_vec = reference_pose[i] - reference_pose[parents[i]]
                reference_directions[i] = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
    
    for i in range(n_joints):
        if parents[i] < 0:
            rotations_6d[i] = rotation_matrix_to_6d(np.eye(3))
        else:
            parent_idx = parents[i]
            bone_vec = positions[i] - positions[parent_idx]
            bone_vec = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
            ref_dir = reference_directions[i] / (np.linalg.norm(reference_directions[i]) + 1e-8)
            R_mat = compute_rotation_from_vectors(ref_dir, bone_vec)
            rotations_6d[i] = rotation_matrix_to_6d(R_mat)
    
    return rotations_6d


def count_frames_in_bag(bag_path: str) -> int:
    """
    PASS 1: Count the actual number of frames in the bag file.
    
    Detects when the bag file loops by tracking timestamps.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path))
    
    # Start pipeline
    pipeline.start(config)
    
    # Disable real-time to go fast
    device = config.resolve(pipeline).get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    
    frame_count = 0
    seen_timestamps = set()
    first_timestamp = None
    
    try:
        # Count frames and detect looping
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
                
                # Check timestamp to detect loop
                timestamp = frames.get_timestamp()
                
                if first_timestamp is None:
                    first_timestamp = timestamp
                
                # If we've seen this timestamp before, we're looping!
                if timestamp in seen_timestamps:
                    print(f"Loop detected at frame {frame_count} (timestamp {timestamp} seen again)")
                    break
                
                # If timestamp goes backwards, we're looping
                if len(seen_timestamps) > 10 and timestamp < (first_timestamp + 100):
                    print(f"Loop detected at frame {frame_count} (timestamp went backwards)")
                    break
                
                seen_timestamps.add(timestamp)
                
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_count += 1
                    
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Counting frames... {frame_count}")
                    
            except RuntimeError:
                # End of file (timeout)
                print(f"No more frames (timeout)")
                break
    finally:
        pipeline.stop()
    
    print(f"Total frames counted: {frame_count}")
    return frame_count


class SMPLPoseExtractor:
    """Extract SMPL/MANO poses with TWO-PASS approach."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        log_level = getattr(logging, cfg.logging.level)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        if cfg.logging.save_log:
            fh = logging.FileHandler(cfg.logging.log_path)
            fh.setLevel(log_level)
            self.logger.addHandler(fh)
        
        model_path = Path(cfg.models.path)
        pose_model_path = model_path / cfg.models.pose_model
        hand_model_path = model_path / cfg.models.hand_model
        
        self.logger.info(f"Loading pose model from: {pose_model_path}")
        self.logger.info(f"Loading hand model from: {hand_model_path}")
        
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
        
        self.reference_smpl_pose = None
        self.reference_mano_left = None
        self.reference_mano_right = None
        
        if cfg.reference.custom_reference is not None:
            self._load_custom_reference(cfg.reference.custom_reference)
    
    def _load_custom_reference(self, ref_path: str):
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
        Extract SMPL/MANO poses using TWO-PASS approach.
        
        Pass 1: Count frames
        Pass 2: Process frames
        """
        bag_path = Path(bag_path)
        if output_path is None:
            output_path = bag_path.parent / f"{bag_path.stem}_smpl.json"
        else:
            output_path = Path(output_path)
        
        self.logger.info(f"Processing: {bag_path}")
        
        # PASS 1: Count frames (the RIGHT way!)
        self.logger.info("Pass 1: Counting total frames...")
        total_frames = count_frames_in_bag(str(bag_path))
        self.logger.info(f"Total frames in bag file: {total_frames}")
        
        if total_frames == 0:
            raise ValueError(f"No frames found in bag file: {bag_path}")
        
        # PASS 2: Process frames
        self.logger.info(f"Pass 2: Processing {total_frames} frames...")
        self.logger.info(f"Output will be saved to: {output_path}")
        
        # Setup pipeline for second pass
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(bag_path))
        pipeline.start(config)
        
        # Disable real-time for faster processing
        device = config.resolve(pipeline).get_device()
        playback = device.as_playback()
        playback.set_real_time(self.cfg.processing.real_time)
        
        all_frames_data = []
        stats = {'total_frames': 0, 'pose_detected': 0, 'left_hand_detected': 0, 'right_hand_detected': 0}
        
        try:
            processed_count = 0
            
            # NOW we can use a proper FINITE loop!
            for frame_idx in range(total_frames):
                # Check max frames limit from config
                if self.cfg.processing.max_frames is not None:
                    if processed_count >= self.cfg.processing.max_frames:
                        self.logger.info(f"Reached max frames limit: {self.cfg.processing.max_frames}")
                        break
                
                # Get frame
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    self.logger.warning(f"Could not get frame {frame_idx}/{total_frames}")
                    continue
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                stats['total_frames'] += 1
                
                # Frame skipping
                if frame_idx % self.cfg.processing.frame_skip != 0:
                    continue
                
                processed_count += 1
                
                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Detect pose and hands
                pose_result = self.pose_landmarker.detect(mp_image)
                hand_result = self.hand_landmarker.detect(mp_image)
                
                # Process results
                frame_data = self._process_frame(frame_idx, pose_result, hand_result, stats)
                all_frames_data.append(frame_data)
                
                # Visualization
                if self.cfg.processing.visualize:
                    self._visualize_frame(color_image, frame_idx, frame_data, stats, total_frames)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("User quit")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                # Progress logging
                if processed_count % self.cfg.logging.print_progress_every == 0:
                    progress_pct = (frame_idx + 1) / total_frames * 100
                    self.logger.info(f"Processed {processed_count} frames... ({progress_pct:.1f}% complete)")
        
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise
        finally:
            pipeline.stop()
            if self.cfg.processing.visualize:
                cv2.destroyAllWindows()
        
        # Prepare and save output
        output_data = self._prepare_output(bag_path, all_frames_data, stats)
        
        self.logger.info(f"Saving to {output_path}...")
        with open(output_path, 'w') as f:
            if self.cfg.output.pretty_print:
                json.dump(output_data, f, indent=2)
            else:
                json.dump(output_data, f)
        
        self._print_summary(stats)
        return output_data
    
    def _process_frame(self, frame_idx: int, pose_result, hand_result, stats: Dict) -> Dict:
        frame_data = {
            'frame_idx': frame_idx,
            'smpl_body': None,
            'mano_left': None,
            'mano_right': None,
            'root_translation': None
        }
        
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            stats['pose_detected'] += 1
            smpl_positions = mediapipe_pose_to_smpl_positions(pose_result.pose_landmarks[0])
            
            if self.reference_smpl_pose is None and self.cfg.reference.use_first_frame:
                self.reference_smpl_pose = smpl_positions.copy()
                self.logger.info(f"Set reference SMPL pose from frame {frame_idx}")
            
            smpl_rotations_6d = positions_to_6d_rotations(smpl_positions, SMPL_PARENT, self.reference_smpl_pose)
            root_translation = smpl_positions[0]
            
            frame_data['smpl_body'] = {'rotations_6d': smpl_rotations_6d.tolist()}
            if self.cfg.output.save_positions:
                frame_data['smpl_body']['positions'] = smpl_positions.tolist()
            frame_data['root_translation'] = root_translation.tolist()
        
        if hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[idx][0].category_name
                mano_positions = mediapipe_hand_to_mano_positions(hand_landmarks)
                
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
                
                mano_rotations_6d = positions_to_6d_rotations(mano_positions, MANO_PARENT, reference)
                hand_data = {'rotations_6d': mano_rotations_6d.tolist()}
                if self.cfg.output.save_positions:
                    hand_data['positions'] = mano_positions.tolist()
                
                if handedness == 'Left':
                    frame_data['mano_left'] = hand_data
                else:
                    frame_data['mano_right'] = hand_data
        
        return frame_data
    
    def _prepare_output(self, bag_path: Path, frames_data: List[Dict], stats: Dict) -> Dict:
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
                        frame_data: Dict, stats: Dict, total_frames: int):
        vis_image = color_image.copy()
        if frame_data['smpl_body'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['smpl_body']['positions'])
            self._draw_skeleton(vis_image, positions, SMPL_PARENT, (0, 255, 0))
        if frame_data['mano_left'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['mano_left']['positions'])
            self._draw_skeleton(vis_image, positions, MANO_PARENT, (255, 0, 0))
        if frame_data['mano_right'] is not None and self.cfg.output.save_positions:
            positions = np.array(frame_data['mano_right']['positions'])
            self._draw_skeleton(vis_image, positions, MANO_PARENT, (0, 0, 255))
        
        # Show progress
        progress_pct = (frame_idx + 1) / total_frames * 100
        cv2.putText(vis_image, f"Frame: {frame_idx+1}/{total_frames} ({progress_pct:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Body: {'YES' if frame_data['smpl_body'] else 'NO'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if frame_data['smpl_body'] else (0, 0, 255), 2)
        cv2.imshow('SMPL/MANO Extraction', vis_image)
    
    def _draw_skeleton(self, image: np.ndarray, positions: np.ndarray,
                      parents: List[int], color: Tuple[int, int, int]):
        height, width = image.shape[:2]
        for i, parent in enumerate(parents):
            if parent >= 0:
                start, end = positions[parent], positions[i]
                start_x, start_y = int(start[0] * width), int(start[1] * height)
                end_x, end_y = int(end[0] * width), int(end[1] * height)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
        for pos in positions:
            x, y = int(pos[0] * width), int(pos[1] * height)
            cv2.circle(image, (x, y), 4, color, -1)
            cv2.circle(image, (x, y), 5, (255, 255, 255), 1)
    
    def _print_summary(self, stats: Dict):
        total = stats['total_frames']
        self.logger.info("="*60)
        self.logger.info("EXTRACTION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total frames: {total}")
        self.logger.info(f"Body pose detected: {stats['pose_detected']} ({100*stats['pose_detected']/max(total,1):.1f}%)")
        self.logger.info(f"Left hand detected: {stats['left_hand_detected']} ({100*stats['left_hand_detected']/max(total,1):.1f}%)")
        self.logger.info(f"Right hand detected: {stats['right_hand_detected']} ({100*stats['right_hand_detected']/max(total,1):.1f}%)")
        self.logger.info("="*60)
    
    def __del__(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()


@hydra.main(version_base=None, config_path="../configs", config_name="pose")
def main(cfg: DictConfig) -> None:
    if cfg.bag_file is None:
        raise ValueError("bag_file must be specified!")
    
    logger = logging.getLogger(__name__)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    extractor = SMPLPoseExtractor(cfg)
    output_data = extractor.extract_from_bag(bag_path=cfg.bag_file, output_path=cfg.output_path)
    logger.info("Done! Output saved.")


if __name__ == '__main__':
    main()