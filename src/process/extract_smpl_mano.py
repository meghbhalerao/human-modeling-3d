#!/usr/bin/env python3
"""
TWO-PASS VERSION WITH REALSENSE DEPTH SUPPORT AND CHUNKING
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
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime

# [Mapping constants remain the same]
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

# --- Math Utilities ---
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

# --- Updated Mapping Functions ---
def landmarks_to_smpl_positions(landmarks_3d: np.ndarray) -> np.ndarray:
    smpl_joints = np.zeros((24, 3))
    for smpl_idx, mp_idx in MEDIAPIPE_TO_SMPL_MAPPING.items():
        if mp_idx is not None:
            smpl_joints[smpl_idx] = landmarks_3d[mp_idx]
            
    smpl_joints[0] = (smpl_joints[1] + smpl_joints[2]) / 2
    pelvis, shoulders_mid = smpl_joints[0], (smpl_joints[16] + smpl_joints[17]) / 2
    smpl_joints[3] = pelvis + (shoulders_mid - pelvis) * 0.33
    smpl_joints[6] = pelvis + (shoulders_mid - pelvis) * 0.66
    smpl_joints[9] = shoulders_mid
    smpl_joints[12] = (smpl_joints[9] + smpl_joints[15]) / 2
    smpl_joints[13] = smpl_joints[9] + (smpl_joints[16] - smpl_joints[9]) * 0.5
    smpl_joints[14] = smpl_joints[9] + (smpl_joints[17] - smpl_joints[9]) * 0.5
    return smpl_joints

def landmarks_to_mano_positions(landmarks_3d: np.ndarray) -> np.ndarray:
    mano_joints = np.zeros((16, 3))
    joint_counts = np.zeros(16)
    for mp_idx, mano_idx in MEDIAPIPE_HAND_TO_MANO_MAPPING.items():
        pos = landmarks_3d[mp_idx]
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

def get_bag_metadata(bag_path: str) -> int:
    """
    INSTANTLY estimates frame count using bag duration metadata.
    """
    ctx = rs.context()
    device = ctx.load_device(str(bag_path))
    playback = device.as_playback()
    
    # 1. Get Duration
    duration = playback.get_duration()
    total_seconds = duration.total_seconds()
    
    # 2. Get FPS
    pipeline = rs.pipeline()
    config = rs.config()
    # ### FIX #1: Disable repeat playback here for metadata check too
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    
    fps = 30 # Default fallback
    try:
        streams = profile.get_streams()
        for stream in streams:
            if stream.stream_type() == rs.stream.color:
                fps = stream.fps()
                break
    except:
        pass
    finally:
        pipeline.stop()
        
    estimated_frames = int(total_seconds * fps)
    print(f"Bag Duration: {total_seconds:.2f}s | FPS: {fps} | Est. Frames: {estimated_frames}")
    return estimated_frames


class SMPLPoseExtractor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.use_realsense_depth = cfg.get("processing", {}).get("use_realsense_depth", False)
        
        log_level = getattr(logging, cfg.logging.level)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        model_path = Path(cfg.models.path)
        
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path / cfg.models.pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=cfg.detection.pose.min_detection_confidence,
            min_pose_presence_confidence=cfg.detection.pose.min_presence_confidence,
            min_tracking_confidence=cfg.detection.pose.min_tracking_confidence
        )
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path / cfg.models.hand_model)),
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
        if 'smpl_body' in ref_data: self.reference_smpl_pose = np.array(ref_data['smpl_body']['positions'])
        if 'mano_left' in ref_data: self.reference_mano_left = np.array(ref_data['mano_left']['positions'])
        if 'mano_right' in ref_data: self.reference_mano_right = np.array(ref_data['mano_right']['positions'])

    def _get_metric_landmarks(self, landmarks, depth_frame, depth_intrinsics, width, height) -> np.ndarray:
        metric_points = []
        for lm in landmarks:
            x_pixel = min(max(0, int(lm.x * width)), width - 1)
            y_pixel = min(max(0, int(lm.y * height)), height - 1)
            
            dist = depth_frame.get_distance(x_pixel, y_pixel)
            
            if dist <= 0:
                metric_points.append([0.0, 0.0, 0.0])
                continue

            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel], dist)
            metric_points.append(point_3d)
            
        return np.array(metric_points)

    def _get_normalized_landmarks(self, landmarks) -> np.ndarray:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    
    def extract_from_bag(self, bag_path: str, output_path: Optional[str] = None) -> List[str]:
        bag_path = Path(bag_path)
        
        if output_path is None:
            output_dir = bag_path.parent
            base_output_name = f"{bag_path.stem}_smpl"
        else:
            p = Path(output_path)
            output_dir = p.parent
            base_output_name = p.stem

        self.logger.info(f"Processing: {bag_path}")
        self.logger.info(f"Depth Mode: {'RealSense Metric Depth' if self.use_realsense_depth else 'MediaPipe Inferred Depth'}")

        chunk_duration = self.cfg.output.get("chunk_duration_sec", None)
        if chunk_duration:
            self.logger.info(f"Output will be split into {chunk_duration}s chunks.")
        else:
            self.logger.info(f"Output will be saved as a single file (no chunking).")

        # PASS 1: Count
        total_frames = get_bag_metadata(str(bag_path))
        
        # PASS 2: Process
        pipeline = rs.pipeline()
        config = rs.config()
        
        # ### FIX #2: DISABLE LOOPING IN CONFIG
        # This tells pyrealsense2 to NOT repeat the video, raising RuntimeError at EOF.
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
        
        if self.use_realsense_depth:
            align_to = rs.stream.color
            align = rs.align(align_to)
        
        pipeline.start(config)
        device = config.resolve(pipeline).get_device()
        playback = device.as_playback()
        playback.set_real_time(self.cfg.processing.real_time)
        
        # State Tracking
        current_chunk_data = []
        chunk_index = 0
        chunk_start_time = None
        generated_files = []
        stats = {'total_frames': 0, 'pose_detected': 0, 'left_hand_detected': 0, 'right_hand_detected': 0}
        
        try:
            processed_count = 0
            frame_idx = 0
            
            while True:
                # ### FIX #3: Safety Brake
                # If we go 5% past the estimate, force stop to prevent infinite loops
                if frame_idx > (total_frames * 1.05):
                     self.logger.info(f"Force breaking: Exceeded estimated frame count ({frame_idx} > {total_frames})")
                     break

                if self.cfg.processing.max_frames and processed_count >= self.cfg.processing.max_frames:
                    break
                
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    self.logger.info("End of bag reached (RuntimeError).")
                    break
                
                # --- TIME CHECK FOR CHUNKING ---
                ts_ms = frames.get_timestamp()
                ts_sec = ts_ms / 1000.0
                
                if chunk_start_time is None:
                    chunk_start_time = ts_sec
                # ### FIX #4: Check for timestamp reset (Loop detection safety)
                elif ts_sec < chunk_start_time:
                    self.logger.info("Timestamp reset detected (Looping). Breaking.")
                    break
                
                if chunk_duration is not None and (ts_sec - chunk_start_time) >= chunk_duration:
                    if len(current_chunk_data) > 0:
                        saved_file = self._save_chunk(
                            current_chunk_data, output_dir, base_output_name, 
                            chunk_index, str(bag_path)
                        )
                        generated_files.append(saved_file)
                        chunk_index += 1
                        current_chunk_data = []
                        chunk_start_time = ts_sec
                        self.logger.info(f"Starting chunk {chunk_index} at {ts_sec:.2f}s")

                # --- ALIGNMENT PROCESS ---
                color_frame = None
                depth_frame = None
                depth_intrinsics = None

                if self.use_realsense_depth:
                    try:
                        # ### FIX #5: Catch Null Pointer/Ghost Frame error here
                        aligned_frames = align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()
                        
                        if 'depth_intrinsics_cache' not in locals() and depth_frame:
                            depth_intrinsics_cache = depth_frame.profile.as_video_stream_profile().intrinsics
                        if 'depth_intrinsics_cache' in locals():
                            depth_intrinsics = depth_intrinsics_cache
                            
                    except RuntimeError as e:
                        if "null pointer" in str(e) or "frame_ref" in str(e):
                            continue 
                        else:
                            self.logger.warning(f"Alignment error: {e}")
                            continue
                else:
                    color_frame = frames.get_color_frame()

                if not color_frame: continue
                if color_frame.get_width() == 0 or color_frame.get_height() == 0: continue
                
                stats['total_frames'] += 1
                
                # ### FIX #6: Increment frame_idx here at the valid frame point
                frame_idx += 1

                if frame_idx % self.cfg.processing.frame_skip != 0: continue
                processed_count += 1
                
                # MP Detection
                color_image = np.asanyarray(color_frame.get_data())
                height, width = color_image.shape[:2]
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                pose_result = self.pose_landmarker.detect(mp_image)
                hand_result = self.hand_landmarker.detect(mp_image)
                
                # --- PROCESS FRAME ---
                frame_data = self._process_frame(
                    frame_idx, pose_result, hand_result, 
                    depth_frame, depth_intrinsics, width, height, stats
                )
                
                frame_data['timestamp'] = ts_sec
                current_chunk_data.append(frame_data)
                
                # Visualization
                if self.cfg.processing.visualize:
                    self._visualize_frame(color_image, frame_idx, frame_data, stats, total_frames)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                if processed_count % self.cfg.logging.print_progress_every == 0:
                    pct = min(100.0, (frame_idx / total_frames) * 100)
                    self.logger.info(f"Processed {processed_count} frames... (~{pct:.1f}%)")
                    
        finally:
            pipeline.stop()
            if self.cfg.processing.visualize: cv2.destroyAllWindows()
            if len(current_chunk_data) > 0:
                saved_file = self._save_chunk(
                    current_chunk_data, output_dir, base_output_name, 
                    chunk_index, str(bag_path)
                )
                generated_files.append(saved_file)

        self._print_summary(stats)
        return generated_files

    def _save_chunk(self, frames_data, output_dir, base_name, index, source_path) -> str:
        filename = f"{base_name}_part{index:03d}.json"
        out_path = output_dir / filename
        output_data = {
            'frames': frames_data,
            'metadata': {
                'source': source_path,
                'chunk_index': index,
                'frame_count': len(frames_data),
                'config': OmegaConf.to_container(self.cfg, resolve=True)
            }
        }
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2 if self.cfg.output.pretty_print else None)
        self.logger.info(f"Saved chunk {index}: {out_path} ({len(frames_data)} frames)")
        return str(out_path)

    def _process_frame(self, frame_idx, pose_result, hand_result, 
                      depth_frame, depth_intrinsics, width, height, stats) -> Dict:
        # (This method remains unchanged)
        frame_data = {
            'frame_idx': frame_idx,
            'smpl_body': None, 'mano_left': None, 'mano_right': None, 'root_translation': None
        }

        # --- BODY ---
        if pose_result.pose_landmarks:
            stats['pose_detected'] += 1
            landmarks = pose_result.pose_landmarks[0]
            if self.use_realsense_depth and depth_frame:
                lm_3d = self._get_metric_landmarks(landmarks, depth_frame, depth_intrinsics, width, height)
            else:
                lm_3d = self._get_normalized_landmarks(landmarks)
            smpl_positions = landmarks_to_smpl_positions(lm_3d)
            if self.reference_smpl_pose is None and self.cfg.reference.use_first_frame:
                self.reference_smpl_pose = smpl_positions.copy()
            smpl_rotations_6d = positions_to_6d_rotations(smpl_positions, SMPL_PARENT, self.reference_smpl_pose)
            frame_data['smpl_body'] = {'rotations_6d': smpl_rotations_6d.tolist()}
            if self.cfg.output.save_positions:
                frame_data['smpl_body']['positions'] = smpl_positions.tolist()
            frame_data['root_translation'] = smpl_positions[0].tolist()

        # --- HANDS ---
        if hand_result.hand_landmarks:
            for idx, landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[idx][0].category_name
                if self.use_realsense_depth and depth_frame:
                    lm_3d = self._get_metric_landmarks(landmarks, depth_frame, depth_intrinsics, width, height)
                else:
                    lm_3d = self._get_normalized_landmarks(landmarks)
                mano_positions = landmarks_to_mano_positions(lm_3d)
                
                is_left = (handedness == 'Left')
                if is_left:
                    if self.reference_mano_left is None and self.cfg.reference.use_first_frame:
                        self.reference_mano_left = mano_positions.copy()
                    ref = self.reference_mano_left
                    stats['left_hand_detected'] += 1
                else:
                    if self.reference_mano_right is None and self.cfg.reference.use_first_frame:
                        self.reference_mano_right = mano_positions.copy()
                    ref = self.reference_mano_right
                    stats['right_hand_detected'] += 1
                
                rots = positions_to_6d_rotations(mano_positions, MANO_PARENT, ref)
                data = {'rotations_6d': rots.tolist()}
                if self.cfg.output.save_positions: data['positions'] = mano_positions.tolist()
                if is_left: frame_data['mano_left'] = data
                else: frame_data['mano_right'] = data
                
        return frame_data

    def _prepare_output(self, bag_path, frames_data, stats):
        return {
            'frames': frames_data,
            'metadata': {
                'source': str(bag_path),
                'depth_mode': 'RealSense' if self.use_realsense_depth else 'MediaPipe',
                'stats': stats
            }
        }
    
    def _visualize_frame(self, *args):
        pass

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

@hydra.main(version_base=None, config_path="../configs", config_name="pose")
def main(cfg: DictConfig) -> None:
    if cfg.bag_file is None: raise ValueError("bag_file must be specified!")
    extractor = SMPLPoseExtractor(cfg)
    extractor.extract_from_bag(bag_path=cfg.bag_file, output_path=cfg.output_path)

if __name__ == '__main__':
    main()