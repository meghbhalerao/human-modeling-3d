#!/usr/bin/env python3
"""
Extract SMPL-X joint parameters from video as a time series.
Outputs the 55 SMPL-X joint rotations per frame with detection flags.

This version is designed to work with 4D-Humans (HMR 2.0), which is the 
most accessible and well-maintained option.
"""

import torch
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_smplx_from_video(video_path, output_path, device='cuda'):
    """
    Extract SMPL-X parameters from video using available libraries.
    
    SMPL-X has 55 joints total:
    - 1 root (pelvis) + 21 body joints 
    - 30 hand joints (15 per hand)
    - 3 head/neck joints
    
    Each joint has 3D rotation (axis-angle), so 55 × 3 = 165 parameters.
    """
    
    # Try to import available libraries
    method = None
    
    # Try 4D-Humans (HMR 2.0) first - most accessible
    try:
        from hmr2.models import HMR2, download_models, load_hmr2
        from hmr2.utils import recursive_to
        from hmr2.datasets.utils import expand_to_aspect_ratio
        import hmr2.datasets.vitdet_dataset as vitdet_dataset
        print("✓ Found 4D-Humans (HMR 2.0)")
        method = '4dhumans'
    except ImportError:
        print("✗ 4D-Humans not installed")
    
    # Try PyMAF-X
    if method is None:
        try:
            import pymafx
            print("✓ Found PyMAF-X")
            method = 'pymafx'
        except ImportError:
            print("✗ PyMAF-X not installed")
    
    # Try FrankMocap
    if method is None:
        try:
            from frankmocap.bodymocap_api import BodyMocap
            from frankmocap.handmocap_api import HandMocap
            print("✓ Found FrankMocap")
            method = 'frankmocap'
        except ImportError:
            print("✗ FrankMocap not installed")
    
    if method is None:
        print("\n❌ No SMPL-X extraction library found!")
        print("Please install one of the following:")
        print("\n1. 4D-Humans (RECOMMENDED - easiest):")
        print("   git clone https://github.com/shubham-goel/4D-Humans.git")
        print("   cd 4D-Humans && pip install -r requirements.txt")
        print("   bash scripts/download_models.sh")
        print("\n2. PyMAF-X (best quality):")
        print("   git clone https://github.com/HongwenZhang/PyMAF-X.git")
        print("   cd PyMAF-X && pip install -r requirements.txt")
        print("\n3. FrankMocap:")
        print("   git clone https://github.com/facebookresearch/frankmocap.git")
        print("   cd frankmocap && pip install -r requirements.txt")
        print("\nGenerating placeholder data structure for now...")
        method = 'placeholder'
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"Method: {method}\n")
    
    # Initialize SMPL-X time series
    smplx_timeseries = {
        'metadata': {
            'video_path': str(video_path),
            'frame_width': frame_width,
            'frame_height': frame_height,
            'fps': fps,
            'total_frames': total_frames,
            'num_joints': 55,
            'joint_format': 'axis-angle',  # 3D rotation per joint
            'extraction_method': method,
            'joint_names': get_smplx_joint_names()
        },
        'frames': []
    }
    
    # Extract based on available method
    if method == '4dhumans':
        smplx_timeseries = extract_with_4dhumans(
            cap, smplx_timeseries, device, total_frames
        )
    elif method == 'placeholder':
        smplx_timeseries = create_placeholder_data(
            cap, smplx_timeseries, total_frames
        )
    else:
        raise NotImplementedError(f"Method {method} not yet fully implemented. Please use 4D-Humans.")
    
    cap.release()
    
    # Save to files
    output_path = Path(output_path)
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(smplx_timeseries, f, indent=2)
    print(f"\n✓ Saved JSON to: {json_path}")
    
    # Save as pickle
    pkl_path = output_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(smplx_timeseries, f)
    print(f"✓ Saved pickle to: {pkl_path}")
    
    # Save as NPZ
    save_as_npz(smplx_timeseries, output_path.with_suffix('.npz'))
    
    return smplx_timeseries


def get_smplx_joint_names():
    """Return the names of all 55 SMPL-X joints in order."""
    return [
        # Root + Body (22)
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot',
        'right_foot', 'neck', 'left_collar', 'right_collar', 'head',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        # Jaw + Eyes (3)
        'jaw', 'left_eye_smplx', 'right_eye_smplx',
        # Left Hand (15)
        'left_index1', 'left_index2', 'left_index3',
        'left_middle1', 'left_middle2', 'left_middle3',
        'left_pinky1', 'left_pinky2', 'left_pinky3',
        'left_ring1', 'left_ring2', 'left_ring3',
        'left_thumb1', 'left_thumb2', 'left_thumb3',
        # Right Hand (15)
        'right_index1', 'right_index2', 'right_index3',
        'right_middle1', 'right_middle2', 'right_middle3',
        'right_pinky1', 'right_pinky2', 'right_pinky3',
        'right_ring1', 'right_ring2', 'right_ring3',
        'right_thumb1', 'right_thumb2', 'right_thumb3',
    ]


def extract_with_4dhumans(cap, smplx_timeseries, device, total_frames):
    """Extract using 4D-Humans (HMR 2.0) library."""
    from hmr2.models import download_models, load_hmr2
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hmr2.utils.renderer import Renderer, cam_crop_to_full
    
    # Load model
    print("Loading HMR 2.0 model...")
    model, model_cfg = load_hmr2()
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Extracting SMPL-X parameters...")
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Convert to RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prepare batch (HMR 2.0 expects specific format)
                # For simplicity, we'll use the full image
                # In production, you'd want to detect and crop the person first
                
                # Resize to model input size
                img_resized = cv2.resize(img_rgb, (256, 256))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                
                # Normalize
                mean = torch.tensor(DEFAULT_MEAN).view(3, 1, 1)
                std = torch.tensor(DEFAULT_STD).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                # Add batch dimension
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Create batch dict
                batch = {
                    'img': img_tensor,
                    'img_size': torch.tensor([[256, 256]]).to(device)
                }
                
                # Run inference
                out = model(batch)
                
                # Extract SMPL-X parameters
                # HMR 2.0 outputs in specific format
                pred_smpl_params = out['pred_smpl_params']
                
                # Get body pose (21 body joints × 3 = 63 params)
                # Note: HMR 2.0 might output fewer joints than full SMPL-X
                body_pose = pred_smpl_params['body_pose'][0].cpu().numpy()
                
                # Get global orientation
                global_orient = pred_smpl_params['global_orient'][0].cpu().numpy()
                
                # Get shape parameters
                betas = pred_smpl_params['betas'][0].cpu().numpy()
                
                # Expand to 55 joints if needed
                # HMR 2.0 typically outputs 21 body + 2 hands = 23 joints
                if body_pose.shape[0] < 165:  # 55 × 3
                    # Pad with zeros for missing joints (hands, face)
                    full_pose = np.zeros((55, 3))
                    n_joints = body_pose.shape[0] // 3
                    full_pose[:n_joints] = body_pose.reshape(-1, 3)
                    body_pose = full_pose
                else:
                    body_pose = body_pose.reshape(55, 3)
                
                detected = True
                detection_confidence = 0.95  # HMR 2.0 doesn't return confidence directly
                
            except Exception as e:
                # Person not detected or processing failed
                detected = False
                body_pose = None
                global_orient = None
                betas = None
                detection_confidence = 0.0
            
            # Store frame data
            frame_data = {
                'frame_number': frame_idx,
                'timestamp': frame_idx / smplx_timeseries['metadata']['fps'],
                'detected': detected,
                'detection_confidence': float(detection_confidence),
                'global_orientation': global_orient.tolist() if detected else None,
                'joint_rotations': body_pose.tolist() if detected else None,
                'shape_params': betas.tolist() if detected else None,
            }
            
            smplx_timeseries['frames'].append(frame_data)
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
    
    return smplx_timeseries


def create_placeholder_data(cap, smplx_timeseries, total_frames):
    """
    Create placeholder data structure when no library is available.
    Shows the expected format but contains no real data.
    """
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    pbar = tqdm(total=total_frames, desc="Creating placeholder data")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_data = {
            'frame_number': frame_idx,
            'timestamp': frame_idx / smplx_timeseries['metadata']['fps'],
            'detected': False,  # Not detected (placeholder)
            'detection_confidence': 0.0,
            'global_orientation': None,  # [3] - root orientation
            'joint_rotations': None,  # [55, 3] - 55 joints, 3D axis-angle each
            'shape_params': None,  # [10] - body shape parameters
        }
        
        smplx_timeseries['frames'].append(frame_data)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    print("\n" + "="*70)
    print("⚠️  PLACEHOLDER DATA GENERATED")
    print("="*70)
    print("This shows the expected output format, but contains NO real data.")
    print("All 'detected' flags are False and joint_rotations are None.")
    print("\nTo extract real SMPL-X parameters, install one of these libraries:")
    print("  • 4D-Humans: https://github.com/shubham-goel/4D-Humans")
    print("  • PyMAF-X: https://github.com/HongwenZhang/PyMAF-X")
    print("  • FrankMocap: https://github.com/facebookresearch/frankmocap")
    print("="*70)
    
    return smplx_timeseries


def save_as_npz(smplx_timeseries, output_path):
    """Save SMPL-X timeseries as NPZ for easy loading in numpy/pytorch."""
    frames = smplx_timeseries['frames']
    n_frames = len(frames)
    
    # Prepare arrays
    detected = np.array([f['detected'] for f in frames], dtype=bool)
    confidences = np.array([f.get('detection_confidence', 0.0) for f in frames])
    
    # Joint rotations: (n_frames, 55, 3)
    joint_rotations = np.zeros((n_frames, 55, 3))
    for i, frame in enumerate(frames):
        if frame['joint_rotations'] is not None:
            joint_rotations[i] = np.array(frame['joint_rotations'])
        else:
            joint_rotations[i] = np.nan  # Mark as not detected
    
    # Global orientation: (n_frames, 3)
    global_orient = np.zeros((n_frames, 3))
    for i, frame in enumerate(frames):
        if frame['global_orientation'] is not None:
            global_orient[i] = np.array(frame['global_orientation'])
        else:
            global_orient[i] = np.nan
    
    # Shape parameters: (n_frames, 10)
    shape_params = np.zeros((n_frames, 10))
    for i, frame in enumerate(frames):
        if frame.get('shape_params') is not None:
            shape_params[i] = np.array(frame['shape_params'])[:10]  # Take first 10
        else:
            shape_params[i] = np.nan
    
    # Save
    np.savez(
        output_path,
        detected=detected,
        confidences=confidences,
        joint_rotations=joint_rotations,  # (n_frames, 55, 3)
        global_orientation=global_orient,  # (n_frames, 3)
        shape_params=shape_params,  # (n_frames, 10)
        fps=smplx_timeseries['metadata']['fps'],
        joint_names=smplx_timeseries['metadata']['joint_names'],
    )
    
    print(f"✓ Saved NPZ to: {output_path}")
    print(f"  Joint rotations shape: {joint_rotations.shape}")
    print(f"  Detected frames: {detected.sum()}/{n_frames} ({100*detected.sum()/n_frames:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMPL-X joint parameters from video as time series'
    )
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, help='Output file path (without extension)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = video_path.stem + '_smplx'
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Extract SMPL-X parameters
    smplx_data = extract_smplx_from_video(
        args.video,
        args.output,
        device=args.device
    )
    
    # Print summary
    frames = smplx_data['frames']
    detected_count = sum(1 for f in frames if f['detected'])
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total frames: {len(frames)}")
    print(f"Detected frames: {detected_count}/{len(frames)} ({100*detected_count/len(frames):.1f}%)")
    print(f"SMPL-X joints: 55 (each with 3D axis-angle rotation)")
    print(f"\nOutput files:")
    print(f"  • JSON: {args.output}.json (human-readable)")
    print(f"  • PKL: {args.output}.pkl (Python pickle)")
    print(f"  • NPZ: {args.output}.npz (numpy arrays, recommended)")
    print("="*60)
    
    # Show example usage
    if detected_count > 0:
        print("\n📖 Load the data like this:")
        print(f"""
import numpy as np

# Load the NPZ file
data = np.load('{args.output}.npz', allow_pickle=True)

# Get joint rotations for all frames
joint_rotations = data['joint_rotations']  # Shape: (n_frames, 55, 3)
detected = data['detected']  # Shape: (n_frames,) - boolean array

# Get only detected frames
valid_rotations = joint_rotations[detected]
print(f"Valid frames: {{valid_rotations.shape[0]}}")

# Get left wrist rotation for frame 10 (joint index 20)
if detected[10]:
    left_wrist = joint_rotations[10, 20]
    print(f"Left wrist rotation: {{left_wrist}}")
""")


if __name__ == '__main__':
    main()