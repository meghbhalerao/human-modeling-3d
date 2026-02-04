# Extract SMPL-X Time Series from Video

Extract the 55 SMPL-X joint rotations from video as a time series, with detection flags for occluded/missing data.

## What You Get

For each frame, you'll get:
- **55 joint rotations** (axis-angle format, 3D per joint = 165 values total)
- **Detection flag** (boolean: was a person detected?)
- **Detection confidence** (0-1 score)
- **Global orientation** (3D root rotation)
- **Shape parameters** (10 body shape coefficients)

If a person isn't detected in a frame, all values are set to `null` or `NaN`.

## Installation Options

You need to install ONE of these libraries. Listed from easiest to most advanced:

### Option 1: 4DHumans (HMR 2.0) - RECOMMENDED for ease
**Pros**: Easy to install, works well, actively maintained
**Cons**: Slightly less accurate hands than PyMAF-X

```bash
# Install 4DHumans
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans

# Create environment
conda create -n 4dhumans python=3.10
conda activate 4dhumans

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download models
bash scripts/download_models.sh

# Install in editable mode
pip install -e .

# Test installation
python demo.py --img_folder example_images
```

### Option 2: PyMAF-X - BEST quality
**Pros**: State-of-the-art accuracy, best hand reconstruction
**Cons**: More complex installation

```bash
# Clone repository
git clone https://github.com/HongwenZhang/PyMAF-X.git
cd PyMAF-X

# Create environment
conda create -n pymafx python=3.8
conda activate pymafx

# Install PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install -r requirements.txt

# Download pretrained models
bash fetch_data.sh

# Test installation
python demo.py --img_folder demo/images --out_folder demo/results
```

### Option 3: FrankMocap - For hand focus
**Pros**: Excellent hand tracking, body + hands combined
**Cons**: Older, less maintained

```bash
git clone https://github.com/facebookresearch/frankmocap.git
cd frankmocap

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision

# Download models
bash scripts/download_models.sh

# Test
python -m demo.demo_frankmocap --input_path demo/sample.mp4 --out_dir output
```

## Usage

Once you've installed ONE of the above libraries:

```bash
# Basic usage (auto-detects which library you have)
python extract_smplx_timeseries.py video.mp4

# Specify output name
python extract_smplx_timeseries.py video.mp4 -o my_output

# Use CPU instead of GPU
python extract_smplx_timeseries.py video.mp4 --device cpu
```

## Output Files

The script generates 3 files:

### 1. JSON file (human-readable)
```json
{
  "metadata": {
    "num_joints": 55,
    "joint_names": ["pelvis", "left_hip", "right_hip", ...],
    "fps": 30.0
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "detected": true,
      "detection_confidence": 0.95,
      "global_orientation": [0.1, -0.05, 0.02],
      "joint_rotations": [
        [0.0, 0.0, 0.0],      // pelvis
        [0.1, 0.2, 0.3],      // left_hip
        [-0.1, 0.2, -0.3],    // right_hip
        // ... 52 more joints
      ],
      "shape_params": [0.1, -0.2, 0.05, ...]
    },
    {
      "frame_number": 1,
      "detected": false,      // Person not detected!
      "joint_rotations": null  // All data is null
    }
  ]
}
```

### 2. PKL file (Python pickle - compact)
```python
import pickle

with open('video_smplx.pkl', 'rb') as f:
    data = pickle.load(f)

# Access frames
for frame in data['frames']:
    if frame['detected']:
        print(f"Frame {frame['frame_number']}: person detected")
```

### 3. NPZ file (NumPy arrays - best for ML)
```python
import numpy as np

# Load data
data = np.load('video_smplx.npz', allow_pickle=True)

# Extract arrays
joint_rotations = data['joint_rotations']  # Shape: (n_frames, 55, 3)
detected = data['detected']                 # Shape: (n_frames,) boolean
confidences = data['confidences']           # Shape: (n_frames,)
global_orient = data['global_orientation']  # Shape: (n_frames, 3)

print(f"Total frames: {len(joint_rotations)}")
print(f"Detected frames: {detected.sum()}")

# Get only frames where person was detected
valid_frames = joint_rotations[detected]
print(f"Valid data shape: {valid_frames.shape}")

# Access specific joint rotation
# Joint 20 = left wrist
left_wrist_rotations = joint_rotations[:, 20, :]  # Shape: (n_frames, 3)

# Check for missing data (NaN values)
has_nan = np.isnan(joint_rotations)
print(f"Frames with missing data: {np.any(has_nan, axis=(1,2)).sum()}")
```

## SMPL-X Joint Order (All 55 Joints)

```
Index  | Joint Name        | Description
-------|-------------------|---------------------------
0      | pelvis            | Root joint
1-2    | left/right_hip    | Hip joints
3      | spine1            | Lower spine
4-5    | left/right_knee   | Knees
6      | spine2            | Mid spine
7-8    | left/right_ankle  | Ankles
9      | spine3            | Upper spine
10-11  | left/right_foot   | Feet
12     | neck              | Neck
13-14  | left/right_collar | Collarbones
15     | head              | Head
16-17  | left/right_shoulder | Shoulders
18-19  | left/right_elbow  | Elbows
20-21  | left/right_wrist  | Wrists
22     | jaw               | Jaw
23-24  | left_eye_smplx/left_eye | Left eye
25-26  | right_eye_smplx/right_eye | Right eye

Left Hand (27-41): 15 joints
  27-29:  left_index (1,2,3)
  30-32:  left_middle (1,2,3)
  33-35:  left_pinky (1,2,3)
  36-38:  left_ring (1,2,3)
  39-41:  left_thumb (1,2,3)

Right Hand (42-54): 15 joints
  42-44:  right_index (1,2,3)
  45-47:  right_middle (1,2,3)
  48-50:  right_pinky (1,2,3)
  51-53:  right_ring (1,2,3)
  54-56:  right_thumb (1,2,3)
```

## Working with the Data

### Example 1: Load and visualize detection rate
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('video_smplx.npz', allow_pickle=True)
detected = data['detected']
confidences = data['confidences']

# Plot detection over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(detected.astype(int))
plt.title('Detection Per Frame')
plt.xlabel('Frame')
plt.ylabel('Detected (1=yes, 0=no)')

plt.subplot(1, 2, 2)
plt.plot(confidences)
plt.title('Detection Confidence')
plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.show()

print(f"Detection rate: {detected.mean()*100:.1f}%")
```

### Example 2: Extract hand motion
```python
import numpy as np

data = np.load('video_smplx.npz', allow_pickle=True)
joint_rotations = data['joint_rotations']
detected = data['detected']

# Left hand joints: indices 27-41
# Right hand joints: indices 42-54

left_hand = joint_rotations[:, 27:42, :]   # Shape: (n_frames, 15, 3)
right_hand = joint_rotations[:, 42:55, :]  # Shape: (n_frames, 15, 3)

# Get only valid (detected) frames
left_hand_valid = left_hand[detected]
right_hand_valid = right_hand[detected]

print(f"Left hand motion shape: {left_hand_valid.shape}")
print(f"Right hand motion shape: {right_hand_valid.shape}")

# Calculate hand motion magnitude
left_hand_motion = np.linalg.norm(
    np.diff(left_hand_valid, axis=0),  # Frame-to-frame difference
    axis=2  # Magnitude of 3D rotation
).mean(axis=1)  # Average over all hand joints

print(f"Average left hand motion per frame: {left_hand_motion.mean():.4f}")
```

### Example 3: Smooth noisy data
```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

data = np.load('video_smplx.npz', allow_pickle=True)
joint_rotations = data['joint_rotations']
detected = data['detected']

# Smooth rotations for detected frames only
smoothed = joint_rotations.copy()

for joint_idx in range(55):
    for dim in range(3):
        # Get values for this joint dimension
        values = joint_rotations[:, joint_idx, dim]
        
        # Only smooth detected frames
        valid_mask = detected & ~np.isnan(values)
        
        if valid_mask.sum() > 3:  # Need enough points to smooth
            # Apply Gaussian smoothing
            smoothed[valid_mask, joint_idx, dim] = gaussian_filter1d(
                values[valid_mask], sigma=2.0
            )

# Save smoothed version
np.savez('video_smplx_smoothed.npz',
         joint_rotations=smoothed,
         detected=detected,
         **{k: v for k, v in data.items() if k not in ['joint_rotations', 'detected']})
```

### Example 4: Convert to SMPL-X mesh
```python
import torch
import smplx
import numpy as np

# Load SMPL-X model
smplx_model = smplx.create(
    'smplx_models',  # Path to SMPL-X model files
    model_type='smplx',
    gender='neutral',
    num_betas=10,
)

# Load extracted parameters
data = np.load('video_smplx.npz', allow_pickle=True)
joint_rotations = data['joint_rotations']
global_orient = data['global_orientation']
shape_params = data['shape_params']
detected = data['detected']

# Generate mesh for frame 100
frame_idx = 100
if detected[frame_idx]:
    output = smplx_model(
        body_pose=torch.tensor(joint_rotations[frame_idx]).unsqueeze(0),
        global_orient=torch.tensor(global_orient[frame_idx]).unsqueeze(0),
        betas=torch.tensor(shape_params[frame_idx]).unsqueeze(0),
    )
    
    vertices = output.vertices[0].detach().numpy()  # Shape: (10475, 3)
    faces = smplx_model.faces  # Triangle indices
    
    print(f"Generated mesh with {len(vertices)} vertices")
    
    # Save as OBJ or visualize with trimesh/open3d
```

## Troubleshooting

### "No module named 'hmr2' / 'mafu' / 'frankmocap'"
You haven't installed any of the required libraries. Pick one from the Installation Options above.

### "CUDA out of memory"
Use CPU mode: `python extract_smplx_timeseries.py video.mp4 --device cpu`

### Low detection rate
- Make sure the person is clearly visible
- Try a different library (PyMAF-X is most robust)
- Check if video quality is good enough

### All frames show "detected": false
This means no library was properly installed. The script is running in placeholder mode. Install one of the libraries above.

## Citation

If you use this for research, please cite the appropriate papers:

**4DHumans:**
```
@inproceedings{goel20234dhumans,
  title={Humans in 4D: Reconstructing and Tracking Humans with Transformers},
  author={Goel, Shubham and Pavlakos, Georgios and Rajasegaran, Jathushan and Kanazawa, Angjoo and Malik, Jitendra},
  booktitle={ICCV},
  year={2023}
}
```

**PyMAF-X:**
```
@inproceedings{pymafx2023,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```

**SMPL-X:**
```
@inproceedings{SMPL-X:2019,
  title={Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author={Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle={CVPR},
  year={2019}
}
```
