"""
Debug script: trace the rot6d → rotation matrix → Euler pipeline
to find where the sumo-pose distortion originates.

Run with the same paths as your replay script.
"""

import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation

# ── Paste your paths here ────────────────────────────────────────────────────
MOTION_FILE = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/samples_humanml_trans_dec_512_bert_000600000_seed10_a_person_is_walking_forward_and_picking_something_from_the_ground_with_one_hand/results.pkl")
MEAN_PATH = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Mean.npy")
STD_PATH  = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Std.npy")
MOTION_IDX = 0

JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1/Torso", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3/Chest", "L_Foot/Toe",
    "R_Foot/Toe", "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

def rot6d_to_rotmat(rot6d):
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)  # rows (fixed version)

# ── Load ─────────────────────────────────────────────────────────────────────
with open(MOTION_FILE, 'rb') as f:
    raw = pickle.load(f)

print("=== PKL KEYS ===")
print(list(raw.keys()))

data = raw['motion_full_repr']  # (9, 263, 1, 120)
print(f"\nRaw data shape: {data.shape}")
print(f"Raw data range: [{data.min():.4f}, {data.max():.4f}]")
print(f"Raw data mean:  {data.mean():.4f}")
print(f"Raw data std:   {data.std():.4f}")

mean = np.load(MEAN_PATH)
std  = np.load(STD_PATH)
print(f"\nMean shape: {mean.shape}, range: [{mean.min():.4f}, {mean.max():.4f}]")
print(f"Std  shape: {std.shape},  range: [{std.min():.4f}, {std.max():.4f}]")
print(f"Std  zeros: {(std == 0).sum()} / {len(std)}")

motion_raw = data[MOTION_IDX, :, 0, :].T  # (T, 263)
print(f"\nMotion (before unnorm) shape: {motion_raw.shape}")
print(f"Motion (before unnorm) range: [{motion_raw.min():.4f}, {motion_raw.max():.4f}]")

motion = motion_raw * std + mean
print(f"Motion (after unnorm) range:  [{motion.min():.4f}, {motion.max():.4f}]")

T = len(motion)
frame = 0  # inspect frame 0

# ── 1. Check root info ──────────────────────────────────────────────────────
print("\n=== FRAME 0: ROOT INFO ===")
print(f"  Root angular vel (idx 0):   {motion[frame, 0]:.6f}")
print(f"  Root linear vel  (idx 1:3): {motion[frame, 1:3]}")
print(f"  Root height      (idx 3):   {motion[frame, 3]:.4f}")

# ── 2. Check local joint positions (sanity) ──────────────────────────────────
print("\n=== FRAME 0: LOCAL JOINT POSITIONS (idx 4:67) ===")
jpos = motion[frame, 4:67].reshape(21, 3)
for i, name in enumerate(JOINT_NAMES):
    print(f"  {name:15s}: {jpos[i]}")

# ── 3. Check rot6d raw values ────────────────────────────────────────────────
print("\n=== FRAME 0: ROT6D RAW VALUES (idx 67:193) ===")
rot6d_raw = motion[frame, 67:193].reshape(21, 6)
for i, name in enumerate(JOINT_NAMES):
    r = rot6d_raw[i]
    print(f"  {name:15s}: [{r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} | {r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f}]")

# ── 4. Convert to rotation matrices, check determinant and identity-closeness
print("\n=== FRAME 0: ROTATION MATRICES ===")
rotmats = rot6d_to_rotmat(rot6d_raw)  # (21, 3, 3)
for i, name in enumerate(JOINT_NAMES):
    R = rotmats[i]
    det = np.linalg.det(R)
    # How far from identity?
    angle = Rotation.from_matrix(R).magnitude()  # total rotation angle in rad
    euler_ZYX = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
    euler_zyx = Rotation.from_matrix(R).as_euler('zyx', degrees=True)
    print(f"  {name:15s}: det={det:.4f}  angle={np.degrees(angle):7.2f}°  "
          f"ZYX(intrinsic)=[{euler_ZYX[0]:7.2f} {euler_ZYX[1]:7.2f} {euler_ZYX[2]:7.2f}]  "
          f"zyx(extrinsic)=[{euler_zyx[0]:7.2f} {euler_zyx[1]:7.2f} {euler_zyx[2]:7.2f}]")

# ── 5. Check if rot6d columns are close to identity ─────────────────────────
print("\n=== FRAME 0: ROT6D IDENTITY CHECK ===")
print("  (If the person is roughly in T-pose at frame 0, most should be near identity)")
print("  (rot6d of identity = [1,0,0, 0,1,0])")
for i, name in enumerate(JOINT_NAMES):
    r = rot6d_raw[i]
    identity_6d = np.array([1, 0, 0, 0, 1, 0])
    dist = np.linalg.norm(r - identity_6d)
    print(f"  {name:15s}: dist_from_identity = {dist:.4f}")

# ── 6. Check multiple frames to see if motion varies ────────────────────────
print("\n=== ROT6D VARIANCE ACROSS FRAMES (should be nonzero for moving joints) ===")
all_rot6d = motion[:, 67:193].reshape(T, 21, 6)
for i, name in enumerate(JOINT_NAMES):
    var = all_rot6d[:, i, :].var(axis=0).mean()
    print(f"  {name:15s}: mean_var = {var:.6f}")

# ── 7. Quick check: what does the POSITION channel say about frame 0? ───────
print("\n=== POSITION-BASED SANITY CHECK ===")
print("  (Joint positions relative to root — do legs point down, arms out?)")
print(f"  L_Hip  pos: {jpos[0]}   (expect small offset left, downward)")
print(f"  L_Knee pos: {jpos[3]}   (expect further down)")
print(f"  L_Ankle pos: {jpos[6]}  (expect even further down)")
print(f"  L_Shoulder pos: {jpos[15]} (expect left, up)")
print(f"  L_Elbow pos: {jpos[17]}    (expect further left)")
print(f"  Head pos: {jpos[14]}       (expect up)")

# ── 8. Check if maybe the data is ALREADY unnormalized ──────────────────────
print("\n=== DOUBLE-UNNORMALIZATION CHECK ===")
print("  If rot6d columns 1&4 of identity should be ~1.0 and rest ~0.0,")
print("  check what the raw (pre-unnorm) rot6d looks like:")
rot6d_prenorm = motion_raw[frame, 67:193].reshape(21, 6)
print(f"  L_Hip rot6d (pre-unnorm):  {rot6d_prenorm[0]}")
print(f"  L_Hip rot6d (post-unnorm): {rot6d_raw[0]}")
print(f"  Mean for rot6d[0:6]:       {mean[67:73]}")
print(f"  Std  for rot6d[0:6]:       {std[67:73]}")