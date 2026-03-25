"""
replay_rot6d.py

Kinematic replay of HumanML3D 263-dim motion data using rot6d representation.

The 263-dim vector structure:
  [0]      root angular velocity (Y axis)
  [1:3]    root linear velocity (XZ plane)
  [3]      root height (Y)
  [4:67]   local joint positions (21 joints × 3)  ← not used here
  [67:193] joint rot6d (21 joints × 6)             ← THIS IS WHAT WE USE
  [193:259] joint velocities (22 joints × 3)       ← not used here
  [259:263] foot contacts (4)                      ← not used here

Pipeline:
  263-dim → extract rot6d [67:193] → rot6d_to_rotmat → rotmat_to_euler
  → set sim.data.qpos → sim.forward() → MuJoCo runs FK internally
"""

import mujoco_py
import numpy as np
import pickle
import time
import os
import sys
from scipy.spatial.transform import Rotation

# ── CONFIG ───────────────────────────────────────────────────────────────────
SCENE_XML   = os.path.expanduser("~/projects/human-modeling-3d/assets/human/scene.xml")
MOTION_FILE = os.path.expanduser("/home/mb230/projects/human-modeling-3d/data/modiff-2022-samp-from-text/forward-walking-only/after_inv_transform/results.pkl")  # ← change this
MEAN_PATH = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Mean.npy")
STD_PATH = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Std.npy")
MOTION_IDX  = 0      # which motion sequence to replay
FPS         = 30
LOOP        = True
# ─────────────────────────────────────────────────────────────────────────────

# ── Joint mapping: HumanML3D order (joints 1-21) → MuJoCo qpos start index ──
# Each joint occupies 3 consecutive qpos entries in order: z, y, x
# HumanML3D skips the root (joint 0), so index 0 here = HumanML3D joint 1
HML_JOINT_TO_QPOS = [
    0,   # 0: left_hip       → L_Hip   (qpos 0,1,2)
    12,  # 1: right_hip      → R_Hip   (qpos 12,13,14)
    24,  # 2: spine1         → Torso   (qpos 24,25,26)
    3,   # 3: left_knee      → L_Knee  (qpos 3,4,5)
    15,  # 4: right_knee     → R_Knee  (qpos 15,16,17)
    27,  # 5: spine2         → Spine   (qpos 27,28,29)
    6,   # 6: left_ankle     → L_Ankle (qpos 6,7,8)
    18,  # 7: right_ankle    → R_Ankle (qpos 18,19,20)
    30,  # 8: spine3         → Chest   (qpos 30,31,32)
    9,   # 9: left_foot      → L_Toe   (qpos 9,10,11)
    21,  # 10: right_foot    → R_Toe   (qpos 21,22,23)
    33,  # 11: neck          → Neck    (qpos 33,34,35)
    39,  # 12: left_collar   → L_Thorax(qpos 39,40,41)
    54,  # 13: right_collar  → R_Thorax(qpos 54,55,56)
    36,  # 14: head          → Head    (qpos 36,37,38)
    42,  # 15: left_shoulder → L_Shoulder(qpos 42,43,44)
    57,  # 16: right_shoulder→ R_Shoulder(qpos 57,58,59)
    45,  # 17: left_elbow    → L_Elbow (qpos 45,46,47)
    60,  # 18: right_elbow   → R_Elbow (qpos 60,61,62)
    48,  # 19: left_wrist    → L_Wrist (qpos 48,49,50)
    63,  # 20: right_wrist   → R_Wrist (qpos 63,64,65)
    # L_Hand (51,52,53) and R_Hand (66,67,68) not in HumanML3D → stay 0
]


def rot6d_to_rotmat(rot6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation stores the first two columns of a rotation matrix.

    The third column is recovered via cross product (Gram-Schmidt).

    rot6d: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = rot6d[..., 0:3]   # first column of R
    a2 = rot6d[..., 3:6]   # second column of R

    # Gram-Schmidt orthogonalization
    # b1 = normalize(a1)
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)

    # b2 = normalize(a2 - (b2·b1)b1)
    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    # b3 = b1 × b2
    b3 = np.cross(b1, b2)

    # Stack columns → rotation matrix
    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)


def recover_root_rot_pos(data):
    """
    Recover root rotation (quaternion) and global position from 263-dim data.
    This is the numpy equivalent of recover_root_rot_pos() in HumanML3D.

    data:    (T, 263)
    returns: r_rot_quat (T, 4) in wxyz convention
             r_pos      (T, 3) global root position
    """
    T = len(data)

    # ── Root rotation ────────────────────────────────────────────────────────
    # Index 0: angular velocity around Y axis
    # Cumulative sum gives absolute Y rotation angle
    rot_vel = data[:, 0]
    r_rot_ang = np.zeros(T)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)

    # Build quaternion for rotation around Y axis only
    # q = [cos(a/2), 0, sin(a/2), 0] in wxyz
    r_rot_quat = np.zeros((T, 4))
    r_rot_quat[:, 0] = np.cos(r_rot_ang)   # w
    r_rot_quat[:, 2] = np.sin(r_rot_ang)   # y

    # ── Root position ────────────────────────────────────────────────────────
    # Indices 1:3 = linear velocity on XZ plane
    # Index 3    = root height (Y)
    r_pos = np.zeros((T, 3))
    r_pos[1:, [0, 2]] = data[:-1, 1:3]    # XZ velocity → cumsum later

    # Rotate velocity by inverse root rotation to get world-space displacement
    # scipy uses xyzw quaternion convention, HumanML3D uses wxyz
    # Convert: wxyz → xyzw for scipy
    r_rot_scipy = r_rot_quat[:, [1, 2, 3, 0]]  # xyzw
    rot_obj = Rotation.from_quat(r_rot_scipy)
    r_pos = rot_obj.inv().apply(r_pos)

    # Cumulative sum to get absolute position from velocity
    r_pos = np.cumsum(r_pos, axis=0)

    # Override Y with direct root height
    r_pos[:, 1] = data[:, 3]

    return r_rot_quat, r_pos


def extract_euler_angles(data_263):
    """
    Extract joint euler angles from 263-dim HumanML3D data.

    data_263: (T, 263)
    returns:
        euler_angles: (T, 21, 3) in degrees, ordered XYZ
        r_rot_quat:   (T, 4)     root rotation in wxyz
        r_pos:        (T, 3)     root global position
    """
    T = len(data_263)

    # ── Extract rot6d for joints 1-21 ────────────────────────────────────────
    # Indices 67:193 = 21 joints × 6 = 126 values
    rot6d = data_263[:, 67:193]         # (T, 126)
    rot6d = rot6d.reshape(T, 21, 6)     # (T, 21, 6)

    # ── 6D → rotation matrix ─────────────────────────────────────────────────
    rotmat = rot6d_to_rotmat(rot6d)     # (T, 21, 3, 3)

    # ── Rotation matrix → euler angles ───────────────────────────────────────
    # Flatten for scipy: (T*21, 3, 3)
    rotmat_flat  = rotmat.reshape(T * 21, 3, 3)
    rot_obj      = Rotation.from_matrix(rotmat_flat)

    # Use XYZ euler convention — matches MuJoCo's x,y,z hinge order
    # Note: MuJoCo qpos stores z,y,x so we reorder when setting qpos
    euler_flat   = rot_obj.as_euler('XYZ', degrees=True)  # (T*21, 3) in degrees
    euler_angles = euler_flat.reshape(T, 21, 3)            # (T, 21, 3)

    # ── Root rotation and position ───────────────────────────────────────────
    r_rot_quat, r_pos = recover_root_rot_pos(data_263)

    return euler_angles, r_rot_quat, r_pos


def set_qpos(sim, euler_angles_frame, r_rot_quat_frame, r_pos_frame):
    """
    Set sim.data.qpos for one frame.

    euler_angles_frame: (21, 3) euler angles in degrees, XYZ order
    r_rot_quat_frame:   (4,)   root quaternion wxyz
    r_pos_frame:        (3,)   root position xyz
    """
    # qpos has 69 entries — all hinge joints, no free joint for root
    # Root is handled separately via mocap_pos and mocap_quat
    qpos = np.zeros(sim.model.nq)

    for hml_idx, qpos_start in enumerate(HML_JOINT_TO_QPOS):
        # euler_angles_frame[hml_idx] = [ex, ey, ez] in degrees, XYZ order
        ex, ey, ez = euler_angles_frame[hml_idx]

        # MuJoCo qpos stores z, y, x for each joint
        qpos[qpos_start + 0] = ez   # z hinge
        qpos[qpos_start + 1] = ey   # y hinge
        qpos[qpos_start + 2] = ex   # x hinge

    sim.data.qpos[:] = qpos


def set_root(sim, r_rot_quat, r_pos):
    """
    Set root body position and orientation via mocap.

    r_rot_quat: (4,) wxyz
    r_pos:      (3,) xyz in Y-up space → convert to Z-up for MuJoCo
    """
    # Find mocap id for root body 'object'
    body_id  = sim.model.body_name2id('object')
    mocap_id = sim.model.body_mocapid[body_id]

    # Convert Y-up position to Z-up
    # Y-up: x=right, y=up,      z=forward
    # Z-up: x=right, y=forward, z=up
    pos_zup = np.array([r_pos[0], r_pos[2], r_pos[1]])
    sim.data.mocap_pos[mocap_id] = pos_zup

    # Convert root quaternion from Y-up to Z-up
    # 90 degree rotation around X axis: [cos45, sin45, 0, 0] in wxyz
    rot_yup_to_zup = Rotation.from_euler('x', 90, degrees=True)

    # Convert HumanML3D wxyz → scipy xyzw
    r_rot_scipy = Rotation.from_quat(r_rot_quat[[1, 2, 3, 0]])

    # Compose: first apply data rotation, then coordinate system rotation
    final_rot = rot_yup_to_zup * r_rot_scipy

    # Convert back to wxyz for MuJoCo
    q_xyzw = final_rot.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    sim.data.mocap_quat[mocap_id] = q_wxyz


def main():
    # ── Load motion data 
    print(f"Loading motion data from {MOTION_FILE} ...")
    with open(MOTION_FILE, 'rb') as f:
        data = pickle.load(f)['motion_full_repr']

    # Shape: (9, 263, 1, 120) — num_samples, dims, 1, num_frames
    # Take sample 0, squeeze the singleton dim, transpose to (T, 263)
    mean = np.load(MEAN_PATH)   # shape (263,)
    std  = np.load(STD_PATH)    # shape (263,)

    motion = data[0, :, 0, :].T   # (120, 263)
    motion = motion * std + mean

    T = len(motion)
    print(f"  Motion shape: {motion.shape} — {T} frames")

    # ── Extract rotation data ─────────────────────────────────────────────────
    print("Extracting euler angles from rot6d ...")
    
    euler_angles, r_rot_quat, r_pos = extract_euler_angles(motion)



    print(f"  euler_angles: {euler_angles.shape}")  # (T, 21, 3)
    print(f"  r_rot_quat:   {r_rot_quat.shape}")    # (T, 4)
    print(f"  r_pos:        {r_pos.shape}")          # (T, 3)
    print(f"  Root Y range: [{r_pos[:, 1].min():.3f}, {r_pos[:, 1].max():.3f}]")

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    print(f"\nLoading MuJoCo scene from {SCENE_XML} ...")
    model  = mujoco_py.load_model_from_path(SCENE_XML)
    sim    = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    print(f"  nq = {model.nq}")
    print("  ✓ Scene loaded")

    # ── Replay loop ───────────────────────────────────────────────────────────
    print(f"\nStarting replay: {T} frames at {FPS} fps")
    print("  Controls: scroll=zoom, drag=rotate, Esc=exit")

    frame   = 0
    t_start = time.time()

    while True:
        # Set root position and orientation
        set_root(sim, r_rot_quat[frame], r_pos[frame])

        # Set all joint angles from rot6d
        set_qpos(sim, euler_angles[frame], r_rot_quat[frame], r_pos[frame])

        # Forward kinematics — MuJoCo computes all body positions from qpos
        sim.forward()

        # Render
        viewer.render()

        # Advance frame
        frame += 1
        if frame >= T:
            if LOOP:
                frame = 0
                print("  Looping...")
            else:
                print("  Done.")
                break

        # Framerate control
        elapsed  = time.time() - t_start
        expected = frame / FPS
        sleep_t  = expected - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


if __name__ == "__main__":
    main()