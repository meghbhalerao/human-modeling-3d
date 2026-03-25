"""
replay_positions.py

Position-based kinematic replay of HumanML3D 263-dim motion in MuJoCo.

Instead of fighting rotation convention differences between HumanML3D and
MuJoCo, we use the POSITION channel (indices 4:67) to get global joint
positions, then compute MuJoCo Euler angles geometrically from bone directions.

Pipeline:
  263-dim → unnormalize → recover_from_ric → global joint positions (22×3)
  → for each joint: compute bone direction → local rotation → ZYX Euler angles
  → set qpos
"""

import mujoco
import mujoco_viewer
import numpy as np
import torch 
import pickle
import time
import os
import sys
from scipy.spatial.transform import Rotation
from data.humanml.scripts.motion_process import recover_from_ric
from data.humanml.common.quaternion import cont6d_to_matrix
import mediapy as media
from typing import List
import hydra
from omegaconf import DictConfig

# ═══════════════════════════════════════════════════════════════════════════════
# SMPL skeleton definition (22 joints)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  0: pelvis         (root)
#  1: left_hip       2: right_hip      3: spine_1
#  4: left_knee      5: right_knee     6: spine_2
#  7: left_ankle     8: right_ankle    9: spine_3 (chest)
# 10: left_foot     11: right_foot    12: neck
# 13: left_clavicle 14: right_clavicle 15: head
# 16: left_shoulder 17: right_shoulder
# 18: left_elbow    19: right_elbow
# 20: left_wrist    21: right_wrist

SMPL_PARENT = {
    0: -1,
    1: 0, 2: 0, 3: 0,
    4: 1, 5: 2, 6: 3,
    7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
    15: 12, 16: 13, 17: 14,
    18: 16, 19: 17,
    20: 18, 21: 19,
}

# For computing bone direction at each joint: which child to use
# Leaf joints (10, 11, 15, 20, 21) have no child → identity rotation
BONE_CHILD = {
    1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9,
    7: 10, 8: 11, 9: 12,
    12: 15, 13: 16, 14: 17,
    16: 18, 17: 19, 18: 20, 19: 21,
}

# MuJoCo joint pivot positions from scene.xml (Y-up, in meters)
MJ_POS = {
    0:  np.array([0.0,     0.0,     0.0]),
    1:  np.array([0.0677, -0.3147,  0.0214]),
    2:  np.array([-0.0695,-0.3139,  0.0239]),
    3:  np.array([-0.0043,-0.1144,  0.0015]),
    4:  np.array([0.1020, -0.6899,  0.0169]),
    5:  np.array([-0.1078,-0.6964,  0.0150]),
    6:  np.array([0.0012,  0.0208,  0.0026]),
    7:  np.array([0.0884, -1.0879, -0.0268]),
    8:  np.array([-0.0920,-1.0948, -0.0273]),
    9:  np.array([0.0026,  0.0737,  0.0280]),
    10: np.array([0.1148, -1.1437,  0.0925]),
    11: np.array([-0.1174,-1.1430,  0.0961]),
    12: np.array([-0.0002, 0.2876, -0.0148]),
    13: np.array([0.0815,  0.1955, -0.0060]),
    14: np.array([-0.0791, 0.1926, -0.0106]),
    15: np.array([0.0050,  0.3526,  0.0365]),
    16: np.array([0.1724,  0.2260, -0.0149]),
    17: np.array([-0.1752, 0.2251, -0.0197]),
    18: np.array([0.4320,  0.2132, -0.0424]),
    19: np.array([-0.4289, 0.2118, -0.0411]),
    20: np.array([0.6813,  0.2222, -0.0435]),
    21: np.array([-0.6842, 0.2196, -0.0467]),
}

# SMPL joint index → MuJoCo qpos start index
SMPL_TO_QPOS = {
    1: 0,   2: 12,  3: 24,  4: 3,   5: 15,  6: 27,
    7: 6,   8: 18,  9: 30, 10: 9,  11: 21, 12: 33,
    13: 39, 14: 54, 15: 36, 16: 42, 17: 57,
    18: 45, 19: 60, 20: 48, 21: 63,
}

# Pre-compute rest bone directions (normalized)
REST_BONE_DIR = {}
for j, child in BONE_CHILD.items():
    d = MJ_POS[child] - MJ_POS[j]
    REST_BONE_DIR[j] = d / np.linalg.norm(d)

def rot_to_quat(rot: Rotation) -> List:
    """Convert a scipy rotation to a mujoco conform quaternion (w, x, y, z)."""
    quat = rot.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]


def quat_to_rot(quat: List) -> Rotation:
    """Convert a mujoco conform quaternion (w, x, y, z) to a scipy rotation."""
    scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
    return Rotation.from_quat(scipy_quat)


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def apply_transformations(m, d, pose, root_pose):
    for i, transform in enumerate(pose):
        rotation_matrix = cont6d_to_matrix(transform)
        translation = root_pose

        # Convert rotation to quaternion (xyzw) and reorder to wxyz
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation_matrix.reshape(-1))

        if i == 0:  # Free joint: apply both translation and rotation
            d.qpos[:3] = translation  # translation - TODO
            d.qpos[3:7] = quaternion  # rotation
        else:  # Ball joints: only apply quaternion rotation
            joint_qpos_start = 7 + 4 * (i - 1)
            print(quaternion)
            d.qpos[joint_qpos_start:joint_qpos_start + 4] = quaternion


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="../configs", config_name="mujoco_replay")
def main(cfg: DictConfig):

    scene_xml   = os.path.expanduser(cfg.scene_xml)
    motion_file = os.path.expanduser(cfg.motion_file)
    mean_path   = os.path.expanduser(cfg.mean_path)
    std_path    = os.path.expanduser(cfg.std_path)
    motion_idx  = cfg.motion_idx
    fps         = cfg.fps
    loop        = cfg.loop
    motion_rep  = cfg.motion_rep  # "position" or "rotation"

    print(f"Motion representation: {motion_rep}")

    # ── Load motion ──────────────────────────────────────────────────────────
    print(f"Loading motion from {motion_file} ...")

    with open(motion_file, 'rb') as f:
        data = pickle.load(f)
    data = data['motion_full_repr']
    print("Raw data shape is ", data.shape)
    mean = np.load(mean_path)
    std  = np.load(std_path)
    motion = data[motion_idx, :, 0, :].T  # (T, 263)
    motion = motion * std + mean
    print("Motion shape is ", motion.shape)
    njoints = 22
    T_frames = len(motion)
    print(f"  {T_frames} frames")
    motion = torch.tensor(motion)

    if motion_rep == 'position':
        motion = recover_from_ric(motion, njoints)
    elif motion_rep == 'rotation':
        root_motion = motion[:, 4:7]
        motion = motion[:, 67: 193].reshape(T_frames, njoints - 1, 6)


    print(f"Motion shape using motion representation {motion_rep} is ", motion.shape) 


    # initialize mujoco simulation from xml
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    print("shape of qpos of data is ", data.qpos.shape)
    print("shape of xpos of data is ", data.xpos.shape)
    print("shape of mocap pos of data is", data.mocap_pos.shape)

    glob_counter = 0
    print("number of bodies in model is ", model.nbody)
    print("number of mocap bodies is", model.nmocap)
    print("number of geoms is", model.ngeom)

    while True:
        for i in range(T_frames):
            frame_idx = glob_counter % T_frames
            glob_counter += 1
            if viewer.is_alive:
                mujoco.mj_step(model, data)
                data.mocap_pos[0] = root_motion[frame_idx] 
                viewer.render()
            else:
                break

    viewer.close()
    sys.exit()


if __name__ == "__main__":
    main()