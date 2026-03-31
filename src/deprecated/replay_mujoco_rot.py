"""
replay_mujoco_rot.py

Rotation-based kinematic replay of HumanML3D 263-dim motion in MuJoCo.

Pipeline:
  263-dim → unnormalize
  → root rot/pos via recover_root_rot_pos  (mocap body)
  → cont6d joint rotations [67:193]        (qpos via ZYX Euler)
"""

import mujoco
import mujoco_viewer
import numpy as np
import torch
import pickle
import sys
import os
from scipy.spatial.transform import Rotation
from data.humanml.scripts.motion_process import recover_root_rot_pos
import hydra
from omegaconf import DictConfig
from data.humanml.common.quaternion import cont6d_to_matrix

# ═══════════════════════════════════════════════════════════════════════════════
# SMPL joint index → MuJoCo qpos start index
#
# Depth-first joint traversal of scene.xml (3 hinges z,y,x per body):
#   L_Hip(0) L_Knee(3) L_Ankle(6) L_Toe(9)
#   R_Hip(12) R_Knee(15) R_Ankle(18) R_Toe(21)
#   Torso(24) Spine(27) Chest(30) Neck(33) Head(36)
#   L_Thorax(39) L_Shoulder(42) L_Elbow(45) L_Wrist(48) L_Hand(51)
#   R_Thorax(54) R_Shoulder(57) R_Elbow(60) R_Wrist(63) R_Hand(66)
# ═══════════════════════════════════════════════════════════════════════════════


SMPL_TO_QPOS = {
    1:  0,   # L_Hip
    4:  3,   # L_Knee
    7:  6,   # L_Ankle
    10: 9,   # L_Toe
    2:  12,  # R_Hip
    5:  15,  # R_Knee
    8:  18,  # R_Ankle
    11: 21,  # R_Toe
    3:  24,  # Torso
    6:  27,  # Spine
    9:  30,  # Chest
    12: 33,  # Neck
    15: 36,  # Head
    13: 39,  # L_Thorax
    16: 42,  # L_Shoulder
    18: 45,  # L_Elbow
    20: 48,  # L_Wrist
    22: 51,  # L_Hand
    14: 54,  # R_Thorax
    17: 57,  # R_Shoulder
    19: 60,  # R_Elbow
    21: 63,  # R_Wrist
    23: 66,  # R_Hand
}

def precompute_joint_euler(joint_rot6d: torch.Tensor) -> np.ndarray:
    """
    Convert cont6d joint rotations to ZYX Euler angles for all frames.

    Args:
        joint_rot6d: (T, 21, 6) cont6d rotations for SMPL joints 1-21

    Returns:
        euler: (T, 21, 3) ZYX Euler angles in radians
               axis order matches scene.xml hinge declaration (z→y→x)
    """
    T = joint_rot6d.shape[0]

    rot_mat = cont6d_to_matrix(joint_rot6d.reshape(-1, 6))   # (T*21, 3, 3)
    rot_mat_np = rot_mat.numpy()
    euler = Rotation.from_matrix(rot_mat_np).as_euler('ZYX', degrees = False)  # (T*21, 3)
    return euler.reshape(T, 21, 3)


def mujoco_adj(r_pos):
    """
    This function performs some axis "adjustments" specific to the mujoco physics simulator to "align" the representations of motion outputted by the motion diffusion model to the axis and angle conventions used by the mujoco 3D physics simulator.
    """
    r_pos[:, [1, 2]] = r_pos[:, [2, 1]] # swap the y and z axis. MuJoCo uses the z up convention and the positions in xml files of mujoco are generally in the order of x, y, z
    r_pos[:, 1] = -r_pos[:, 1] # for some reason the human avatar is moving backward, and hence I am switching the sign of the x axis.
    return r_pos

@hydra.main(version_base=None, config_path="../configs", config_name="mujoco_replay")
def main(cfg: DictConfig):

    scene_xml   = os.path.expanduser(cfg.scene_xml)
    motion_file = os.path.expanduser(cfg.motion_file)
    mean_path   = os.path.expanduser(cfg.mean_path)
    std_path    = os.path.expanduser(cfg.std_path)
    motion_idx  = cfg.motion_idx
    fps         = cfg.fps

    # ── Load & unnormalize motion ─────────────────────────────────────────────
    print(f"Loading motion from {motion_file} ...")
    with open(motion_file, 'rb') as f:
        data = pickle.load(f)

    raw = data['motion_full_repr']
    print(f"Raw data shape: {raw.shape}")

    mean = np.load(mean_path)
    std  = np.load(std_path)
    motion_np = raw[motion_idx, :, 0, :].T        # (T, 263)
    motion_np = motion_np * std + mean
    T_frames  = len(motion_np)
    print(f"Motion: {T_frames} frames")

    motion = torch.tensor(motion_np, dtype=torch.float32)  # (T, 263)

    # ── Recover root trajectory ───────────────────────────────────────────────
    # r_rot_quat: (T, 4) quaternions (w, x, y, z), Y-axis rotation only
    # r_pos:      (T, 3) world positions (Y-up)
    r_rot_quat, r_pos = recover_root_rot_pos(motion)
    r_rot_quat = r_rot_quat.numpy()   # (T, 4)
    r_pos      = r_pos.numpy()        # (T, 3)

    # ── Extract & pre-convert joint rotations ─────────────────────────────────
    # HumanML3D 263-dim layout:
    #   [0]      root angular velocity (Y)
    #   [1:3]    root XZ linear velocity
    #   [3]      root height (Y)
    #   [4:67]   21×3 local joint positions
    #   [67:193] 21×6 cont6d local rotations  ← we use these
    #   [193:259] 21×3 local joint velocities
    #   [259:263] 4 foot contacts
    joint_rot6d = motion[:, 67:193].reshape(T_frames, 21, 6)   # (T, 21, 6)
    joint_euler = precompute_joint_euler(joint_rot6d)           # (T, 21, 3) ZYX radians

    # ── MuJoCo setup ─────────────────────────────────────────────────────────
    model  = mujoco.MjModel.from_xml_path(scene_xml)
    mjdata = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, mjdata)

    print(f"qpos size: {mjdata.qpos.shape[0]}")   # expected 69 (23 bodies × 3)
    print(f"nmocap:    {model.nmocap}")            # expected 1
    print(r_pos[0])
    # change axis orientation
    r_pos = mujoco_adj(r_pos)


    # ── Replay loop ───────────────────────────────────────────────────────────
    while True:
        for t in range(T_frames):
            if not viewer.is_alive:
                viewer.close()
                sys.exit()

            # -- Root position & orientation via mocap body -------------------
            # r_pos is in HumanML3D Y-up world frame; mocap body "object" is
            # the parent of Pelvis (which has pos="0 0 2" and a 90° X rotation
            # baked into scene.xml).  Set mocap directly from recovered data;
            # tweak scene.xml offsets if the avatar appears shifted.
            mjdata.mocap_pos[0]  = r_pos[t]
            mjdata.mocap_quat[0] = r_rot_quat[t]   # (w, x, y, z)

            # -- Joint rotations via qpos ------------------------------------
            # joint_euler[t, j] = [angle_z, angle_y, angle_x] in radians.
            # Matches the z→y→x hinge declaration order in scene.xml.
            for smpl_j in range(1, 22):             # SMPL joints 1–21
                q0 = SMPL_TO_QPOS[smpl_j]
                mjdata.qpos[q0:q0 + 3] = joint_euler[t, smpl_j - 1] 

            mujoco.mj_step(model, mjdata)
            viewer.render()

        if not cfg.loop:
            break

    viewer.close()
    sys.exit()


if __name__ == "__main__":
    main()
