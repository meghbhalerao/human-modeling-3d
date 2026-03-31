"""
replay_mujoco.py

Modular MuJoCo motion replay using SMPL joint rotations via Rotation2xyz.

Motion sources
--------------
  diffusion  – load from a diffusion-model .pkl output (HumanML3D 263-dim)
  dataset    – load from a raw dataset file (not yet implemented)

Pipeline
--------
  pkl  →  unnormalize (263-dim)
       →  build rot6d tensor  (1, 24, 6, T)
       →  Rotation2xyz  →  SMPL body-joint rotation matrices  (T, 23, 3, 3)
       →  ZYX Euler angles  →  MuJoCo qpos
       +  recover_root_rot_pos  →  MuJoCo mocap body (root position + quaternion)

Usage
-----
  python replay_mujoco.py
  python replay_mujoco.py source=diffusion motion_idx=2
  python replay_mujoco.py motion_file=path/to/results.pkl fps=20
"""

import os
import sys
import pickle

import numpy as np
import torch

import hydra

from models.human.rotation2xyz import Rotation2xyz
from data.humanml.scripts.motion_process import recover_root_rot_pos


# ── SMPL joint index → MuJoCo qpos start index ───────────────────────────────
# Depth-first joint traversal of scene.xml (3 hinges z,y,x per body):
#   L_Hip(0) L_Knee(3) L_Ankle(6) L_Toe(9)
#   R_Hip(12) R_Knee(15) R_Ankle(18) R_Toe(21)
#   Torso(24) Spine(27) Chest(30) Neck(33) Head(36)
#   L_Thorax(39) L_Shoulder(42) L_Elbow(45) L_Wrist(48) L_Hand(51)
#   R_Thorax(54) R_Shoulder(57) R_Elbow(60) R_Wrist(63) R_Hand(66)


# ── Data loading ──────────────────────────────────────────────────────────────



def _load_diffusion_motion(cfg: DictConfig):
    """Load a diffusion-model .pkl file containing HumanML3D 263-dim features.

    Expected pkl key: 'motion_full_repr'  shape (N, 263, 1, T)
    """
    motion_file = os.path.expanduser(cfg.motion_file)
    mean_path   = os.path.expanduser(cfg.mean_path)
    std_path    = os.path.expanduser(cfg.std_path)
    motion_idx  = cfg.motion_idx

    print(f"[diffusion] loading {motion_file}")
    with open(motion_file, "rb") as f:
        data = pickle.load(f)

    raw  = data["motion_full_repr"]          # (N, 263, 1, T)
    mean = np.load(mean_path)
    std  = np.load(std_path)

    motion_np = raw[motion_idx, :, 0, :].T  # (T, 263)
    motion_np = motion_np * std + mean
    T = len(motion_np)
    print(f"  {T} frames after unnormalization")

    motion = torch.tensor(motion_np, dtype=torch.float32)  # (T, 263)

    # Root trajectory for the mocap weld body
    r_rot_quat, r_pos = recover_root_rot_pos(motion)
    r_rot_quat = r_rot_quat.numpy()          # (T, 4) w,x,y,z
    r_pos      = _to_mujoco_axes(r_pos.numpy())  # (T, 3) z-up

    # Build (1, 24, 6, T) rot6d tensor
    motion_tensor = _build_rot6d_tensor(motion, r_rot_quat, T)

    return motion_tensor, r_pos, r_rot_quat



def _build_rot6d_tensor(
    motion: torch.Tensor,
    r_rot_quat: np.ndarray,
    T: int,
) -> torch.Tensor:
    """Convert (T, 263) HumanML3D motion to a (1, 24, 6, T) rot6d tensor.

    Joint layout (24 = SMPL joints 0-23):
      Joint  0  : global orient  – derived from the root quaternion
      Joints 1-21: body joints   – cont6d from motion[:, 67:193]
      Joints 22-23: hands        – identity rotation (absent in HumanML3D)
    """
    rot6d = torch.zeros(T, 24, 6, dtype=torch.float32)

    # Joint 0: root rotation  (quat w,x,y,z → rot matrix → two columns = rot6d)
    scipy_quat = r_rot_quat[:, [1, 2, 3, 0]]           # (T, 4) x,y,z,w
    root_mat   = torch.tensor(
        Rotation.from_quat(scipy_quat).as_matrix(),
        dtype=torch.float32,
    )  # (T, 3, 3)
    rot6d[:, 0, :3] = root_mat[:, :, 0]  # first column
    rot6d[:, 0, 3:] = root_mat[:, :, 1]  # second column

    # Joints 1-21: body cont6d from HumanML3D feature indices [67:193]
    rot6d[:, 1:22, :] = motion[:, 67:193].reshape(T, 21, 6)

    # Joints 22-23: identity  (rot6d of I = [1,0,0, 0,1,0])
    rot6d[:, 22, :] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
    rot6d[:, 23, :] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)

    return rot6d.permute(1, 2, 0).unsqueeze(0)  # (1, 24, 6, T)


# ── SMPL rotation conversion ──────────────────────────────────────────────────




# ── MuJoCo setup ─────────────────────────────────────────────────────────────


# ── Axis helpers ──────────────────────────────────────────────────────────────

def _to_mujoco_axes(r_pos: np.ndarray) -> np.ndarray:
    """Convert root position from HumanML3D (Y-up) to MuJoCo (Z-up) convention.

    Also negates the new Y axis to correct the backward-facing artefact.
    """
    r_pos = r_pos.copy()
    r_pos[:, [1, 2]] = r_pos[:, [2, 1]]  # swap Y ↔ Z
    r_pos[:, 1]      = -r_pos[:, 1]      # flip new Y
    return r_pos


@hydra.main(version_base=None, config_path="../configs", config_name="mujoco_replay")


if __name__ == "__main__":
    main()
