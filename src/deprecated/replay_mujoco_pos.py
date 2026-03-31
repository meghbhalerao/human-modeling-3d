"""
replay_mujoco_positions.py

Position-based kinematic replay of HumanML3D 263-dim motion in MuJoCo.

KEY INSIGHT: The cont6d rotations in HumanML3D's 263-dim representation are
NOT raw SMPL local rotations. They are derived via a custom IK process on a
"uniform skeleton" whose rest-pose bone directions differ from any particular
MuJoCo model. Using them directly causes the infamous "sumo stance" distortion.

Instead, this script:
  1. Recovers global 3D joint positions from the 263-dim vector
  2. Computes per-joint rotations via analytical IK matched to the MuJoCo skeleton
  3. Handles Y-up → Z-up conversion properly for both positions and orientations
"""

import mujoco
import mujoco_viewer
import numpy as np
import torch
import pickle
import sys
import os
from scipy.spatial.transform import Rotation

import hydra
from omegaconf import DictConfig

from data.humanml.scripts.motion_process import recover_root_rot_pos, recover_from_ric
from data.humanml.common.quaternion import cont6d_to_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# HumanML3D joint index → name (from paramUtil.py HumanML3D_JOINT_NAMES)
# ═══════════════════════════════════════════════════════════════════════════════
# 0:  pelvis         1:  left_hip        2:  right_hip       3:  spine_1
# 4:  left_knee      5:  right_knee      6:  spine_2         7:  left_ankle
# 8:  right_ankle    9:  spine_3         10: left_foot       11: right_foot
# 12: neck           13: left_clavicle   14: right_clavicle  15: head
# 16: left_shoulder  17: right_shoulder  18: left_elbow      19: right_elbow
# 20: left_wrist     21: right_wrist

# HumanML3D kinematic tree (parent[j] = parent joint index)
HML_PARENTS = {
    0: -1,  # root
    1: 0,   2: 0,   3: 0,
    4: 1,   5: 2,   6: 3,
    7: 4,   8: 5,   9: 6,
    10: 7,  11: 8,  12: 9,
    13: 9,  14: 9,  15: 12,
    16: 13, 17: 14, 18: 16,
    19: 17, 20: 18, 21: 19,
}

# HumanML3D joint index → MuJoCo body name
HML_TO_MJBODY = {
    1: "L_Hip",      2: "R_Hip",      3: "Torso",
    4: "L_Knee",     5: "R_Knee",     6: "Spine",
    7: "L_Ankle",    8: "R_Ankle",    9: "Chest",
    10: "L_Toe",     11: "R_Toe",     12: "Neck",
    13: "L_Thorax",  14: "R_Thorax",  15: "Head",
    16: "L_Shoulder", 17: "R_Shoulder", 18: "L_Elbow",
    19: "R_Elbow",   20: "L_Wrist",   21: "R_Wrist",
}

# HumanML3D joint → MuJoCo qpos start index
# (same as your SMPL_TO_QPOS - the joint ordering IS the same as SMPL)
HML_TO_QPOS = {
    1: 0,   2: 12,  3: 24,  4: 3,   5: 15,  6: 27,
    7: 6,   8: 18,  9: 30,  10: 9,  11: 21, 12: 33,
    13: 39, 14: 54, 15: 36, 16: 42, 17: 57,
    18: 45, 19: 60, 20: 48, 21: 63,
}


def recover_global_positions(motion: torch.Tensor, joints_num: int = 22) -> np.ndarray:
    """
    Recover global 3D joint positions from the 263-dim HumanML3D vector.

    Uses the rotation-invariant features (positions relative to root)
    plus the recovered root trajectory.

    Args:
        motion: (T, 263) tensor
    Returns:
        positions: (T, 22, 3) global positions in Y-up frame
    """
    # recover_from_ric extracts the rotation-invariant (root-relative) positions
    # from the 263-dim vector and returns them as global positions
    # The first 4+63 = 67 dims contain: root_rot_vel, root_lin_vel_xz, root_height, joint_positions
    local_positions = recover_from_ric(motion, joints_num)  # (T, 22, 3)

    # Also recover the root trajectory
    r_rot_quat, r_pos = recover_root_rot_pos(motion)

    # The positions from recover_from_ric are already in a "local" frame
    # (facing direction removed). We need to rotate them back to world frame.
    # Actually, recover_from_ric gives positions in the root-relative frame
    # with root rotation removed. Let's reconstruct world positions.

    T = motion.shape[0]
    positions = local_positions.numpy()  # (T, 22, 3)

    # The root position from recover_from_ric is at origin,
    # we need to add back the world root position
    # But first, the positions are in the "facing-removed" frame
    # We need to rotate by root orientation and translate by root position

    r_rot_quat_np = r_rot_quat.numpy()  # (T, 4) w,x,y,z
    r_pos_np = r_pos.numpy()            # (T, 3)

    world_positions = np.zeros_like(positions)
    for t in range(T):
        # Root rotation (Y-axis only)
        R = Rotation.from_quat([r_rot_quat_np[t, 1], r_rot_quat_np[t, 2],
                                 r_rot_quat_np[t, 3], r_rot_quat_np[t, 0]])  # scipy uses x,y,z,w
        # Rotate all positions and add root translation
        for j in range(joints_num):
            world_positions[t, j] = R.apply(positions[t, j])
        world_positions[t, :, :] += r_pos_np[t:t+1, :]  # broadcast root pos

    return world_positions


def get_mujoco_joint_positions(model):
    """
    Extract the rest-pose joint positions from the MuJoCo XML.
    Returns dict: body_name → position in pelvis frame (Y-up, before the 90° X rotation).

    In the XML, all bodies have pos="0 0 0" and the joint positions are specified
    directly. We extract the joint position for each body.
    """
    joint_pos = {}
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        body_id = model.jnt_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Only take the first hinge (z) for each body to get the joint position
        if joint_name.endswith("_z"):
            pos = model.jnt_pos[i].copy()  # position in body frame
            joint_pos[body_name] = pos

    return joint_pos


def compute_bone_directions_mujoco(mj_joint_pos):
    """
    Compute rest-pose bone directions from MuJoCo joint positions.
    These are in the pelvis local frame (Y-up, SMPL convention).

    Returns dict: HML_joint_index → unit bone direction (parent→child)
    """
    # Map body names to HML indices (reverse of HML_TO_MJBODY)
    mjbody_to_hml = {v: k for k, v in HML_TO_MJBODY.items()}

    bone_dirs = {}
    for hml_j in range(1, 22):
        body_name = HML_TO_MJBODY[hml_j]
        parent_hml = HML_PARENTS[hml_j]

        if parent_hml == 0:
            # Parent is pelvis (origin)
            parent_pos = np.array([0.0, 0.0, 0.0])
        else:
            parent_body = HML_TO_MJBODY[parent_hml]
            parent_pos = mj_joint_pos.get(parent_body, np.array([0.0, 0.0, 0.0]))

        child_pos = mj_joint_pos.get(body_name, np.array([0.0, 0.0, 0.0]))
        bone_vec = child_pos - parent_pos
        length = np.linalg.norm(bone_vec)
        if length > 1e-6:
            bone_dirs[hml_j] = bone_vec / length
        else:
            bone_dirs[hml_j] = np.array([0.0, 1.0, 0.0])  # fallback

    return bone_dirs


def position_based_ik(positions_t, mj_joint_pos, mj_bone_dirs):
    """
    Compute per-joint ZYX Euler angles from a single frame of global positions.

    For each joint j in the kinematic chain:
      1. Compute the actual bone direction (parent→child) from positions
      2. Compute the rest-pose bone direction from the MuJoCo skeleton
      3. Find the rotation from rest to actual
      4. Account for parent rotations (accumulated chain rotation)
      5. Decompose into ZYX Euler angles for the three hinge joints

    Args:
        positions_t: (22, 3) global joint positions for one frame (in Y-up)
        mj_joint_pos: dict of body_name → rest position in pelvis frame
        mj_bone_dirs: dict of HML_joint_index → unit rest bone direction

    Returns:
        euler_angles: dict of HML_joint_index → (3,) ZYX Euler angles
    """
    euler_angles = {}

    # We need to traverse the kinematic tree and compute LOCAL rotations
    # For each joint, the local rotation transforms from rest to posed
    # in the parent's frame

    # Store accumulated world rotations for each joint
    world_rotations = {0: np.eye(3)}  # Root starts with identity

    # BFS traversal order that respects parent-child relationships
    traversal_order = []
    queue = [0]
    while queue:
        j = queue.pop(0)
        traversal_order.append(j)
        children = [c for c, p in HML_PARENTS.items() if p == j]
        queue.extend(sorted(children))

    for j in traversal_order:
        if j == 0:
            continue  # Root is handled by mocap body

        parent = HML_PARENTS[j]

        # Actual bone direction from positions
        actual_bone = positions_t[j] - positions_t[parent]
        actual_length = np.linalg.norm(actual_bone)
        if actual_length < 1e-6:
            euler_angles[j] = np.array([0.0, 0.0, 0.0])
            world_rotations[j] = world_rotations[parent]
            continue
        actual_dir = actual_bone / actual_length

        # Rest-pose bone direction in WORLD frame
        # (In the MuJoCo model, the rest direction is in the pelvis frame,
        #  which is the world frame before any joint rotations)
        rest_dir = mj_bone_dirs[j]

        # Transform rest_dir to the parent's LOCAL frame
        # (after parent's accumulated rotation)
        R_parent_world = world_rotations[parent]
        rest_dir_world = R_parent_world @ rest_dir

        # The local rotation for this joint rotates rest_dir_world to actual_dir
        # R_local @ rest_dir_world = actual_dir (approximately)

        # However, for a proper kinematic chain, the local rotation R_j satisfies:
        # R_world_j = R_world_parent @ R_local_j
        # And the bone direction: actual_dir = R_world_j @ rest_dir_local
        #   where rest_dir_local is the bone direction in the joint's own rest frame

        # Simpler approach: compute the world rotation that aligns rest to actual
        # Using Rodrigues / qbetween
        R_local = rotation_between_vectors(rest_dir_world, actual_dir)

        # World rotation for this joint
        R_world_j = R_local @ R_parent_world
        world_rotations[j] = R_world_j

        # The qpos rotation is R_parent_world^{-1} @ R_world_j = R_parent_world^T @ R_local @ R_parent_world
        # which is the local rotation expressed in the parent's local frame
        R_qpos = R_parent_world.T @ R_local @ R_parent_world

        # Decompose into ZYX Euler
        try:
            euler = Rotation.from_matrix(R_qpos).as_euler('ZYX', degrees=False)
        except:
            euler = np.array([0.0, 0.0, 0.0])

        euler_angles[j] = euler

    return euler_angles


def rotation_between_vectors(v_from, v_to):
    """
    Compute the rotation matrix that rotates v_from to v_to.
    Both should be unit vectors.
    """
    v_from = v_from / (np.linalg.norm(v_from) + 1e-10)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-10)

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)

    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # Find a perpendicular vector
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, v_from) * v_from
        perp = perp / np.linalg.norm(perp)
        return Rotation.from_rotvec(np.pi * perp).as_matrix()

    # Rodrigues formula via skew-symmetric matrix
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + skew + skew @ skew / (1 + dot)
    return R


def yup_to_zup_position(pos):
    """Convert position from Y-up to Z-up: (x, y, z) → (x, -z, y)"""
    result = pos.copy()
    result[:, [1, 2]] = result[:, [2, 1]]
    result[:, 1] = -result[:, 1]
    return result


def yup_to_zup_quat_wxyz(quat_wxyz):
    """
    Convert a quaternion from Y-up to Z-up convention.
    The coordinate transform is: x→x, y→z, z→-y
    This is a 90° rotation around X: R_x(90°)

    For quaternion composition: q_zup = q_coord_change * q_yup
    where q_coord_change = (cos(45°), sin(45°), 0, 0) = (0.707, 0.707, 0, 0)
    in (w,x,y,z) format.
    """
    # Rotation from Y-up to Z-up: 90° around X axis
    R_yup2zup = Rotation.from_euler('X', 90, degrees=True)

    # Convert input quaternion
    R_input = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2],
                                   quat_wxyz[3], quat_wxyz[0]])  # scipy: x,y,z,w

    # Compose: first apply Y-up rotation, then coordinate change
    R_result = R_yup2zup * R_input

    # Convert back to w,x,y,z
    q_scipy = R_result.as_quat()  # x,y,z,w
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])


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
    r_rot_quat, r_pos = recover_root_rot_pos(motion)
    r_rot_quat = r_rot_quat.numpy()   # (T, 4) w,x,y,z
    r_pos      = r_pos.numpy()        # (T, 3) Y-up

    # ── Recover global joint positions ────────────────────────────────────────
    # Use the rotation-invariant position features for reliable 3D positions
    joints_num = 22
    local_joints = recover_from_ric(motion, joints_num).numpy()  # (T, 22, 3) root-relative, facing-removed

    # Reconstruct world-frame positions
    world_positions = np.zeros((T_frames, joints_num, 3))
    for t in range(T_frames):
        # Root rotation: convert (w,x,y,z) to scipy (x,y,z,w)
        R_root = Rotation.from_quat([r_rot_quat[t, 1], r_rot_quat[t, 2],
                                      r_rot_quat[t, 3], r_rot_quat[t, 0]])
        for j in range(joints_num):
            world_positions[t, j] = R_root.apply(local_joints[t, j]) + r_pos[t]

    print(f"World positions shape: {world_positions.shape}")
    print(f"Root pos range: x=[{r_pos[:,0].min():.2f}, {r_pos[:,0].max():.2f}] "
          f"y=[{r_pos[:,1].min():.2f}, {r_pos[:,1].max():.2f}] "
          f"z=[{r_pos[:,2].min():.2f}, {r_pos[:,2].max():.2f}]")

    # ── MuJoCo setup ─────────────────────────────────────────────────────────
    model  = mujoco.MjModel.from_xml_path(scene_xml)
    mjdata = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, mjdata)

    print(f"qpos size: {mjdata.qpos.shape[0]}")
    print(f"nmocap:    {model.nmocap}")

    # ── Extract MuJoCo skeleton rest-pose info ────────────────────────────────
    mj_joint_pos = get_mujoco_joint_positions(model)
    mj_bone_dirs = compute_bone_directions_mujoco(mj_joint_pos)

    print("\nMuJoCo rest-pose bone directions (Y-up, pelvis frame):")
    for j in sorted(mj_bone_dirs.keys()):
        body = HML_TO_MJBODY[j]
        d = mj_bone_dirs[j]
        print(f"  Joint {j:2d} ({body:12s}): [{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}]")

    # ── Precompute all per-frame IK ──────────────────────────────────────────
    print("\nComputing position-based IK for all frames...")
    all_euler = []
    for t in range(T_frames):
        euler_dict = position_based_ik(world_positions[t], mj_joint_pos, mj_bone_dirs)
        all_euler.append(euler_dict)

    # ── Convert root trajectory to Z-up ──────────────────────────────────────
    r_pos_zup = yup_to_zup_position(r_pos)

    # Convert root quaternions to Z-up
    r_quat_zup = np.zeros_like(r_rot_quat)
    for t in range(T_frames):
        r_quat_zup[t] = yup_to_zup_quat_wxyz(r_rot_quat[t])

    # The Pelvis body in the XML has quat="0.707 0.707 0 0" (90° around X)
    # This is ALREADY the Y-up → Z-up conversion baked into the model.
    # So the mocap body should receive the Z-up quaternion, BUT we need to
    # undo the pelvis rotation since it's already there.
    # Actually: mocap_quat controls the "object" body, which is the parent
    # of Pelvis. The Pelvis then applies its own 90° X rotation.
    # So we should send the Y-up quaternion to mocap (NOT Z-up), and let
    # the Pelvis quat handle the coordinate conversion.
    # OR we compose: mocap_quat = q_zup * inv(q_pelvis_offset)

    # Pelvis offset quat: (w=0.707, x=0.707, y=0, z=0) = 90° around X
    R_pelvis_offset = Rotation.from_quat([0.707, 0.0, 0.0, 0.707])  # scipy: x,y,z,w

    r_quat_mocap = np.zeros_like(r_rot_quat)
    for t in range(T_frames):
        # The root rotation is a Y-axis rotation in Y-up frame.
        # In MuJoCo: mocap body → Pelvis(90°X) → joints
        # We want: world rotation of Pelvis = R_yup2zup * R_root
        # Since: world_rot_pelvis = R_mocap * R_pelvis_offset
        # Therefore: R_mocap = R_yup2zup * R_root * R_pelvis_offset^{-1}
        R_root = Rotation.from_quat([r_rot_quat[t, 1], r_rot_quat[t, 2],
                                      r_rot_quat[t, 3], r_rot_quat[t, 0]])
        R_yup2zup = Rotation.from_euler('X', 90, degrees=True)
        R_mocap = R_yup2zup * R_root * R_pelvis_offset.inv()

        q = R_mocap.as_quat()  # scipy: x,y,z,w
        r_quat_mocap[t] = [q[3], q[0], q[1], q[2]]  # MuJoCo: w,x,y,z

    # ── Replay loop ───────────────────────────────────────────────────────────
    print(f"\nStarting replay: {T_frames} frames at {fps} fps")
    while True:
        for t in range(T_frames):
            if not viewer.is_alive:
                viewer.close()
                sys.exit()

            # -- Root position & orientation via mocap body ---
            mjdata.mocap_pos[0]  = r_pos_zup[t]
            mjdata.mocap_quat[0] = r_quat_mocap[t]

            # -- Joint rotations via qpos (from position-based IK) ---
            euler_dict = all_euler[t]
            for hml_j in range(1, 2):
                q0 = HML_TO_QPOS[hml_j]
                if hml_j in euler_dict:
                    mjdata.qpos[q0:q0 + 3] = euler_dict[hml_j]
                else:
                    mjdata.qpos[q0:q0 + 3] = 0.0

            mujoco.mj_step(model, mjdata)
            viewer.render()

        if not cfg.loop:
            break

    viewer.close()
    sys.exit()


if __name__ == "__main__":
    main()