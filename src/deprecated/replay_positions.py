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

import mujoco_py
import numpy as np
import pickle
import time
import os
import sys
from scipy.spatial.transform import Rotation

# ── CONFIG ───────────────────────────────────────────────────────────────────
SCENE_XML   = os.path.expanduser("~/projects/human-modeling-3d/assets/human/scene.xml")
MOTION_FILE = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/samples_humanml_trans_dec_512_bert_000600000_seed10_a_person_is_walking_forward_and_picking_something_from_the_ground_with_one_hand/results.pkl")
MEAN_PATH   = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Mean.npy")
STD_PATH    = os.path.expanduser("~/projects/human-modeling-3d/data/modiff-2022-samp-from-text/Std.npy")

MOTION_IDX  = 0
FPS         = 30
LOOP        = True
# ─────────────────────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# Position recovery (numpy version of recover_from_ric)
# ═══════════════════════════════════════════════════════════════════════════════

def recover_positions(data_263):
    """
    Convert 263-dim HumanML3D data to global joint positions.

    data_263: (T, 263)
    returns:
        positions: (T, 22, 3) global Y-up positions
        r_rot:     (T,) Rotation objects for root heading
    """
    T = len(data_263)

    # ── Root heading (Y-axis rotation) ───────────────────────────────────────
    rot_vel = data_263[:, 0]
    heading = np.zeros(T)
    heading[1:] = rot_vel[:-1]
    heading = np.cumsum(heading)

    # Quaternion (wxyz) for Y-axis rotation
    r_quat_wxyz = np.zeros((T, 4))
    r_quat_wxyz[:, 0] = np.cos(heading)
    r_quat_wxyz[:, 2] = np.sin(heading)

    # scipy uses xyzw
    r_quat_xyzw = r_quat_wxyz[:, [1, 2, 3, 0]]
    r_rot = Rotation.from_quat(r_quat_xyzw)

    # ── Root position ────────────────────────────────────────────────────────
    r_pos = np.zeros((T, 3))
    r_pos[1:, [0, 2]] = data_263[:-1, 1:3]
    r_pos = r_rot.inv().apply(r_pos)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = data_263[:, 3]

    # ── Joint positions ──────────────────────────────────────────────────────
    # Indices 4:67 = 21 joints × 3, root-relative in XZ, absolute Y, rotation-removed
    ric = data_263[:, 4:67].reshape(T, 21, 3)

    positions = np.zeros((T, 22, 3))
    positions[:, 0] = r_pos

    for t in range(T):
        # Undo root rotation (rotation-removed → world orientation)
        positions[t, 1:] = r_rot[t].inv().apply(ric[t])
        # Add root XZ (was subtracted during processing)
        positions[t, 1:, 0] += r_pos[t, 0]
        positions[t, 1:, 2] += r_pos[t, 2]

    return positions, r_rot, r_quat_wxyz


# ═══════════════════════════════════════════════════════════════════════════════
# Position → MuJoCo joint angles
# ═══════════════════════════════════════════════════════════════════════════════

def min_rotation_between_vectors(v_from, v_to):
    """Minimum-angle rotation mapping v_from to v_to (both unit vectors)."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to / np.linalg.norm(v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    if dot > 0.9999:
        return Rotation.identity()
    if dot < -0.9999:
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)
    axis = np.cross(v_from, v_to)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(angle * axis)


def positions_to_qpos(positions_frame, root_rot, nq):
    """
    Convert one frame of global joint positions to MuJoCo qpos.

    positions_frame: (22, 3) global positions
    root_rot: Rotation for root heading
    nq: number of qpos entries

    Algorithm:
      For each joint in topological order:
        1. bone_dir = normalize(child_pos - joint_pos)     [world frame]
        2. rest_dir = normalize(mj_child_pos - mj_joint_pos) [rest frame]
        3. Transform bone_dir into parent's accumulated frame
        4. R_local = min_rotation(rest_dir → bone_dir_in_parent)
        5. Decompose R_local into ZYX intrinsic Euler → qpos
    """
    qpos = np.zeros(nq)
    R_global = {0: root_rot}

    # Process joints 1-21 in topological order (parent index < child index)
    for j in range(1, 22):
        parent = SMPL_PARENT[j]

        if j in BONE_CHILD:
            child = BONE_CHILD[j]

            # Target bone direction in world frame
            target_bone = positions_frame[child] - positions_frame[j]
            norm = np.linalg.norm(target_bone)
            if norm < 1e-8:
                R_local = Rotation.identity()
            else:
                target_bone /= norm

                # Rest bone direction (world frame = parent frame at rest)
                rest_bone = REST_BONE_DIR[j]

                # Transform target into parent's current frame
                target_in_parent = R_global[parent].inv().apply(target_bone)

                # Minimum rotation from rest to target
                R_local = min_rotation_between_vectors(rest_bone, target_in_parent)
        else:
            # Leaf joint: no child → identity
            R_local = Rotation.identity()

        # Accumulate global rotation
        R_global[j] = R_global[parent] * R_local

        # Decompose and write qpos
        if j in SMPL_TO_QPOS:
            qpos_start = SMPL_TO_QPOS[j]
            angles = R_local.as_euler('ZYX', degrees=False)  # intrinsic Z-Y-X
            qpos[qpos_start + 0] = angles[0]  # Z
            qpos[qpos_start + 1] = angles[1]  # Y
            qpos[qpos_start + 2] = angles[2]  # X

    return qpos


# ═══════════════════════════════════════════════════════════════════════════════
# MuJoCo
# ═══════════════════════════════════════════════════════════════════════════════

def set_root(sim, r_pos, r_quat_wxyz):
    body_id  = sim.model.body_name2id('Pelvis')
    mocap_id = sim.model.body_mocapid[body_id]

    # Y-up → Z-up: swap Y and Z
    pos = np.array([r_pos[0], -r_pos[2], r_pos[1]])
    sim.data.mocap_pos[mocap_id] = pos

    # Rotate root orientation: 90° around X to convert Y-up → Z-up
    rot_fix = Rotation.from_euler('x', -90, degrees=True)
    r_rot_scipy = Rotation.from_quat([
        r_quat_wxyz[1], r_quat_wxyz[2], r_quat_wxyz[3], r_quat_wxyz[0]
    ])
    final = rot_fix * r_rot_scipy
    q = final.as_quat()  # xyzw
    sim.data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]  # wxyz

def yup_to_zup(pos):
    """Convert (N, 3) positions from Y-up to Z-up."""
    return np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], axis=-1)
# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load motion ──────────────────────────────────────────────────────────
    print(f"Loading motion from {MOTION_FILE} ...")

    with open(MOTION_FILE, 'rb') as f:
        data = pickle.load(f)
    print(data['motion'].shape)
    sys.exit()
    data = data['motion_full_repr']
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    motion = data[MOTION_IDX, :, 0, :].T  # (T, 263)
    motion = motion * std + mean

    T_frames = len(motion)
    print(f"  {T_frames} frames")

    # ── Recover global positions ─────────────────────────────────────────────
    print("Recovering global joint positions ...")
    positions, r_rot, r_quat_wxyz = recover_positions(motion)



    # positions = yup_to_zup(positions) 
    print(f"  Root Y range: [{positions[:, 0, 1].min():.3f}, {positions[:, 0, 1].max():.3f}]")

    # Sanity check: print bone lengths at frame 0
    print("\n  Bone lengths at frame 0 (should be anatomically reasonable):")
    bone_names = {
        (0,1): "pelvis→L_hip", (1,4): "L_hip→L_knee", (4,7): "L_knee→L_ankle",
        (0,3): "pelvis→spine1", (3,6): "spine1→spine2", (6,9): "spine2→chest",
        (9,12): "chest→neck", (12,15): "neck→head",
        (16,18): "L_shoulder→L_elbow", (18,20): "L_elbow→L_wrist",
    }
    for (a, b), name in bone_names.items():
        length = np.linalg.norm(positions[0, b] - positions[0, a])
        print(f"    {name:25s}: {length:.3f} m")

    # ── Load MuJoCo ──────────────────────────────────────────────────────────
    print(f"\nLoading scene from {SCENE_XML} ...")
    model  = mujoco_py.load_model_from_path(SCENE_XML)
    sim    = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    print(f"  nq = {model.nq}")

    # ── Replay ───────────────────────────────────────────────────────────────
    print(f"\nReplaying {T_frames} frames at {FPS} fps ...")
    frame = 0
    t_start = time.time()

    while True:
        # Set root position and heading
        set_root(sim, positions[frame, 0], r_quat_wxyz[frame])

        # Compute and set joint angles from positions
        # qpos = positions_to_qpos(positions[frame], r_rot[frame], model.nq)
        # sim.data.qpos[:] = qpos

        sim.forward()
        viewer.render()

        frame += 1
        if frame >= T_frames:
            if LOOP:
                frame = 0
                print("  Looping...")
            else:
                break

        elapsed  = time.time() - t_start
        expected = frame / FPS
        if expected > elapsed:
            time.sleep(expected - elapsed)


if __name__ == "__main__":
    main()