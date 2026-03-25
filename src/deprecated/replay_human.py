"""
replay_rot6d_corrected.py

Kinematic replay of HumanML3D 263-dim motion in MuJoCo.

ROOT CAUSE OF THE SUMO POSE:
  HumanML3D's IK computes local rotations relative to arbitrary rest-pose
  bone directions (t2m_raw_offsets), e.g. [1,0,0] for the left hip bone.
  MuJoCo's rest pose (qpos=0) has bones pointing in their actual T-pose
  directions, e.g. the thigh points downward ≈(0.21, -0.98, 0.07).

  The 60° hip rotation in HumanML3D is mostly compensating for this
  convention difference, NOT actual pose rotation. Applying it to MuJoCo
  (where the thigh already points down) doubles up the rotation → sumo pose.

FIX:
  R_mj_local[j] = R_offset[parent]^T  @  R_hml_local[j]  @  R_offset[j]

  where R_offset[j] = minimum rotation from MuJoCo rest bone dir to
                       HumanML3D raw_offset dir for joint j.
"""

import mujoco_py
import numpy as np
import pickle
import time
import os
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
# HumanML3D skeleton definition (from paramUtil.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Joint names for SMPL joints 0-21
SMPL_JOINT_NAMES = [
    "pelvis",           # 0
    "left_hip",         # 1
    "right_hip",        # 2
    "spine_1",          # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine_2",          # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine_3",          # 9  (chest)
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_clavicle",    # 13
    "right_clavicle",   # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
]

# Rest-pose bone direction vectors (from paramUtil.py t2m_raw_offsets)
# These define what direction each bone points in HumanML3D's "rest pose"
T2M_RAW_OFFSETS = np.array([
    [0, 0, 0],    # 0:  pelvis (root, no bone)
    [1, 0, 0],    # 1:  left_hip      → points RIGHT
    [-1, 0, 0],   # 2:  right_hip     → points LEFT
    [0, 1, 0],    # 3:  spine_1       → points UP
    [0, -1, 0],   # 4:  left_knee     → points DOWN
    [0, -1, 0],   # 5:  right_knee    → points DOWN
    [0, 1, 0],    # 6:  spine_2       → points UP
    [0, -1, 0],   # 7:  left_ankle    → points DOWN
    [0, -1, 0],   # 8:  right_ankle   → points DOWN
    [0, 1, 0],    # 9:  spine_3/chest → points UP
    [0, 0, 1],    # 10: left_foot     → points FORWARD
    [0, 0, 1],    # 11: right_foot    → points FORWARD
    [0, 1, 0],    # 12: neck          → points UP
    [1, 0, 0],    # 13: left_clavicle → points RIGHT
    [-1, 0, 0],   # 14: right_clavicle→ points LEFT
    [0, 0, 1],    # 15: head          → points FORWARD
    [0, -1, 0],   # 16: left_shoulder → points DOWN (arm hangs)
    [0, -1, 0],   # 17: right_shoulder→ points DOWN
    [0, -1, 0],   # 18: left_elbow    → points DOWN
    [0, -1, 0],   # 19: right_elbow   → points DOWN
    [0, -1, 0],   # 20: left_wrist    → points DOWN
    [0, -1, 0],   # 21: right_wrist   → points DOWN
], dtype=float)

# Kinematic chains (from paramUtil.py t2m_kinematic_chain)
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],          # right leg
    [0, 1, 4, 7, 10],          # left leg
    [0, 3, 6, 9, 12, 15],      # spine → head
    [9, 14, 17, 19, 21],       # right arm
    [9, 13, 16, 18, 20],       # left arm
]

# Parent for each SMPL joint (derived from kinematic chains)
SMPL_PARENTS = [-1] * 22
SMPL_PARENTS[0] = -1  # root
for chain in T2M_KINEMATIC_CHAIN:
    for i in range(1, len(chain)):
        SMPL_PARENTS[chain[i]] = chain[i - 1]

# MuJoCo joint positions from XML site elements (Y-up coordinate system)
# These define where each joint pivot is in the T-pose
MJ_JOINT_POS = {
    0:  np.array([0.0,     0.0,     0.0]),       # Pelvis
    1:  np.array([0.0677, -0.3147,  0.0214]),     # L_Hip
    2:  np.array([-0.0695, -0.3139, 0.0239]),     # R_Hip
    3:  np.array([-0.0043, -0.1144, 0.0015]),     # Torso (Spine1)
    4:  np.array([0.1020, -0.6899,  0.0169]),     # L_Knee
    5:  np.array([-0.1078, -0.6964, 0.0150]),     # R_Knee
    6:  np.array([0.0012,  0.0208,  0.0026]),     # Spine (Spine2)
    7:  np.array([0.0884, -1.0879, -0.0268]),     # L_Ankle
    8:  np.array([-0.0920, -1.0948, -0.0273]),    # R_Ankle
    9:  np.array([0.0026,  0.0737,  0.0280]),     # Chest (Spine3)
    10: np.array([0.1148, -1.1437,  0.0925]),     # L_Toe (L_Foot)
    11: np.array([-0.1174, -1.1430, 0.0961]),     # R_Toe (R_Foot)
    12: np.array([-0.0002, 0.2876, -0.0148]),     # Neck
    13: np.array([0.0815,  0.1955, -0.0060]),     # L_Thorax (L_Clavicle)
    14: np.array([-0.0791, 0.1926, -0.0106]),     # R_Thorax (R_Clavicle)
    15: np.array([0.0050,  0.3526,  0.0365]),     # Head
    16: np.array([0.1724,  0.2260, -0.0149]),     # L_Shoulder
    17: np.array([-0.1752, 0.2251, -0.0197]),     # R_Shoulder
    18: np.array([0.4320,  0.2132, -0.0424]),     # L_Elbow
    19: np.array([-0.4289, 0.2118, -0.0411]),     # R_Elbow
    20: np.array([0.6813,  0.2222, -0.0435]),     # L_Wrist
    21: np.array([-0.6842, 0.2196, -0.0467]),     # R_Wrist
}

# Mapping: rot6d index (0-20) → SMPL joint (1-21) → MuJoCo qpos start
# rot6d index i = SMPL joint i+1
HML_JOINT_TO_QPOS = [
    0,   # rot6d 0  → SMPL 1  (L_Hip)       → qpos 0,1,2
    12,  # rot6d 1  → SMPL 2  (R_Hip)       → qpos 12,13,14
    24,  # rot6d 2  → SMPL 3  (Torso)       → qpos 24,25,26
    3,   # rot6d 3  → SMPL 4  (L_Knee)      → qpos 3,4,5
    15,  # rot6d 4  → SMPL 5  (R_Knee)      → qpos 15,16,17
    27,  # rot6d 5  → SMPL 6  (Spine)       → qpos 27,28,29
    6,   # rot6d 6  → SMPL 7  (L_Ankle)     → qpos 6,7,8
    18,  # rot6d 7  → SMPL 8  (R_Ankle)     → qpos 18,19,20
    30,  # rot6d 8  → SMPL 9  (Chest)       → qpos 30,31,32
    9,   # rot6d 9  → SMPL 10 (L_Foot)      → qpos 9,10,11
    21,  # rot6d 10 → SMPL 11 (R_Foot)      → qpos 21,22,23
    33,  # rot6d 11 → SMPL 12 (Neck)        → qpos 33,34,35
    39,  # rot6d 12 → SMPL 13 (L_Thorax)    → qpos 39,40,41
    54,  # rot6d 13 → SMPL 14 (R_Thorax)    → qpos 54,55,56
    36,  # rot6d 14 → SMPL 15 (Head)        → qpos 36,37,38
    42,  # rot6d 15 → SMPL 16 (L_Shoulder)  → qpos 42,43,44
    57,  # rot6d 16 → SMPL 17 (R_Shoulder)  → qpos 57,58,59
    45,  # rot6d 17 → SMPL 18 (L_Elbow)     → qpos 45,46,47
    60,  # rot6d 18 → SMPL 19 (R_Elbow)     → qpos 60,61,62
    48,  # rot6d 19 → SMPL 20 (L_Wrist)     → qpos 48,49,50
    63,  # rot6d 20 → SMPL 21 (R_Wrist)     → qpos 63,64,65
]

# ═══════════════════════════════════════════════════════════════════════════════
# Compute rest-pose correction rotations (done once at startup)
# ═══════════════════════════════════════════════════════════════════════════════

def min_rotation_between_vectors(v_from, v_to):
    """Minimum-angle rotation that maps unit vector v_from to v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to / np.linalg.norm(v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    if dot > 0.99999:
        return Rotation.identity()
    if dot < -0.99999:
        # 180° rotation — pick any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)

    axis = np.cross(v_from, v_to)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(angle * axis)


def compute_rest_pose_offsets():
    """
    For each SMPL joint j (1-21), compute R_offset[j]:
    the minimum rotation from MuJoCo's rest bone direction to
    HumanML3D's raw_offset direction.

    R_offset[0] = identity (root has no bone).
    """
    R_offset = [Rotation.identity()] * 22  # index 0-21

    for j in range(1, 22):
        parent = SMPL_PARENTS[j]

        # MuJoCo rest bone direction: child_pos - parent_pos in T-pose
        mj_bone = MJ_JOINT_POS[j] - MJ_JOINT_POS[parent]
        mj_bone_norm = np.linalg.norm(mj_bone)
        if mj_bone_norm < 1e-8:
            R_offset[j] = Rotation.identity()
            continue
        mj_bone_dir = mj_bone / mj_bone_norm

        # HumanML3D rest bone direction
        hml_bone_dir = T2M_RAW_OFFSETS[j]
        hml_bone_norm = np.linalg.norm(hml_bone_dir)
        if hml_bone_norm < 1e-8:
            R_offset[j] = Rotation.identity()
            continue
        hml_bone_dir = hml_bone_dir / hml_bone_norm

        # R_offset maps mj_bone_dir → hml_bone_dir
        R_offset[j] = min_rotation_between_vectors(mj_bone_dir, hml_bone_dir)

    return R_offset


# ═══════════════════════════════════════════════════════════════════════════════
# Motion data processing
# ═══════════════════════════════════════════════════════════════════════════════

def rot6d_to_rotmat(rot6d):
    """rot6d (..., 6) → rotation matrix (..., 3, 3). Rows convention."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def recover_root_rot_pos(data):
    """Recover root rotation (quat wxyz) and position from 263-dim."""
    T = len(data)
    rot_vel = data[:, 0]
    r_rot_ang = np.zeros(T)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)

    r_rot_quat = np.zeros((T, 4))
    r_rot_quat[:, 0] = np.cos(r_rot_ang)
    r_rot_quat[:, 2] = np.sin(r_rot_ang)

    r_pos = np.zeros((T, 3))
    r_pos[1:, [0, 2]] = data[:-1, 1:3]

    r_rot_scipy = r_rot_quat[:, [1, 2, 3, 0]]
    rot_obj = Rotation.from_quat(r_rot_scipy)
    r_pos = rot_obj.inv().apply(r_pos)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = data[:, 3]

    return r_rot_quat, r_pos


def extract_local_rotmats(data_263):
    """
    Extract LOCAL rotation matrices (joints 1-21) from 263-dim data.

    Returns:
        local_rotmats: (T, 21, 3, 3)  — HumanML3D local rotations
        r_rot_quat:    (T, 4)         — root quaternion (wxyz)
        r_pos:         (T, 3)         — root position
    """
    T = len(data_263)
    rot6d = data_263[:, 67:193].reshape(T, 21, 6)
    local_rotmats = rot6d_to_rotmat(rot6d)  # (T, 21, 3, 3)
    r_rot_quat, r_pos = recover_root_rot_pos(data_263)
    return local_rotmats, r_rot_quat, r_pos


def correct_local_rotations(hml_local_rotmats, R_offset):
    """
    Convert HumanML3D local rotations to MuJoCo local rotations.

    Formula: R_mj_local[j] = R_offset[parent]^T @ R_hml_local[j] @ R_offset[j]

    hml_local_rotmats: (T, 21, 3, 3) — rot6d index 0-20 = SMPL joint 1-21
    R_offset:          list of 22 Rotation objects

    Returns: (T, 21, 3, 3) corrected local rotation matrices
    """
    T = hml_local_rotmats.shape[0]
    corrected = np.zeros_like(hml_local_rotmats)

    for rot6d_idx in range(21):
        smpl_j = rot6d_idx + 1          # SMPL joint index
        smpl_parent = SMPL_PARENTS[smpl_j]

        R_off_parent = R_offset[smpl_parent].as_matrix()  # (3,3)
        R_off_j      = R_offset[smpl_j].as_matrix()       # (3,3)

        # R_mj = R_off_parent^T @ R_hml @ R_off_j
        # Vectorized over T frames
        R_hml = hml_local_rotmats[:, rot6d_idx]  # (T, 3, 3)

        # (T,3,3) = (3,3)^T @ (T,3,3) @ (3,3)
        temp = np.einsum('ij,tjk->tik', R_off_parent.T, R_hml)
        corrected[:, rot6d_idx] = np.einsum('tij,jk->tik', temp, R_off_j)

    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
# MuJoCo interaction
# ═══════════════════════════════════════════════════════════════════════════════

def set_qpos(sim, corrected_rotmats_frame):
    """Set MuJoCo joint angles from corrected local rotation matrices."""
    qpos = np.zeros(sim.model.nq)

    for rot6d_idx in range(21):
        qpos_start = HML_JOINT_TO_QPOS[rot6d_idx]
        R = Rotation.from_matrix(corrected_rotmats_frame[rot6d_idx])
        angles = R.as_euler('ZYX', degrees=False)  # intrinsic ZYX for local hinges
        qpos[qpos_start + 0] = angles[0]  # Z
        qpos[qpos_start + 1] = angles[1]  # Y
        qpos[qpos_start + 2] = angles[2]  # X

    sim.data.qpos[:] = qpos


def set_root(sim, r_rot_quat_wxyz, r_pos):
    """Set mocap body position and orientation."""
    body_id  = sim.model.body_name2id('Pelvis')
    mocap_id = sim.model.body_mocapid[body_id]

    # Both HumanML3D and MuJoCo model are Y-up, so direct assignment
    sim.data.mocap_pos[mocap_id] = r_pos

    # Convert wxyz → scipy xyzw → back to MuJoCo wxyz
    q_scipy = Rotation.from_quat([
        r_rot_quat_wxyz[1], r_rot_quat_wxyz[2],
        r_rot_quat_wxyz[3], r_rot_quat_wxyz[0]
    ])
    q = q_scipy.as_quat()  # xyzw
    sim.data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]  # wxyz


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Compute rest-pose corrections (once) ─────────────────────────────────
    print("Computing rest-pose corrections ...")
    R_offset = compute_rest_pose_offsets()

    # Print correction angles for debugging
    for j in range(1, 22):
        angle_deg = np.degrees(R_offset[j].magnitude())
        parent = SMPL_PARENTS[j]
        mj_dir = MJ_JOINT_POS[j] - MJ_JOINT_POS[parent]
        mj_dir = mj_dir / np.linalg.norm(mj_dir)
        hml_dir = T2M_RAW_OFFSETS[j] / np.linalg.norm(T2M_RAW_OFFSETS[j])
        print(f"  {SMPL_JOINT_NAMES[j]:20s}: "
              f"mj_dir={mj_dir}, hml_dir={hml_dir}, "
              f"offset_angle={angle_deg:.1f}°")

    # ── Load motion data ─────────────────────────────────────────────────────
    print(f"\nLoading motion from {MOTION_FILE} ...")
    with open(MOTION_FILE, 'rb') as f:
        data = pickle.load(f)['motion_full_repr']

    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    motion = data[MOTION_IDX, :, 0, :].T  # (T, 263)
    motion = motion * std + mean

    T_frames = len(motion)
    print(f"  {T_frames} frames")

    # ── Extract and correct rotations ────────────────────────────────────────
    print("Extracting and correcting rotations ...")
    hml_local, r_rot_quat, r_pos = extract_local_rotmats(motion)
    mj_local = correct_local_rotations(hml_local, R_offset)

    # Sanity: corrected local rotation magnitudes at frame 0
    print("\nCorrected local rotation magnitudes at frame 0:")
    for rot6d_idx in range(21):
        smpl_j = rot6d_idx + 1
        angle_before = np.degrees(Rotation.from_matrix(hml_local[0, rot6d_idx]).magnitude())
        angle_after  = np.degrees(Rotation.from_matrix(mj_local[0, rot6d_idx]).magnitude())
        print(f"  {SMPL_JOINT_NAMES[smpl_j]:20s}: "
              f"before={angle_before:6.1f}°  after={angle_after:6.1f}°")

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
        set_root(sim, r_rot_quat[frame], r_pos[frame])
        # set_qpos(sim, mj_local[frame])
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