"""
DISCLAIMER - all or part of this file may be generated with a language model.
replay.py

Kinematic replay of motion data on a 22-joint SMPL stick figure.

Data format:  (N, 22, 3, T)
  N  = number of motion sequences
  22 = joints (HumanML3D convention)
  3  = x, y, z (Y-up coordinate system)
  T  = number of frames (120)

Each frame:
  1. Convert joint positions from Y-up to Z-up
  2. Set mocap_pos for each of the 22 joint sphere bodies
  3. Update fromto for each of the 21 bone capsule geoms
  4. Call sim.forward() — kinematics only, no physics
  5. Render
"""

import mujoco_py
import numpy as np
import pickle
import time
import os
import sys

# ── CONFIG ───────────────────────────────────────────────────────────────────
SCENE_XML   = os.path.expanduser("~/projects/human-modeling-3d/assets/human/stick_figure.xml")

MOTION_FILE = os.path.expanduser("/home/mb230/projects/human-modeling-3d/data/modiff-2022-samp-from-text/walking-example-dataset/after_inv_transform/results.pkl")  # ← change this

MOTION_IDX  = 0      # which motion sequence to replay (0 = first)
FPS         = 30     # playback speed
LOOP        = True   # loop the motion when it reaches the end
# ─────────────────────────────────────────────────────────────────────────────

# ── SMPL joint names (HumanML3D order) ───────────────────────────────────────
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]
NUM_JOINTS = len(JOINT_NAMES)  # 22

# ── SMPL parent array ─────────────────────────────────────────────────────────
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def yup_to_zup(positions):
    """
    Convert positions from Y-up to Z-up coordinate system.

    Input:  positions of shape (..., 3) where [..., 0]=x, [..., 1]=y, [..., 2]=z
            Y-up means: x=right, y=up, z=forward (into screen)

    Output: positions of shape (..., 3)
            Z-up means: x=right, y=forward, z=up

    The swap is simply: new_x = old_x, new_y = old_z, new_z = old_y
    """
    converted = positions.copy()
    converted[..., 1] = positions[..., 2]  # new y = old z
    converted[..., 2] = positions[..., 1]  # new z = old y
    return converted


def main():
    # ── Load motion data ─────────────────────────────────────────────────────
    print(f"Loading motion data from {MOTION_FILE} ...")
    with open(MOTION_FILE, 'rb') as f:
        data = pickle.load(f)['motion']

    # data shape: (N, 22, 3, T)
    motion = data[MOTION_IDX]          # shape: (22, 3, T)
    T = motion.shape[2]
    print(f"  Motion index {MOTION_IDX}: {T} frames, {NUM_JOINTS} joints")

    # Rearrange to (T, 22, 3) — easier to index by frame
    # motion[:, :, t] gives all joint positions at frame t
    # After transpose: motion_frames[t] gives (22, 3) array for frame t
    motion_frames = motion.transpose(2, 0, 1)  # (T, 22, 3)
    print(f"  Reshaped to: {motion_frames.shape}")

    # Convert entire sequence from Y-up to Z-up once upfront
    # Rather than converting every frame in the loop
    motion_zup = yup_to_zup(motion_frames)  # (T, 22, 3) in Z-up
    print(f"  Converted to Z-up coordinate system")
    print(f"  Pelvis Z range: [{motion_zup[:, 0, 2].min():.3f}, {motion_zup[:, 0, 2].max():.3f}]")
    print(f"  Head Z range:   [{motion_zup[:, 15, 2].min():.3f}, {motion_zup[:, 15, 2].max():.3f}]")

    # ── Load MuJoCo model ────────────────────────────────────────────────────
    print(f"\nLoading MuJoCo scene from {SCENE_XML} ...")
    model  = mujoco_py.load_model_from_path(SCENE_XML)
    sim    = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    print("  ✓ Scene loaded")
 
    # ── Build joint name → mocap index mapping ───────────────────────────────
    # sim.model.body_mocapid[body_id] gives mocap index for a body
    # We need this to set sim.data.mocap_pos[mocap_idx]
    joint_mocap_ids = []
    for name in JOINT_NAMES:
        body_name = f"joint_{name}"
        body_id   = sim.model.body_name2id(body_name)
        mocap_id  = sim.model.body_mocapid[body_id]
        joint_mocap_ids.append(mocap_id)
        print(f"  joint_{name:20s} body_id={body_id:3d}  mocap_id={mocap_id:3d}")

    # ── Build bone name → geom index mapping ────────────────────────────────
    # We update bone capsule fromto each frame
    # Geom fromto is in sim.model.geom_fromto — shape (ngeom, 6)
    # Each row is [x1 y1 z1 x2 y2 z2] — start and end of capsule
    bone_geom_ids = []
    for child_idx in range(NUM_JOINTS):
        parent_idx = PARENTS[child_idx]
        if parent_idx == -1:
            continue  # root has no bone
        geom_name = f"bone_{parent_idx}_{child_idx}"
        geom_id   = sim.model.geom_name2id(geom_name)
        bone_geom_ids.append((parent_idx, child_idx, geom_id))

    print(f"\n  {len(joint_mocap_ids)} joint mocap bodies mapped")
    print(f"  {len(bone_geom_ids)} bone geoms mapped")

    # ── Replay loop ──────────────────────────────────────────────────────────
    print(f"\nStarting replay: {T} frames at {FPS} fps")
    print("  Controls: Space=pause, Esc=exit, scroll=zoom, drag=rotate")

    frame    = 0
    paused   = False
    t_start  = time.time()

    while True:
        if not paused:
            # Get all joint positions for this frame — shape (22, 3)
            positions = motion_zup[frame]  # (22, 3) in Z-up

            # ── Set joint sphere positions ───────────────────────────────────
            for joint_idx, mocap_id in enumerate(joint_mocap_ids):
                sim.data.mocap_pos[mocap_id]  = positions[joint_idx]
                sim.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                # Identity quaternion — we only translate, no rotation per joint
                # This means mesh segments won't rotate to align with bones,
                # but for a stick figure this is fine.

            # ── Update bone capsule fromto ───────────────────────────────────
            # fromto = [px py pz  cx cy cz] where p=parent, c=child
            for parent_idx, child_idx, geom_id in bone_geom_ids:
                parent_pos = positions[parent_idx]  # (3,)
                child_pos  = positions[child_idx]   # (3,)
                # Cleaner version — replace the two lines with:
                sim.model.geom_fromto[geom_id*6 : geom_id*6+3] = parent_pos
                sim.model.geom_fromto[geom_id*6+3 : geom_id*6+6] = child_pos

            # ── Step simulation (kinematics only) ───────────────────────────
            sim.forward()

            # ── Advance frame ────────────────────────────────────────────────
            frame += 1
            if frame >= T:
                if LOOP:
                    frame = 0
                    print(f"  Looping... (completed one cycle)")
                else:
                    print("  Replay complete.")
                    break

        # ── Render ──────────────────────────────────────────────────────────
        viewer.render()

        # ── Framerate control ────────────────────────────────────────────────
        # Sleep just enough to maintain target FPS
        elapsed  = time.time() - t_start
        expected = frame / FPS
        sleep_t  = expected - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


if __name__ == "__main__":
    main()