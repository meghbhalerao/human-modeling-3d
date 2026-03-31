"""

Code from - https://colab.research.google.com/drive/154YTdflNrLCGHZ98jPSt_TZeafrYqFR-?usp=sharing#scrollTo=NFxtQ6G12r8S


replay_skeleton.py
 
Replay a sequence of poses on a MuJoCo skeleton loaded from an XML file.
Each pose is a list of 4x4 transformation matrices (one per joint).
The first joint uses a free joint (translation + rotation),
the rest use ball joints (rotation only).
 
Launches an interactive MuJoCo viewer window — you can rotate, zoom, and pan
the camera while the motion plays back in real time.



"""
 
import os
import time
import numpy as np
import mujoco
import mujoco_viewer
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
XML_PATH  = os.path.expanduser("~/projects/human-modeling-3d/assets/human/ball_joint_human.xml")
FRAMERATE = 30    # playback speed in Hz
LOOP      = True  # loop the motion when it reaches the end
# ─────────────────────────────────────────────────────────────────────────────
 
 
def apply_transformations(m, d, pose):
    """
    Apply a single frame of pose to the MuJoCo data.
 
    Args:
        m    : MjModel
        d    : MjData
        pose : array of shape (num_joints, 4, 4) - one 4x4 transform per joint
    """
    for i, transform in enumerate(pose):
        rotation_matrix = transform[:3, :3]
        translation     = transform[:3, 3]
 
        # convert rotation matrix to quaternion in MuJoCo format (w, x, y, z)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation_matrix.reshape(-1))
 
        if i == 0:
            # free joint: set both translation and rotation
            d.qpos[:3]  = translation
            d.qpos[3:7] = quaternion
        else:
            # ball joint: rotation only, no translation
            joint_qpos_start = 7 + 4 * (i - 1)
            d.qpos[joint_qpos_start:joint_qpos_start + 4] = quaternion
 
 
def run_interactive(xml_path, pose_sequence, framerate=FRAMERATE, loop=LOOP):
    """
    Launch an interactive MuJoCo viewer and replay the pose sequence in real time.
 
    Args:
        xml_path      : path to the MuJoCo XML file
        pose_sequence : array of shape (T, num_joints, 4, 4)
        framerate     : playback speed in Hz
        loop          : whether to loop the motion
    """
    print(f"Loading model from {xml_path} ...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
 
    print(f"qpos shape   : {data.qpos.shape}")
    print(f"Total frames : {len(pose_sequence)}")
    print(f"Framerate    : {framerate} Hz")
    print(f"Loop         : {loop}")
    print("Launching viewer - close the window to exit.")
 
    frame_duration = 1.0 / framerate
    viewer = mujoco_viewer.MujocoViewer(model, data)
    mujoco.mj_resetData(model, data)
    frame_i = 0
 
    while viewer.is_alive:
        t_start = time.time()
 
        apply_transformations(model, data, pose_sequence[frame_i])
        mujoco.mj_step(model, data)
        viewer.render()
 
        frame_i += 1
        if frame_i >= len(pose_sequence):
            if loop:
                frame_i = 0
            else:
                break
 
        # sleep for the remaining frame time to hit target framerate
        elapsed = time.time() - t_start
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
 
    viewer.close()
    print("Viewer closed.")
 
 
if __name__ == "__main__":
    # ── replace this with your actual pose loading logic ──────────────────────
    
 
    run_interactive(
        xml_path      = XML_PATH,
        pose_sequence = pose_sequence,
        framerate     = FRAMERATE,
        loop          = LOOP,
    )
 