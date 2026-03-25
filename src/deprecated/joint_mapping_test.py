# joint_mapping_test.py
import mujoco_py
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation
import time
SCENE_XML = os.path.expanduser("~/projects/human-modeling-3d/assets/human/scene.xml")
model  = mujoco_py.load_model_from_path(SCENE_XML)
sim    = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Set root upright
body_id  = sim.model.body_name2id('object')
mocap_id = sim.model.body_mocapid[body_id]
rot = Rotation.from_euler('x', 90, degrees=True)
q = rot.as_quat()
sim.data.mocap_pos[mocap_id]  = [0, 0, 1.0]
sim.data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]

# HumanML3D joint index → expected body part that should move
# We test by setting X rotation (qpos_start + 2) to 45 degrees
HML_JOINT_TO_QPOS = [
     0,  # 0: left_hip
    12,  # 1: right_hip
    24,  # 2: spine1
     3,  # 3: left_knee
    15,  # 4: right_knee
    27,  # 5: spine2
     6,  # 6: left_ankle
    18,  # 7: right_ankle
    30,  # 8: spine3
     9,  # 9: left_foot
    21,  # 10: right_foot
    33,  # 11: neck
    39,  # 12: left_collar
    54,  # 13: right_collar
    36,  # 14: head
    42,  # 15: left_shoulder
    57,  # 16: right_shoulder
    45,  # 17: left_elbow
    60,  # 18: right_elbow
    48,  # 19: left_wrist
    63,  # 20: right_wrist
]

HML_JOINT_NAMES = [
    'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]

for hml_idx, (name, qpos_start) in enumerate(zip(HML_JOINT_NAMES, HML_JOINT_TO_QPOS)):
    sim.data.qpos[:] = 0
    # Set X rotation to 45 degrees for this joint
    sim.data.qpos[qpos_start + 2] = 45
    sim.forward()

    print(f"HML joint {hml_idx:2d}: {name:20s} → qpos[{qpos_start+2}]  -- watch which body moves")
    for _ in range(60):  # show for 2 seconds
        viewer.render()
        time.sleep(1/30)