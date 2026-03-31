from models.human.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import sys
import torch
import mujoco
import time
import mujoco_viewer
from visualize.simplify_loc2rot import joints2smpl
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

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



class npy2obj:
    def __init__(self, cfg, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.cfg = cfg
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces

        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        print("Number of joint features is", self.nfeats)
        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            print("Shape of motion tensor after running smpl on joint locations is", motion_tensor.shape)
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        else:
            raise ValueError(f"Number of features {self.nfeats} is not supported yet!")

        
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape

        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        print("path to numpy file is", npy_path )
        
        self.vertices, self.rotations, self.x_translations, self.global_orient = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None, pose_rep='rot6d', translation=True, glob=True, jointstype='vertices', vertstrans=True, njoints_body = 24)

        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        # self.vertices += self.root_loc


    def run_mujoco_vis(self):
        scene_xml = os.path.expanduser(self.cfg.scene_xml)
        device    = self.cfg.get("device", "cpu")

        print(f"Source    : {self.cfg.get('source', 'diffusion')}")
        print(f"Scene XML : {scene_xml}")
        print(f"Device    : {device}")


        # 2. Run Rotation2xyz → SMPL body-joint ZYX Euler angles
        print("Running Rotation2xyz ...")
        joint_euler = self.rotation_to_euler(device)
        print(f"joint_euler shape  : {joint_euler.shape}")

        # 3. Initialise MuJoCo
        model, mjdata, viewer = self.setup_mujoco(scene_xml)

        # 4. Replay
        self.replay_loop(model, mjdata, viewer, self.x_translations, self.global_orient, joint_euler)


    def replay_loop(self, model, mjdata, viewer, r_pos, r_rot_quat, joint_euler):
        r_pos = r_pos.squeeze().transpose(1, 0)
        r_pos = r_pos[:, [0, 2, 1]]
        r_pos[:, 1] = -r_pos[:, 1]

        R_coord = Rotation.from_euler('X', 90, degrees=True)
        R_smpl = Rotation.from_matrix(r_rot_quat.squeeze())
        R_mocap = R_coord * R_smpl * R_coord.inv()
        r_rot_quat = R_mocap.as_quat()[:, [3, 0, 1, 2]]

        joint_euler = joint_euler.squeeze()

        T = len(joint_euler)
        fps = self.cfg.get("fps", 15)
        loop = self.cfg.get("loop", True)
        frame_duration = 1.0 / fps
        mocap_t = 0
        # Replace the hardcoded indices with name lookups
        human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_on_table")

        human_mocap_idx = model.body_mocapid[human_body_id]
        box_mocap_idx = model.body_mocapid[box_body_id]

        print(f"Human mocap index: {human_mocap_idx}, Box mocap index: {box_mocap_idx}")
        # Get the site ID for R_Hand
        rhand_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "R_Hand")

        while viewer.is_alive:
            frame_start = time.time()

            # Set human pose
            mjdata.mocap_pos[human_mocap_idx]  = r_pos[mocap_t]
            mjdata.mocap_quat[human_mocap_idx] = r_rot_quat[mocap_t]

            for smpl_j, q0 in SMPL_TO_QPOS.items():
                rot_idx = smpl_j - 1
                if rot_idx < joint_euler.shape[1]:
                    mjdata.qpos[q0:q0 + 3] = joint_euler[mocap_t, rot_idx]

            # First forward pass: computes site positions from human pose
            mujoco.mj_forward(model, mjdata)

            # Read R_Hand world position and snap box to it
            rhand_pos = mjdata.site_xpos[rhand_site_id].copy()
            mjdata.mocap_pos[box_mocap_idx] = rhand_pos

            # Second forward pass: updates box rendering position
            mujoco.mj_forward(model, mjdata)

            viewer.render()

            mocap_t += 1
            if mocap_t >= T:
                if loop:
                    mocap_t = 0
                else:
                    break

            elapsed = time.time() - frame_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        viewer.close()
        sys.exit()

    def rotation_to_euler(self, device: str) -> np.ndarray:
        # body_rotations: (1, T, 23, 3, 3)  –  SMPL joints 1-23
        body_rotations = self.rotations[0].cpu()  # (T, 23, 3, 3)
        T, njoints, _, _ = body_rotations.shape

        rot_np = body_rotations.numpy().reshape(T * njoints, 3, 3)
        euler  = Rotation.from_matrix(rot_np).as_euler("ZYX", degrees=False)  # (T*23, 3)
        return euler.reshape(T, njoints, 3)  # (T, 23, 3)


    def setup_mujoco(self, scene_xml: str):
        """Load scene.xml and create a MuJoCo viewer.

        Returns:
            model  : MjModel
            mjdata : MjData
            viewer : MujocoViewer
        """
        model  = mujoco.MjModel.from_xml_path(scene_xml)
        mjdata = mujoco.MjData(model)
        viewer = mujoco_viewer.MujocoViewer(model, mjdata)
        print(f"  qpos size : {mjdata.qpos.shape[0]}")
        print(f"  nmocap    : {model.nmocap}")
        return model, mjdata, viewer

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i), faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):

        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }

        np.save(save_path, data_dict)