import torch
import utils.rotation_conversions as geometry
import sys

from models.human.smpl import SMPL, JOINTSTYPE_ROOT

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]


class Rotation2xyz:
    def __init__(self, device, dataset = 'amass'):
        self.device = device 
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob, jointstype, vertstrans, betas = None, beta = 0, glob_rot = None, get_rotations_back = False, njoints_body = 24, **kwargs):
        if pose_rep == 'xyz':
            return x
        
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype = bool, device = x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")
        
        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")


        if translation:
            x_translations = x[:, 0:3, :, :]
            x_rotations = x[:, :-3, :, :]
        else:
            x_rotations = x

        
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats   = x_rotations.shape

        if pose_rep == 'rotvec':
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == 'rotmat':
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == 'rotquat':
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == 'rot6d':
            #rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
            x_rotations = x_rotations.squeeze().reshape(nsamples, time, njoints_body, 6)
            rotations = geometry.rotation_6d_to_matrix(x_rotations)            
        else:
            raise NotImplementedError(f"Pose representation {pose_rep} is not defined!")

        if not glob:
            global_orient = torch.tensor(glob_rot, device = x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, :, 0]
            rotations = rotations[:, : , 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas], dtype=rotations.dtype, device=rotations.device)        
            betas[:, 1] = beta

        nsamples, time, njoints_body, _, _ = rotations.shape  

        # Fold time into batch
        rotations_flat = rotations.reshape(nsamples * time, njoints_body, 3, 3) 
        global_orient_flat = global_orient.reshape(nsamples * time, 1, 3, 3)    
        betas_flat = betas.unsqueeze(1).repeat(1, time, 1).reshape(nsamples * time, -1) 

        out = self.smpl_model(body_pose = rotations_flat, global_orient = global_orient_flat, betas = betas_flat)
        joints = out[jointstype]
        joints = joints.reshape(nsamples, time, joints.shape[1], 3)  # (5, 120, n_joints, 3)
        x_xyz = torch.empty(nsamples, time, joints.shape[2], 3, device = x.device, dtype = x.dtype)
  
        x_xyz[~mask]  = 0
        x_xyz[mask] = joints[mask]
        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        if jointstype != 'vertices':
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]


        if translation and vertstrans:
            x_translations = x_translations - x_translations[:,:,[0]]
            # x_xyz = x_xyz + x_translations[:, None, :, :]
            x_xyz = x_xyz + x_translations.squeeze(2)[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz