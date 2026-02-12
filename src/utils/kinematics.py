import torch
import torch.nn.functional as F

class ForwardKinematics:
    def __init__(self, device='cpu'):
        self.device = device
        # SMPL kinematic tree (Standard)
        self.parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        
        # Hardcoded T-Pose offsets (Relative distance from parent joint)
        self.offsets = torch.zeros((24, 3), device=device)
        # Lower Body
        self.offsets[1] = torch.tensor([-0.1, 0.05, 0.0])   # L_Hip
        self.offsets[2] = torch.tensor([0.1, 0.05, 0.0])    # R_Hip
        self.offsets[4] = torch.tensor([0.0, -0.45, 0.0])   # L_Knee
        self.offsets[5] = torch.tensor([0.0, -0.45, 0.0])   # R_Knee
        self.offsets[7] = torch.tensor([0.0, -0.4, 0.0])    # L_Ankle
        self.offsets[8] = torch.tensor([0.0, -0.4, 0.0])    # R_Ankle
        self.offsets[10] = torch.tensor([0.0, -0.1, 0.15])  # L_Foot
        self.offsets[11] = torch.tensor([0.0, -0.1, 0.15])  # R_Foot
        # Spine
        self.offsets[3] = torch.tensor([0.0, 0.1, 0.0])     # Spine1
        self.offsets[6] = torch.tensor([0.0, 0.15, 0.0])    # Spine2
        self.offsets[9] = torch.tensor([0.0, 0.15, 0.0])    # Spine3
        self.offsets[12] = torch.tensor([0.0, 0.1, 0.0])    # Neck
        self.offsets[15] = torch.tensor([0.0, 0.15, 0.0])   # Head
        # Upper Body
        self.offsets[13] = torch.tensor([-0.1, 0.05, 0.0])  # L_Collar
        self.offsets[14] = torch.tensor([0.1, 0.05, 0.0])   # R_Collar
        self.offsets[16] = torch.tensor([-0.15, 0.0, 0.0])  # L_Shoulder
        self.offsets[17] = torch.tensor([0.15, 0.0, 0.0])   # R_Shoulder
        self.offsets[18] = torch.tensor([-0.3, 0.0, 0.0])   # L_Elbow
        self.offsets[19] = torch.tensor([0.3, 0.0, 0.0])    # R_Elbow
        self.offsets[20] = torch.tensor([-0.25, 0.0, 0.0])  # L_Wrist
        self.offsets[21] = torch.tensor([0.25, 0.0, 0.0])   # R_Wrist

    def rot6d_to_matrix(self, d6):
        # Helper to convert 6D -> 3x3 Matrix
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def forward(self, root_trans, rotations_6d):
        """
        Inputs:
            root_trans: (B, T, 3)
            rotations_6d: (B, T, 24, 6)
        Returns:
            global_positions: (B, T, 24, 3)
        """
        B, T, J, _ = rotations_6d.shape
        local_rots = self.rot6d_to_matrix(rotations_6d)
        
        # Prepare containers
        global_rots = [torch.eye(3, device=self.device).expand(B, T, 3, 3) for _ in range(J)]
        global_pos = [torch.zeros((B, T, 3), device=self.device) for _ in range(J)]
        
        # Root (Joint 0) logic
        global_rots[0] = local_rots[:, :, 0]
        global_pos[0] = root_trans
        
        # Propagate through the tree
        for i in range(1, J):
            parent = self.parents[i]
            # Rotation accumulates
            global_rots[i] = torch.matmul(global_rots[parent], local_rots[:, :, i])
            # Position = Parent_Pos + (Parent_Rot @ Offset)
            offset_vec = self.offsets[i].view(1, 1, 3, 1).expand(B, T, -1, -1)
            rotated_offset = torch.matmul(global_rots[parent], offset_vec).squeeze(-1)
            global_pos[i] = global_pos[parent] + rotated_offset
            
        return torch.stack(global_pos, dim=2)