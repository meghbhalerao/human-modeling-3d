import torch
import json
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from diffusion.arch.transformer import MotionDiT
from diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from utils.kinematics import ForwardKinematics
from viz_engine import SkeletonPlayer

class MotionSampler:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        # 1. Setup Input Dimensions
        self.use_body = 'body' in cfg.data.parts
        input_dim = 3
        if self.use_body: input_dim += 144
        if 'left' in cfg.data.parts: input_dim += 96
        if 'right' in cfg.data.parts: input_dim += 96
        
        # 2. Initialize and Load Model
        self.model = MotionDiT(
            input_dim=input_dim,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth
        ).to(self.device)
        
        ckpt_path = cfg.sampling.checkpoint_path
        print(f"Loading checkpoint from: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        
        # 3. Initialize Diffusion
        betas = get_named_beta_schedule(cfg.diffusion.schedule, cfg.diffusion.steps)
        self.diffusion = GaussianDiffusion(betas=betas, model_mean_type="eps")
        
        # 4. Initialize Kinematics
        self.fk = ForwardKinematics(device=self.device)

    @torch.no_grad()
    def sample(self):
        output_dir = Path(self.cfg.sampling.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        B = self.cfg.sampling.num_samples
        T = self.cfg.data.max_seq_len
        shape = (B, T, self.model.input_proj.in_features)
        
        print(f"Generating {B} samples of length {T}...")
        
        # 1. Reverse Diffusion
        noise = torch.randn(shape, device=self.device)
        sample = self.diffusion.p_sample_loop(
            self.model,
            shape,
            noise=noise,
            clip_denoised=False,
            progress=True
        )
        
        # 2. Process and Save
        self._save_results(sample, output_dir)

    def _save_results(self, samples, output_dir):
        """Convert Neural Net output -> positions -> video"""
        B, T, _ = samples.shape
        
        for i in range(B):
            motion_vec = samples[i]
            
            # --- Decode Vector ---
            # Assume order: Root(3) -> Body(144) -> Hands...
            root_trans = motion_vec[:, :3]
            
            # Use FK if body is present
            smpl_positions = None
            if self.use_body:
                body_rots = motion_vec[:, 3:147].reshape(T, 24, 6)
                smpl_positions = self.fk.forward(root_trans.unsqueeze(0), body_rots.unsqueeze(0))
                smpl_positions = smpl_positions.squeeze(0) # (T, 24, 3)

            # --- Prepare Data for Viz ---
            frames_data = []
            for t in range(T):
                frame = {'root_translation': root_trans[t].tolist()}
                if smpl_positions is not None:
                    frame['smpl_body'] = {'positions': smpl_positions[t].tolist()}
                frames_data.append(frame)
            
            # --- Render ---
            if self.cfg.sampling.save_viz:
                temp_path = output_dir / f"temp_{i}.json"
                out_mp4 = output_dir / f"sample_{i}.mp4"
                
                with open(temp_path, 'w') as f:
                    json.dump({'frames': frames_data}, f)
                
                print(f"Rendering {out_mp4}...")
                player = SkeletonPlayer(temp_path)
                player.render(str(out_mp4), components=['body'], fps=self.cfg.sampling.fps)
                temp_path.unlink() # Cleanup