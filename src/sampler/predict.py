import os
from typing import Dict, List
from omegaconf import DictConfig
from pathlib import Path

import torch

import data.humanml.utils.paramUtil as paramUtil
from data.get_data import get_dataset_loader
from data.humanml.scripts.motion_process import recover_from_ric
from data.humanml.utils.plot_script import plot_3d_motion
from data.tensors import collate
from utils.sampler_utils import ClassifierFreeSampleModel
from utils import dist_utils
from utils.model_utils import create_model_and_diffusion, load_model_wo_clip
from visualize.motions2hik import motions2hik
from sampler.generate import construct_template_variables
from data.humanml.scripts.motion_process import recover_from_ric, get_target_location, sample_goal

"""
In case of matplot lib issues it may be needed to delete model/data_loaders/humanml/utils/plot_script.py" in lines 89~92 as
suggested in https://github.com/GuyTevet/motion-diffusion-model/issues/6
"""


class MotionGenerator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.num_frames = int(self.config.sampling.fps * self.config.sampling.motion_length)
        
        print('Loading dataset...')
        # temporary data
        self.data = get_dataset_loader(
            name=self.config.data.dataset,
            batch_size=1,
            num_frames=196,
            split='test',
            hml_mode='text_only'
        )
        self.data.fixed_length = float(self.num_frames)

        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(self.config, self.data)

        print(f"Loading checkpoints from {self.config.sampling.model_path}...")
        state_dict = torch.load(self.config.sampling.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)

        if self.config.sampling.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)
        
        self.model.to(dist_util.dev())
        self.model.eval()

    def generate(
        self,
        prompt: str,
        num_repetitions: int = 3,
        output_format: str = "animation"  # or "json_file"
    ) -> Dict:
        """
        Generate motion from text prompt.
        
        Args:
            prompt: Text description of the motion
            num_repetitions: Number of motion samples to generate
            output_format: Either "animation" or "json_file"
        
        Returns:
            dict with either 'animation' (list of paths) or 'json_file' (motion data dict)
        """
        config = self.config
        
        self.data = get_dataset_loader(
            name=self.config.data.dataset,
            batch_size=num_repetitions,
            num_frames=self.num_frames,
            split='test',
            hml_mode='text_only'
        )

        collate_args = [{
            'inp': torch.zeros(self.num_frames),
            'tokens': None,
            'lengths': self.num_frames,
            'text': str(prompt)
        }]
        _, model_kwargs = collate(collate_args)

        # add CFG scale to batch
        if config.sampling.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(
                num_repetitions, 
                device=dist_util.dev()
            ) * config.sampling.guidance_param

        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.model,
            (num_repetitions, self.model.njoints, self.model.nfeats, self.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if self.model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = self.data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(
            num_repetitions, self.num_frames
        ).bool()
        
        sample = self.model.rot2xyz(
            x=sample,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype='smpl',
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False
        )

        all_motions = sample.cpu().numpy()

        if output_format == 'json_file':
            data_dict = motions2hik(all_motions)
            return {'json_file': data_dict}

        caption = str(prompt)
        skeleton = paramUtil.t2m_kinematic_chain

        sample_print_template, row_print_template, all_print_template, \
            sample_file_template, row_file_template, all_file_template = construct_template_variables(
            config.training.unconstrained
        )

        rep_files = []
        for rep_i in range(num_repetitions):
            motion = all_motions[rep_i].transpose(2, 0, 1)[:self.num_frames]
            save_file = sample_file_template.format(1, rep_i)
            print(sample_print_template.format(caption, 1, rep_i, save_file))
            plot_3d_motion(
                save_file,
                skeleton,
                motion,
                dataset=config.data.dataset,
                title=caption,
                fps=config.sampling.fps
            )
            rep_files.append(Path(save_file))

        return {'animation': rep_files}


# Example usage
if __name__ == "__main__":
    from hydra import compose, initialize
    
    # Load config
    initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="config")
    
    # Create generator
    generator = MotionGenerator(cfg)
    
    # Generate motion
    result = generator.generate(
        prompt="the person walked forward and is picking up his toolbox.",
        num_repetitions=3,
        output_format="animation"
    )
    
    print(f"Generated {len(result['animation'])} animations")
    for anim_path in result['animation']:
        print(f"  - {anim_path}")