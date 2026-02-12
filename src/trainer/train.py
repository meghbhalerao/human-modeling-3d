import os
import json
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Import your existing utilities
from utils.fixseed import fixseed
from utils import dist_utils
from trainer.training_loop import TrainLoop
from data.get_data import get_dataset_loader
from utils.model_utils import create_model_and_diffusion
from trainer.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup System
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    # Accessing hierarchical parameter: cfg.experiment.seed
    fixseed(cfg.experiment.seed) 
    
    # 2. Setup Platform
    train_platform_type = eval(cfg.experiment.train_platform_type) 
    train_platform = train_platform_type(cfg.experiment.save_dir)
    
    # Convert OmegaConf to dict for reporting/logging
    train_platform.report_args(OmegaConf.to_container(cfg, resolve=True), name='Args')

    # 3. Directory Checks
    save_dir = cfg.experiment.save_dir
    if save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(save_dir) and not cfg.experiment.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(save_dir))
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save the full hierarchical config
    args_path = os.path.join(save_dir, 'args.yaml')
    with open(args_path, 'w') as fw:
        OmegaConf.save(cfg, fw)

    # 4. Setup Distributed Training
    dist_utils.setup_dist(cfg.training.device)

    print("creating data loader...")
    data = get_dataset_loader(
        name=cfg.data.dataset, 
        batch_size=cfg.data.batch_size, 
        num_frames=cfg.data.num_frames, 
        fixed_len=cfg.data.pred_len + cfg.data.context_len, 
        pred_len=cfg.data.pred_len,
        device=dist_utils.dev()
    )

    print("creating model and diffusion...")
    # Passing the hierarchical 'cfg' object. 
    # You will need to update create_model_and_diffusion to handle cfg.model.target, etc.
    model, diffusion = create_model_and_diffusion(cfg, data)
    model.to(dist_utils.dev())
    
    if hasattr(model, 'rot2xyz') and hasattr(model.rot2xyz, 'smpl_model'):
        model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    
    # Passing 'cfg' to TrainLoop
    TrainLoop(cfg, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()