import torch
import hydra
import wandb
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from models.human.transformer import MotionDiT
from models.human.gaussian_diffusion import GaussianDiffusion, DiffusionConfig, get_named_beta_schedule
from src.data.datasets import get_motion_dataloader

def train(cfg: DictConfig):
    wandb.init(project=cfg.training.wandb_project, config=cfg)
    
    # Setup Input Dimension based on parts
    # body=144, each hand=96, root=3
    input_dim = 0
    if 'body' in cfg.data.parts: input_dim += 144
    if 'left' in cfg.data.parts: input_dim += 96
    if 'right' in cfg.data.parts: input_dim += 96
    input_dim += 3 # Root translation

    model = MotionDiT(input_dim, cfg.model.latent_dim, cfg.model.n_heads, cfg.model.n_layers, cfg.model.dropout).to(cfg.training.device)
    
    # Initialize Diffusion (using your uploaded gaussian_diffusion logic)
    betas = get_named_beta_schedule(cfg.diffusion.noise_schedule, cfg.diffusion.steps)
    diffconfig = DiffusionConfig(betas = betas) 
    diffusion = GaussianDiffusion(diffconfig) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    json_paths = [str(f.resolve()) for f in Path(cfg.data.json_dir).iterdir() if f.is_file()]

    dataloader = get_motion_dataloader(cfg, json_paths)

    for epoch in range(100):
        print(f"Epoch {epoch+1}/100")
        for i, batch in enumerate(dataloader):
            batch = batch.to(cfg.training.device)
            t = torch.randint(0, cfg.diffusion.steps, (batch.shape[0],), device=batch.device)
            
            # Forward Diffusion
            noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise=noise)
            
            # Predict noise
            pred_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.training.log_interval == 0:
                wandb.log({"loss": loss.item(), "epoch": epoch})

@hydra.main(config_path="../configs", config_name="dit_train")
def main(cfg):

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.training.wandb_project, config=wandb_config
    )

    train(cfg)

if __name__ == "__main__":
    main()
