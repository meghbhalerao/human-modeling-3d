import hydra
from omegaconf import DictConfig
from trainer.sampler import MotionSampler

@hydra.main(config_path="../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig):
    # You can override config params via command line, e.g.:
    # python run_sample.py sampling.num_samples=10 sampling.checkpoint_path=checkpoints/epoch_90.pt
    
    sampler = MotionSampler(cfg)
    sampler.sample()

if __name__ == "__main__":
    main()