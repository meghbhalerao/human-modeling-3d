"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import torch
import re
from utils import dist_utils
from utils.sampler_utils import ClassifierFreeSampleModel
from data.get_data import get_dataset_loader
from eval.a2m.tools import save_metrics
from utils.fixseed import fixseed
from utils.model_utils import create_model_and_diffusion, load_model_wo_clip
import hydra
from omegaconf import DictConfig


def evaluate(cfg, model, diffusion, data):
    scale = None
    if cfg.evaluation.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(cfg.data.batch_size) * cfg.evaluation.guidance_param,
        }
    
    model.to(dist_utils.dev())
    model.eval()  # disable random masking
    
    folder, ckpt_name = os.path.split(cfg.evaluation.model_path)
    
    if cfg.data.dataset == "humanact12":
        from eval.a2m.gru_eval import evaluate as eval_fn
        eval_results = eval_fn(cfg, model, diffusion, data)
    elif cfg.data.dataset == "uestc":
        from eval.a2m.stgcn_eval import evaluate as eval_fn
        eval_results = eval_fn(cfg, model, diffusion, data)
    else:
        raise NotImplementedError("This dataset is not supported.")
    
    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale_val = 1 if scale is None else scale['action'][0].item()
    scale_str = str(scale_val).replace('.', 'p')
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(
        iter, cfg.evaluation.num_samples, scale_str
    )
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)
    
    return eval_results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    fixseed(cfg.experiment.seed)
    dist_utils.setup_dist(cfg.training.device)
    
    print(f'Eval mode [{cfg.evaluation.eval_mode}]')
    assert cfg.evaluation.eval_mode in ['debug', 'full'], \
        f'eval_mode {cfg.evaluation.eval_mode} is not supported for dataset {cfg.data.dataset}'
    
    if cfg.evaluation.eval_mode == 'debug':
        num_samples = 10
        num_seeds = 2
    else:
        num_samples = 1000
        num_seeds = 20
    
    # Override the config values with computed ones
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.num_seeds = num_seeds
    
    data_loader = get_dataset_loader(
        name=cfg.data.dataset, 
        num_frames=60, 
        batch_size=cfg.data.batch_size,
    )
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(cfg, data_loader)
    
    print(f"Loading checkpoints from [{cfg.evaluation.model_path}]...")
    state_dict = torch.load(cfg.evaluation.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    
    eval_results = evaluate(cfg, model, diffusion, data_loader.dataset)
    
    fid_to_print = {
        k: sum([float(vv) for vv in v])/len(v) 
        for k, v in eval_results['feats'].items() 
        if 'fid' in k and 'gen' in k
    }
    print(fid_to_print)


if __name__ == '__main__':
    main()