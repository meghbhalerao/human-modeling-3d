import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="render_mesh")
def main(cfg: DictConfig):
    assert cfg.input_path.endswith('.mp4'), "input_path must be an .mp4 file"
    
    parsed_name = os.path.basename(cfg.input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
    sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    
    npy_path = os.path.join(os.path.dirname(cfg.input_path), 'results.npy')
    out_npy_path = cfg.input_path.replace('.mp4', '_smpl_params.npy')
    assert os.path.exists(npy_path), f"results.npy not found at {npy_path}"
    
    results_dir = cfg.input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    
    npy2obj = vis_utils.npy2obj(
        npy_path, 
        sample_i, 
        rep_i,
        device=cfg.device, 
        cuda=cfg.cuda
    )
    
    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
    
    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)


if __name__ == '__main__':
    main()