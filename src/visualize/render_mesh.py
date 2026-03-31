import os
import sys
from visualize import vis_utils
import shutil
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
import imageio

@hydra.main(version_base=None, config_path="../configs", config_name="render_mesh")
def main(cfg: DictConfig):
    assert cfg.input_path.endswith('.mp4'), "input_path must be an .mp4 file"
    
    parsed_name = os.path.basename(cfg.input_path).replace('.mp4', '').replace('samples_', '').replace('rep', '').replace('rows_', '').replace('to_', '')

    sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    
    npy_path = os.path.join(os.path.dirname(cfg.input_path), 'results.npy')
    out_npy_path = cfg.input_path.replace('.mp4', '_smpl_params.npy')
    assert os.path.exists(npy_path), f"results.npy not found at {npy_path}"
    
    results_dir = cfg.input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    
    npy2obj = vis_utils.npy2obj(
        cfg, 
        npy_path, 
        sample_i, 
        rep_i,
        device=cfg.device_id, 
        cuda= True if cfg.device == 'cuda' else False
    )   
    
    if cfg.get('replay_mujoco', False):
        npy2obj.run_mujoco_vis()
        sys.exit()

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
    
    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)

    if cfg.get('save_video', False):
        import pyrender
        import trimesh
        
        data = np.load(out_npy_path, allow_pickle=True).item()
        vertices = data['vertices']  # [nframes, 6890, 3]
        faces = data['faces']

        video_path = cfg.input_path.replace('.mp4', '_smpl.mp4')
        print('Saving video to [{}]'.format(os.path.abspath(video_path)))

        print(f"Shape of vertices is {vertices.shape}")
        print(f"Shape of faces is {faces.shape}")

        images = []

        for frame_i in tqdm(range(npy2obj.real_num_frames)):
            mesh = trimesh.Trimesh(vertices=vertices[:,:,frame_i], faces=faces)
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(mesh))
            
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = np.eye(4)
            camera_pose[2, 3] = 6.0
            # print("Camera pose is \n", camera_pose)
            scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)
            
            r = pyrender.OffscreenRenderer(640, 480)
            color, _ = r.render(scene)
            images.append(color)
            r.delete()
        
        imageio.mimsave(video_path, images, fps=cfg.get('fps', 20))



if __name__ == '__main__':
    main()