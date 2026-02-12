import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import os
import shutil
import cv2
import hydra
from omegaconf import DictConfig

class SkeletonPlayer:
    # SMPL Chains
    SMPL_CHAINS = [
        [0, 3, 6, 9, 12, 15],       # Torso
        [9, 14, 17, 19, 21],        # R Arm
        [9, 13, 16, 18, 20],        # L Arm
        [0, 2, 5, 8, 11],           # R Leg
        [0, 1, 4, 7, 10]            # L Leg
    ]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print(f"Loading: {cfg.input_path}")
        with open(cfg.input_path, 'r') as f:
            self.data = json.load(f)
        self.frames = self.data.get('frames', [])

    def render(self):
        # 1. Output Setup
        output_dir = "final_frames"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 2. Filter & Process Data
        # We only want frames that have data
        valid_frames = []
        for i, f in enumerate(self.frames):
            if f.get('smpl_body') and f['smpl_body'].get('positions'):
                p = np.array(f['smpl_body']['positions'])
                # Filter out empty/zero frames
                if np.max(np.abs(p)) > 0.01:
                    valid_frames.append(p)
        
        if not valid_frames:
            print("ERROR: No valid frames found.")
            return

        print(f"Found {len(valid_frames)} valid frames.")
        
        # 3. RENDER LOOP
        # We process one frame at a time, creating a NEW figure each time.
        for idx, raw_pose in enumerate(valid_frames):
            
            # --- TRANSFORM (Same as Sanity Check) ---
            pose = np.zeros_like(raw_pose)
            pose[:, 0] = raw_pose[:, 0]
            pose[:, 1] = raw_pose[:, 2]
            pose[:, 2] = -raw_pose[:, 1]
            
            # --- CENTERING (Same as Sanity Check) ---
            pose -= pose[0] # Center Hips at (0,0,0)

            # --- PLOT (Same as Sanity Check) ---
            # Use 'default' style (White background) because we know it works
            plt.style.use('default') 
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Force Limits (Normalized Data)
            radius = 0.8
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(-radius, radius)
            ax.view_init(elev=20, azim=-60)
            ax.set_axis_off()

            # Draw Skeleton (Red Lines)
            for chain in self.SMPL_CHAINS:
                ax.plot(pose[chain, 0], pose[chain, 1], pose[chain, 2], 
                        color='red', linewidth=3, solid_capstyle='round')
            
            # Draw Joints (Black Dots)
            ax.scatter(pose[:,0], pose[:,1], pose[:,2], s=20, color='black')

            # Save
            filename = os.path.join(output_dir, f"frame_{idx:04d}.png")
            plt.savefig(filename, dpi=80)
            
            # CRITICAL: Close the figure explicitly to free memory
            plt.close(fig)

            if idx % 20 == 0:
                print(f"Saved {filename} ...")

        # 4. STITCH VIDEO
        print("Stitching video...")
        self.make_video(output_dir, "bulletproof_viz.mp4")

    def make_video(self, input_dir, output_file):
        images = sorted([img for img in os.listdir(input_dir) if img.endswith(".png")])
        if not images: return
        
        # Read first frame to get size
        frame = cv2.imread(os.path.join(input_dir, images[0]))
        h, w, l = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, 20, (w, h))

        for img in images:
            video.write(cv2.imread(os.path.join(input_dir, img)))

        video.release()
        print(f"SUCCESS! Video saved to {output_file}")

@hydra.main(version_base=None, config_path="../configs", config_name="viz")
def main(cfg: DictConfig):
    player = SkeletonPlayer(cfg)
    player.render()

if __name__ == "__main__":
    main()