import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys
from pathlib import Path

class SkeletonPlayer:
    """
    A 3D skeleton renderer for SMPL and MANO data.
    """
    # Kinematic trees: defines which joint connects to which parent
    SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    MANO_PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.data = self._load_data()
        self.frames = self.data.get('frames', [])
        
    def _load_data(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"Could not find JSON at {self.json_path}")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _draw_bones(self, ax, joints, parents, color):
        """Helper to draw line segments between parent and child joints."""
        if joints is None:
            return
        
        pts = np.array(joints)
        for i, p in enumerate(parents):
            if p != -1 and i < len(pts):
                # ax.plot(x_values, y_values, z_values)
                ax.plot(
                    [pts[p, 0], pts[i, 0]],
                    [pts[p, 1], pts[i, 1]],
                    [pts[p, 2], pts[i, 2]],
                    color=color, linewidth=2, alpha=0.8
                )
        # Add joints as dots
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=15)

    def render(self, output_name, components=['body', 'left', 'right'], fps=15):
        # Validate inputs
        valid_parts = {'body', 'left', 'right'}
        for c in components:
            if c not in valid_parts:
                print(f"Error: '{c}' is not a valid component. Use 'body', 'left', or 'right'.")
                return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Diagonal view: elevate 20 degrees, rotate -45 degrees
        ax.view_init(elev=20, azim=-45)

        def update(i):
            ax.clear()
            # Set consistent bounds so the "camera" doesn't jitter
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(-0.5, 0.5)
            
            # Labeling
            ax.set_title(f"Frame {i+1}/{len(self.frames)}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            frame = self.frames[i]

            # Render requested parts if they exist in the JSON
            if 'body' in components and frame.get('smpl_body'):
                self._draw_bones(ax, frame['smpl_body'].get('positions'), self.SMPL_PARENT, 'forestgreen')
            
            if 'left' in components and frame.get('mano_left'):
                self._draw_bones(ax, frame['mano_left'].get('positions'), self.MANO_PARENT, 'crimson')
                
            if 'right' in components and frame.get('mano_right'):
                self._draw_bones(ax, frame['mano_right'].get('positions'), self.MANO_PARENT, 'royalblue')

        ani = FuncAnimation(fig, update, frames=len(self.frames), interval=1000/fps)
        
        print(f"Encoding video: {output_name}...")
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        ani.save(output_name, writer=writer)
        plt.close()
        print("Done.")


def main():
    # Path to your JSON output from the extractor
    input_json = "/home/mb230/projects/human-modeling-3d/data/TOY/processed/20260206_114125_smpl.json" 
    
    # Initialize the player
    player = SkeletonPlayer(input_json)
    
    # Specify what you want to see
    # Try: ['body'], ['left', 'right'], or ['body', 'left', 'right']
    targets = ['body', 'left', 'right']
    
    output_file = "full_skeleton_viz.mp4"
    
    # Run the visualization
    player.render(
        output_name=output_file, 
        components=targets, 
        fps=5
    )

if __name__ == "__main__":
    main()