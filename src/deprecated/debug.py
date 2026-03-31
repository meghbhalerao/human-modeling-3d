import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import sys

# --- CONFIG ---
# Update this to your path
INPUT_PATH = "/home/mb230/projects/human-modeling-3d/data/TOY/processed/processed_part008.json"
OUTPUT_IMG = "sanity_check.png"

# SMPL Chains
CHAINS = [
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10]
]

def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    frames = data.get('frames', [])

    # 1. Find the FIRST VALID FRAME
    valid_idx = -1
    raw_pose = None
    
    for i, f in enumerate(frames):
        if f.get('smpl_body') and f['smpl_body'].get('positions'):
            p = np.array(f['smpl_body']['positions'])
            if np.max(np.abs(p)) > 0.01: # Check if not empty
                valid_idx = i
                raw_pose = p
                # Pick a frame in the middle of the action if possible
                if i > len(frames) // 3: 
                    break
    
    if raw_pose is None:
        print("CRITICAL ERROR: No valid skeleton found in any frame!")
        return

    print(f"Found valid skeleton at Frame {valid_idx}")
    
    # 2. Coordinate Transform (X->X, Z->Y, -Y->Z)
    pose = np.zeros_like(raw_pose)
    pose[:, 0] = raw_pose[:, 0]
    pose[:, 1] = raw_pose[:, 2]
    pose[:, 2] = -raw_pose[:, 1]

    # 3. CRITICAL FIX: Per-Frame Centering
    # Center this specific frame on its own Hips (Joint 0)
    # This guarantees the skeleton is at (0,0,0)
    pose -= pose[0]

    # Print bounds to verify centering
    print("Centered Bounds:")
    print(f"  X: {pose[:,0].min():.2f} to {pose[:,0].max():.2f}")
    print(f"  Y: {pose[:,1].min():.2f} to {pose[:,1].max():.2f}")
    print(f"  Z: {pose[:,2].min():.2f} to {pose[:,2].max():.2f}")

    # 4. Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Force Auto-Scale first to see if that works
    # ax.autoscale(True) 
    
    # Or force known good limits for Normalized data
    radius = 0.8
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw
    for chain in CHAINS:
        ax.plot(pose[chain, 0], pose[chain, 1], pose[chain, 2], 
                color='red', linewidth=3, marker='o', markersize=4)
    
    # Draw Origin
    ax.scatter([0], [0], [0], color='blue', s=100, label='Origin')

    print(f"Saving to {OUTPUT_IMG}...")
    plt.savefig(OUTPUT_IMG)
    print("Done. Open the image!")

if __name__ == "__main__":
    main()