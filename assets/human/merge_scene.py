
"""
merge_scene.py

Merges human.xml into a single self-contained scene.xml.
Run from anywhere, just set ASSETS_DIR below to your assets/human folder.
"""

import re
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
ASSETS_DIR  = os.path.expanduser("./")
HUMAN_XML   = os.path.join(ASSETS_DIR, "human.xml")
SCENE_XML   = os.path.join(ASSETS_DIR, "scene.xml")
# ────────────────────────────────────────────────────────────────────────────


def extract_block(content, tag):
    """Extract the inner contents of the first <tag>...</tag> block."""
    pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find <{tag}> block in human.xml")
    return match.group(1).strip()


def main():
    print(f"Reading {HUMAN_XML} ...")
    with open(HUMAN_XML, 'r') as f:
        human = f.read()

    # ── Extract <default> block ──────────────────────────────────────────────
    default_contents = extract_block(human, 'default')
    print("  ✓ Extracted <default>")

    # ── Extract <asset> block ────────────────────────────────────────────────
    asset_contents = extract_block(human, 'asset')
    cleaned_lines = []
    for line in asset_contents.splitlines():
        stripped = line.strip()
        if stripped.startswith('<texture') and 'file=' in stripped:
            continue  # skip texture file references
        if stripped.startswith('<material') and 'texture=' in stripped:
            continue  # skip materials that reference textures
        cleaned_lines.append(line)
    asset_contents = '\n'.join(cleaned_lines)
    print("  ✓ Extracted <asset> and removed texture references")
    # import re
    # asset_contents = re.sub(r'<texture[^/]*/>', '', asset_contents)
    # asset_contents = re.sub(r'<material[^/]*/>', '', asset_contents)
    # print("  ✓ Extracted <asset>")

    # ── Extract <worldbody> block ────────────────────────────────────────────
    worldbody_contents = extract_block(human, 'worldbody')

    # Strip the outermost anonymous <body> wrapper (has no name attribute)
    # It looks like:  <body>\n    <body name="object" mocap="true"> ...
    # We want everything INSIDE it, not the wrapper itself.
    worldbody_contents = re.sub(r'^\s*<body>\s*', '', worldbody_contents)
    worldbody_contents = worldbody_contents.replace(
    '<body name="object" mocap="true">',
    '<body name="object" mocap="true" pos="0 0 0">')
    worldbody_contents = re.sub(r'\s*</body>\s*$', '', worldbody_contents)
    worldbody_contents = re.sub(r'\s*material="[^"]*"', '', worldbody_contents)
    # ← ADD THIS HERE
    # Make all site markers transparent (removes red blobs)
    worldbody_contents = re.sub(
        r'(<site[^>]*?)rgba="1 0 0 0\.5"',
        r'\1rgba="0 0 0 0"',
        worldbody_contents
    )

    worldbody_contents = worldbody_contents.strip()
    print("  ✓ Extracted <worldbody> and stripped anonymous wrapper")

    # ── Build merged scene.xml ───────────────────────────────────────────────
    scene = f"""<mujoco model="scene">

    <size njmax="8000" nconmax="4000"/>

    <compiler angle="degree" inertiafromgeom="true" coordinate="global"/>
    <!--
        angle="degree"     : human.xml uses degrees for joint ranges
        coordinate="global": joint positions are in world space
        inertiafromgeom    : inertia computed from geometry (needed for physics objects)
    -->

    <visual>
        <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
        <map znear=".01"/>
        <quality shadowsize="2048"/>
    </visual>

    <option timestep="0.0333" gravity="0 -9.81 0"/>
    <!--
        timestep = 1/30 = 30fps
        gravity is Y-up so gravity points in -Y direction
    -->

    <statistic extent="3" center="0 0 1"/>

    <!-- ===== DEFAULT JOINT/GEOM PROPERTIES (from human.xml) ===== -->
    <default>
        {default_contents}
    </default>

    <!-- ===== ASSETS: meshes + textures + materials ===== -->
    <asset>
        <!-- Floor texture -->
        <texture builtin="checker" height="100" name="texplane"
                 rgb1="0.15 0.15 0.15" rgb2="0.7 0.7 0.7"
                 type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.3" shininess="0.5"
                  specular="0.5" texrepeat="4 4" texture="texplane"/>

        <!-- Skybox -->
        <texture type="skybox" builtin="gradient"
                 rgb1=".4 .5 .6" rgb2="0 0 0"
                 width="100" height="100"/>

        <!-- Human meshes and materials (from human.xml) -->
        {asset_contents}
    </asset>

    <worldbody>

        <!-- ===== LIGHTING ===== -->
        <light name="main_light"
               pos="0 2 2" dir="0 -1 -1"
               diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
               directional="true" castshadow="true"/>

        <light name="fill_light"
               pos="0 2 -2" dir="0 -1 1"
               diffuse="0.3 0.3 0.3" specular="0.0 0.0 0.0"
               directional="true" castshadow="false"/>

        <!-- ===== FLOOR ===== -->
        <!--
            pos="0 -1.2 0": Y=-1.2 places floor just below the feet.
            In human.xml the feet (L_Toe) are at Y ~ -1.14, so -1.2 gives
            a small gap. Tweak this value if the human floats or clips.
        -->
        <geom name="floor" type="plane"
              pos="0 -1.2 0" size="5 5 0.1"
              material="MatPlane"
              conaffinity="1" condim="3" contype="1"/>

        <!-- ===== CAMERAS ===== -->
        <!--
            side_cam: view from the right, good for walking motions
            front_cam: view from front
            Switch between them in the viewer with the [ ] keys,
            or specify which to use in offscreen rendering in replay.py
        -->
        <camera name="side_cam"  pos="3 0 0"  xyaxes="0 0 -1 0 1 0"/>
        <camera name="front_cam" pos="0 0 3"  xyaxes="1 0 0 0 1 0"/>

        <!-- ===== HUMAN (merged from human.xml) ===== -->
        {worldbody_contents}

    </worldbody>

</mujoco>
"""

    with open(SCENE_XML, 'w') as f:
        f.write(scene)

    print(f"\n✓ Written to {SCENE_XML}")
    print("\nNext step — test it:")
    print(f"  ~/.mujoco/mujoco210/bin/simulate {SCENE_XML}")


if __name__ == "__main__":
    main()
