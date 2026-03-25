"""
generate_stick_figure.py

Generates a MuJoCo XML with:
  - 22 mocap sphere bodies (one per SMPL joint)
  - 21 capsule bodies (one per bone, connecting parent to child joint)

All bodies are mocap=true — no physics, positions set directly from data.
The capsule fromto values are updated every frame in replay.py.

Run this once to generate the XML, then use replay.py to animate it.
"""

import os

ASSETS_DIR = os.path.expanduser("~/projects/human-modeling-3d/assets/human")
OUTPUT_XML = os.path.join(ASSETS_DIR, "stick_figure.xml")

# ── Joint definitions ────────────────────────────────────────────────────────
# (index, name, initial T-pose position in Y-up space from human.xml)
JOINTS = [
    ( 0, 'pelvis',          -0.0018, -0.2233,  0.0282),
    ( 1, 'left_hip',         0.0677, -0.3147,  0.0214),
    ( 2, 'right_hip',       -0.0695, -0.3139,  0.0239),
    ( 3, 'spine1',          -0.0043, -0.1144,  0.0015),
    ( 4, 'left_knee',        0.1020, -0.6899,  0.0169),
    ( 5, 'right_knee',      -0.1078, -0.6964,  0.0150),
    ( 6, 'spine2',           0.0012,  0.0208,  0.0026),
    ( 7, 'left_ankle',       0.0884, -1.0879, -0.0268),
    ( 8, 'right_ankle',     -0.0920, -1.0948, -0.0273),
    ( 9, 'spine3',           0.0026,  0.0737,  0.0280),
    (10, 'left_foot',        0.1148, -1.1437,  0.0925),
    (11, 'right_foot',      -0.1174, -1.1430,  0.0961),
    (12, 'neck',            -0.0002,  0.2876, -0.0148),
    (13, 'left_collar',      0.0815,  0.1955, -0.0060),
    (14, 'right_collar',    -0.0791,  0.1926, -0.0106),
    (15, 'head',             0.0050,  0.3526,  0.0365),
    (16, 'left_shoulder',    0.1724,  0.2260, -0.0149),
    (17, 'right_shoulder',  -0.1752,  0.2251, -0.0197),
    (18, 'left_elbow',       0.4320,  0.2132, -0.0424),
    (19, 'right_elbow',     -0.4289,  0.2118, -0.0411),
    (20, 'left_wrist',       0.6813,  0.2222, -0.0435),
    (21, 'right_wrist',     -0.6842,  0.2196, -0.0467),
]

# ── Parent array (SMPL convention) ───────────────────────────────────────────
# PARENTS[i] = parent joint index of joint i (-1 = root)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# ── Visual parameters ────────────────────────────────────────────────────────
JOINT_RADIUS = 0.025   # sphere radius for joints (meters)
ROOT_RADIUS  = 0.040   # slightly larger sphere for pelvis (easy to identify)
BONE_RADIUS  = 0.012   # capsule radius for bones

# Colors (RGBA)
JOINT_COLOR  = "0.2 0.6 1.0 1.0"   # blue spheres for joints
ROOT_COLOR   = "1.0 0.5 0.0 1.0"   # orange sphere for pelvis
BONE_COLOR   = "0.8 0.8 0.8 0.9"   # light grey capsules for bones


def yup_to_zup(x, y, z):
    """
    Convert Y-up coordinates to Z-up coordinates.
    Y-up: x=right, y=up,   z=forward
    Z-up: x=right, y=forward, z=up
    Swap: new_x=x, new_y=z, new_z=y
    """
    return x, z, y


def main():
    # ── Build joint sphere bodies ────────────────────────────────────────────
    joint_bodies = []
    for idx, name, x, y, z in JOINTS:
        # Convert T-pose positions from Y-up to Z-up for initial placement
        zx, zy, zz = yup_to_zup(x, y, z)

        radius = ROOT_RADIUS if idx == 0 else JOINT_RADIUS
        color  = ROOT_COLOR  if idx == 0 else JOINT_COLOR

        joint_bodies.append(f"""
        <!-- Joint {idx}: {name} -->
        <body name="joint_{name}" mocap="true" pos="{zx:.4f} {zy:.4f} {zz:.4f}">
            <geom type="sphere" size="{radius}"
                  rgba="{color}"
                  contype="0" conaffinity="0"/>
        </body>""")

    # ── Build bone capsule bodies ────────────────────────────────────────────
    # One capsule per parent-child joint pair
    # fromto connects parent position to child position in Z-up space
    bone_bodies = []
    for child_idx, name, cx, cy, cz in JOINTS:
        parent_idx = PARENTS[child_idx]
        if parent_idx == -1:
            continue  # root has no parent, no bone to draw

        # Get parent position
        _, pname, px, py, pz = JOINTS[parent_idx]

        # Convert both endpoints to Z-up
        px_z, py_z, pz_z = yup_to_zup(px, py, pz)
        cx_z, cy_z, cz_z = yup_to_zup(cx, cy, cz)

        bone_bodies.append(f"""
        <!-- Bone: {pname} -> {name} -->
        <body name="bone_{pname}_{name}" pos="0 0 0">
            <geom name="bone_{parent_idx}_{child_idx}"
                  type="capsule" size="{BONE_RADIUS}"
                  fromto="{px_z:.4f} {py_z:.4f} {pz_z:.4f}  {cx_z:.4f} {cy_z:.4f} {cz_z:.4f}"
                  rgba="{BONE_COLOR}"
                  contype="0" conaffinity="0"/>
        </body>""")

    joints_xml = '\n'.join(joint_bodies)
    bones_xml  = '\n'.join(bone_bodies)

    xml = f"""<mujoco model="stick_figure">

    <size njmax="1000" nconmax="500"/>

    <compiler angle="degree" inertiafromgeom="true" coordinate="global"/>
    <!--
        coordinate="global": all positions in world space.
        This is important — mocap body positions are absolute world coords.
    -->

    <visual>
        <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
        <map znear=".01"/>
        <quality shadowsize="2048"/>
    </visual>

    <option timestep="0.0333" gravity="0 0 -9.81"/>
    <!--
        Z-up coordinate system.
        Gravity along -Z.
        Gravity has no effect on mocap bodies but matters for
        any physics objects we add later (table, chair).
    -->

    <statistic extent="3" center="0 0 1"/>

    <asset>
        <texture builtin="checker" height="100" name="texplane"
                 rgb1="0.15 0.15 0.15" rgb2="0.6 0.6 0.6"
                 type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.3" shininess="0.5"
                  specular="0.5" texrepeat="4 4" texture="texplane"/>
        <texture type="skybox" builtin="gradient"
                 rgb1=".4 .5 .6" rgb2="0 0 0"
                 width="100" height="100"/>
    </asset>

    <worldbody>

        <!-- ===== LIGHTING ===== -->
        <light name="main_light"
               pos="0 -2 4" dir="0 1 -2"
               diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
               directional="true" castshadow="true"/>
        <light name="fill_light"
               pos="0 2 3" dir="0 -1 -1"
               diffuse="0.3 0.3 0.3" specular="0.0 0.0 0.0"
               directional="true" castshadow="false"/>

        <!-- ===== FLOOR ===== -->
        <!--
            Z-up, floor at Z=0.
            Feet in the T-pose data are at Y ~ -1.14 (Y-up)
            which becomes Z ~ -1.14 after conversion.
            We offset the floor slightly below that.
        -->
        <geom name="floor" type="plane"
              pos="0 0 -1.20" size="5 5 0.1"
              material="MatPlane"
              conaffinity="1" condim="3" contype="1"/>

        <!-- ===== CAMERAS ===== -->
        <!--
            side_cam:  view from the side (+X), good for walking
            front_cam: view from the front (+Y)
            top_cam:   bird's eye view from above (+Z)
        -->
        <camera name="side_cam"
                pos="4 0 1"
                xyaxes="0 1 0 0 0 1"/>
        <camera name="front_cam"
                pos="0 -4 1"
                xyaxes="1 0 0 0 0 1"/>
        <camera name="top_cam"
                pos="0 0 5"
                xyaxes="1 0 0 0 1 0"/>

        <!-- ===== JOINT SPHERES (22 mocap bodies) ===== -->
        <!--
            Each body is mocap=true.
            Positions are set directly from motion data in replay.py.
            Initial pos here is just the T-pose — overwritten on frame 1.
            Orange sphere = pelvis (root), blue spheres = all other joints.
        -->
{joints_xml}

        <!-- ===== BONE CAPSULES (21 bodies) ===== -->
        <!--
            One capsule per parent->child bone.
            These are NOT mocap bodies — they are static geom bodies
            whose fromto attribute is updated in Python each frame
            using sim.model.geom_fromto.
            pos="0 0 0" because fromto uses absolute world coordinates.
        -->
{bones_xml}

    </worldbody>

</mujoco>
"""

    with open(OUTPUT_XML, 'w') as f:
        f.write(xml)

    print(f"✓ Written to {OUTPUT_XML}")
    print(f"\nJoints: {len(JOINTS)}")
    print(f"Bones:  {len(JOINTS) - 1}")
    print(f"\nTest static T-pose with:")
    print(f"  ~/.mujoco/mujoco210/bin/simulate {OUTPUT_XML}")
    print(f"\nThen run motion replay with:")
    print(f"  python src/replay.py")


if __name__ == "__main__":
    main()