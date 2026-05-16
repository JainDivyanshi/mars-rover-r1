"""
rover_urdf_generator.py
────────────────────────
Generates a 6-wheeled rocker-bogie URDF rover programmatically.
No external files needed — the URDF is created at runtime and saved
to a temp file that PyBullet can load.

Rover layout (top view):

        FL ──── FR
        |        |
   bogie-L      bogie-R     ← passive pivot joints
        |        |
        ML ──── MR
        |        |
  rocker-L     rocker-R     ← passive pivot joints
        |        |
        RL ──── RR

Body sits above the suspension.  Driven joints: all 6 wheel axles.
Passive joints: 2 rocker pivots + 2 bogie pivots (no motors, free to rotate).
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ── Physical constants (metres, kg) ──────────────────────────────────────────
BODY_W   = 0.50   # body width  (x)
BODY_L   = 0.30   # body length (y)  — depth front-to-back
BODY_H   = 0.12   # body height (z)
BODY_MASS = 3.0

WHEEL_R  = 0.07   # wheel radius
WHEEL_W  = 0.05   # wheel width (cylinder length)
WHEEL_MASS = 0.4

# Suspension link dimensions (thin rods)
LINK_R   = 0.015  # link cylinder radius
LINK_MASS = 0.15

# Wheel positions relative to body centre (x = lateral, y = front/back, z = drop)
WHEEL_X  = BODY_W / 2 + WHEEL_W / 2 + 0.01   # outboard of body
FRONT_Y  =  0.18   # front wheels
MID_Y    =  0.00   # middle wheels
REAR_Y   = -0.18   # rear wheels
WHEEL_Z  = -(BODY_H / 2 + WHEEL_R + 0.02)     # hang below body


def _pretty(element: ET.Element) -> str:
    """Return indented XML string."""
    raw = ET.tostring(element, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _inertia_box(m, x, y, z) -> dict:
    """Inertia tensor for a solid box (kg·m²)."""
    return {
        "ixx": str(round(m / 12 * (y**2 + z**2), 6)),
        "iyy": str(round(m / 12 * (x**2 + z**2), 6)),
        "izz": str(round(m / 12 * (x**2 + y**2), 6)),
        "ixy": "0", "ixz": "0", "iyz": "0",
    }


def _inertia_cylinder(m, r, h) -> dict:
    """Inertia tensor for a solid cylinder (axis = z, kg·m²)."""
    return {
        "ixx": str(round(m / 12 * (3 * r**2 + h**2), 6)),
        "iyy": str(round(m / 12 * (3 * r**2 + h**2), 6)),
        "izz": str(round(m / 2  * r**2,              6)),
        "ixy": "0", "ixz": "0", "iyz": "0",
    }


def _xyz(x, y, z) -> str:
    return f"{round(x,4)} {round(y,4)} {round(z,4)}"


def _rpy(r, p, y) -> str:
    return f"{round(r,4)} {round(p,4)} {round(y,4)}"


# ── Link builders ─────────────────────────────────────────────────────────────

def _make_body_link(robot: ET.Element):
    link = ET.SubElement(robot, "link", name="base_link")

    # Visual — orange chassis (Mars rover colour)
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "box", size=f"{BODY_W} {BODY_L} {BODY_H}")
    mat = ET.SubElement(vis, "material", name="orange")
    ET.SubElement(mat, "color", rgba="0.85 0.45 0.1 1")

    # Collision
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    cgeom = ET.SubElement(col, "geometry")
    ET.SubElement(cgeom, "box", size=f"{BODY_W} {BODY_L} {BODY_H}")

    # Inertial
    inert = ET.SubElement(link, "inertial")
    ET.SubElement(inert, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    ET.SubElement(inert, "mass", value=str(BODY_MASS))
    i = ET.SubElement(inert, "inertia")
    for k, v in _inertia_box(BODY_MASS, BODY_W, BODY_L, BODY_H).items():
        i.set(k, v)


def _make_wheel_link(robot: ET.Element, name: str, side: str):
    """
    side: 'left' or 'right' — affects visual colour only.
    Wheel cylinder axis = Y (so it spins around Y when the rover drives forward).
    """
    link = ET.SubElement(robot, "link", name=name)

    rgba = "0.15 0.15 0.15 1"   # dark rubber grey for all wheels

    # Visual
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(1.5708, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=str(WHEEL_R), length=str(WHEEL_W))
    mat = ET.SubElement(vis, "material", name="rubber")
    ET.SubElement(mat, "color", rgba=rgba)

    # Add tread detail as a slightly larger visual-only cylinder
    vis2 = ET.SubElement(link, "visual")
    ET.SubElement(vis2, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(1.5708, 0, 0))
    geom2 = ET.SubElement(vis2, "geometry")
    ET.SubElement(geom2, "cylinder",
                  radius=str(round(WHEEL_R + 0.006, 4)),
                  length=str(round(WHEEL_W * 0.5, 4)))
    mat2 = ET.SubElement(vis2, "material", name="tread")
    ET.SubElement(mat2, "color", rgba="0.08 0.08 0.08 1")

    # Collision — single cylinder
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(1.5708, 0, 0))
    cgeom = ET.SubElement(col, "geometry")
    ET.SubElement(cgeom, "cylinder", radius=str(WHEEL_R), length=str(WHEEL_W))

    # Inertial
    inert = ET.SubElement(link, "inertial")
    ET.SubElement(inert, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    ET.SubElement(inert, "mass", value=str(WHEEL_MASS))
    i = ET.SubElement(inert, "inertia")
    for k, v in _inertia_cylinder(WHEEL_MASS, WHEEL_R, WHEEL_W).items():
        i.set(k, v)


def _make_suspension_link(robot: ET.Element, name: str, length: float):
    """Thin rod link used for rocker and bogie arms."""
    link = ET.SubElement(robot, "link", name=name)

    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, length / 2, 0), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=str(LINK_R), length=str(length))
    mat = ET.SubElement(vis, "material", name="silver")
    ET.SubElement(mat, "color", rgba="0.7 0.7 0.7 1")

    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(0, length / 2, 0), rpy=_rpy(0, 0, 0))
    cgeom = ET.SubElement(col, "geometry")
    ET.SubElement(cgeom, "cylinder", radius=str(LINK_R), length=str(length))

    inert = ET.SubElement(link, "inertial")
    ET.SubElement(inert, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    ET.SubElement(inert, "mass", value=str(LINK_MASS))
    i = ET.SubElement(inert, "inertia")
    for k, v in _inertia_cylinder(LINK_MASS, LINK_R, length).items():
        i.set(k, v)


def _make_mast_link(robot: ET.Element):
    """Sensor mast on top of body — holds the lidar."""
    link = ET.SubElement(robot, "link", name="mast_link")
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, 0, 0.1), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius="0.02", length="0.20")
    mat = ET.SubElement(vis, "material", name="white")
    ET.SubElement(mat, "color", rgba="0.9 0.9 0.9 1")
    inert = ET.SubElement(link, "inertial")
    ET.SubElement(inert, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    ET.SubElement(inert, "mass", value="0.1")
    i = ET.SubElement(inert, "inertia")
    for k, v in _inertia_cylinder(0.1, 0.02, 0.20).items():
        i.set(k, v)


# ── Joint builders ────────────────────────────────────────────────────────────

def _make_wheel_joint(robot: ET.Element,
                      name: str, parent: str, child: str,
                      x: float, y: float, z: float):
    """Continuous (driven) wheel joint — rotates around Y axis."""
    joint = ET.SubElement(robot, "joint", name=name, type="continuous")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child",  link=child)
    ET.SubElement(joint, "origin", xyz=_xyz(x, y, z), rpy=_rpy(0, 0, 0))
    ET.SubElement(joint, "axis",   xyz="0 1 0")   # rotate around Y
    dyn = ET.SubElement(joint, "dynamics",
                        damping="0.05", friction="0.1")
    lim = ET.SubElement(joint, "limit",
                        effort="10", velocity="20",
                        lower="-1e9", upper="1e9")


def _make_passive_joint(robot: ET.Element,
                        name: str, parent: str, child: str,
                        x: float, y: float, z: float,
                        axis: str = "1 0 0"):
    """
    Revolute passive joint — the rocker-bogie pivot.
    Small angle limits ±30° prevent the suspension from
    flopping unrealistically.
    """
    joint = ET.SubElement(robot, "joint", name=name, type="revolute")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child",  link=child)
    ET.SubElement(joint, "origin", xyz=_xyz(x, y, z), rpy=_rpy(0, 0, 0))
    ET.SubElement(joint, "axis",   xyz=axis)
    ET.SubElement(joint, "limit",
                  lower="-0.52",  # −30°
                  upper="0.52",   # +30°
                  effort="0", velocity="1")
    ET.SubElement(joint, "dynamics", damping="0.5", friction="0.0")


def _make_fixed_joint(robot: ET.Element,
                      name: str, parent: str, child: str,
                      x: float, y: float, z: float):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child",  link=child)
    ET.SubElement(joint, "origin", xyz=_xyz(x, y, z), rpy=_rpy(0, 0, 0))


# ── Main generator ────────────────────────────────────────────────────────────

def generate_rover_urdf() -> str:
    """
    Build the full URDF tree and return it as an XML string.

    Kinematic tree:
      base_link
      ├── mast_link              (fixed, on top)
      ├── rocker_left            (passive revolute, left side)
      │   ├── wheel_front_left   (continuous, driven)
      │   └── bogie_left         (passive revolute)
      │       ├── wheel_mid_left (continuous, driven)
      │       └── wheel_rear_left(continuous, driven)
      └── rocker_right           (mirror of left)
    """
    robot = ET.Element("robot", name="mars_rover")

    # ── Links ──────────────────────────────────────────────────────────────────
    _make_body_link(robot)
    _make_mast_link(robot)

    sides = [("left", -1), ("right", +1)]   # (name_suffix, x_sign)

    for side, sx in sides:
        _make_suspension_link(robot, f"rocker_{side}",  length=0.22)
        _make_suspension_link(robot, f"bogie_{side}",   length=0.18)
        _make_wheel_link(robot, f"wheel_front_{side}", side)
        _make_wheel_link(robot, f"wheel_mid_{side}",   side)
        _make_wheel_link(robot, f"wheel_rear_{side}",  side)

    # ── Joints ─────────────────────────────────────────────────────────────────
    # Mast (fixed to top of body)
    _make_fixed_joint(robot, "mast_joint", "base_link", "mast_link",
                      x=0, y=BODY_L / 2 - 0.04, z=BODY_H / 2 + 0.10)

    for side, sx in sides:
        # Rocker pivot — attached to side of body, slightly forward
        _make_passive_joint(
            robot, f"rocker_{side}_joint",
            "base_link", f"rocker_{side}",
            x=sx * BODY_W / 2, y=FRONT_Y * 0.6, z=WHEEL_Z * 0.5,
            axis="1 0 0"
        )

        # Front wheel — at top of rocker arm
        _make_wheel_joint(
            robot, f"wheel_front_{side}_joint",
            f"rocker_{side}", f"wheel_front_{side}",
            x=sx * WHEEL_W * 0.5,
            y=FRONT_Y - FRONT_Y * 0.6,     # relative to rocker origin
            z=WHEEL_Z * 0.5
        )

        # Bogie pivot — attached to bottom of rocker arm
        _make_passive_joint(
            robot, f"bogie_{side}_joint",
            f"rocker_{side}", f"bogie_{side}",
            x=0, y=-(0.22 * 0.5), z=0,
            axis="1 0 0"
        )

        # Mid wheel
        _make_wheel_joint(
            robot, f"wheel_mid_{side}_joint",
            f"bogie_{side}", f"wheel_mid_{side}",
            x=sx * WHEEL_W * 0.5, y=0.09, z=-WHEEL_R - 0.02
        )

        # Rear wheel
        _make_wheel_joint(
            robot, f"wheel_rear_{side}_joint",
            f"bogie_{side}", f"wheel_rear_{side}",
            x=sx * WHEEL_W * 0.5, y=-0.09, z=-WHEEL_R - 0.02
        )

    return _pretty(robot)


def get_rover_urdf_path() -> str:
    """
    Generate the URDF, write it to a temp file, and return the file path.
    The file persists for the lifetime of the Python process.
    Call this once at the start of your program.
    """
    urdf_content = generate_rover_urdf()

    # Write to project root (so PyBullet's search path can find it)
    # Use a fixed filename so we don't accumulate temp files
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",          # go up to project root
        "mars_rover.urdf"
    )
    path = os.path.normpath(path)

    with open(path, "w") as f:
        f.write(urdf_content)

    return path


# ── CLI test: python rover_urdf_generator.py ──────────────────────────────────
if __name__ == "__main__":
    path = get_rover_urdf_path()
    print(f"URDF written to: {path}")

    # Quick validation — count links and joints
    tree = ET.parse(path)
    root = tree.getroot()
    links  = root.findall("link")
    joints = root.findall("joint")
    print(f"Links : {len(links)}  — {[l.get('name') for l in links]}")
    print(f"Joints: {len(joints)} — {[j.get('name') for j in joints]}")
    print("\nTo preview in PyBullet GUI, run:")
    print("  python src/environment/rover_physics.py")