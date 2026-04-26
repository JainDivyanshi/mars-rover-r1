def generate_rover_urdf():
    urdf = f"""
    <?xml version="1.0"?>
    <robot name="simple_rover">

        <!-- Base -->
        <link name="base">
            <inertial>
                <mass value="10"/>
                <inertia ixx="1" iyy="1" izz="1"/>
            </inertial>
            <visual>
                <geometry>
                    <box size="0.6 0.4 0.2"/>
                </geometry>
                <material name="gray">
                    <color rgba="0.5 0.5 0.5 1"/>
                </material>
            </visual>
            <collision>
                <geometry>
                    <box size="0.6 0.4 0.2"/>
                </geometry>
            </collision>
        </link>
    """

    wheel_positions = [
        (0.3, 0.25), (0.0, 0.25), (-0.3, 0.25),   # left side
        (0.3, -0.25), (0.0, -0.25), (-0.3, -0.25) # right side
    ]

    for i, (x, y) in enumerate(wheel_positions):
        urdf += f"""
        <link name="wheel_{i}">
            <visual>
                <geometry>
                    <cylinder length="0.05" radius="0.08"/>
                </geometry>
            </visual>
            <collision>
                <geometry>
                    <cylinder length="0.05" radius="0.08"/>
                </geometry>
            </collision>
        </link>

        <joint name="wheel_joint_{i}" type="continuous">
            <parent link="base"/>
            <child link="wheel_{i}"/>
            <origin xyz="{x} {y} -0.1" rpy="0 1.57 0"/>
            <axis xyz="0 1 0"/>
        </joint>
        """

    urdf += "</robot>"
    return urdf


def save_urdf(path="simple_rover.urdf"):
    with open(path, "w") as f:
        f.write(generate_rover_urdf())