"""
rover_physics.py
─────────────────
Controls the 6-wheeled rover using PyBullet joint motors.
Replaces the old resetBasePositionAndOrientation teleport hack
with physically accurate wheel torques.

Key concepts:
- VELOCITY_CONTROL mode: we set a target wheel angular velocity
  PyBullet's constraint solver applies the torque needed to reach it
- Differential drive: left wheels at speed V_L, right at V_R
  → linear vel = (V_L + V_R) / 2, angular = (V_R - V_L) / wheelbase
- All 6 wheels are driven (like Curiosity) for maximum traction
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rover_urdf_generator import get_rover_urdf_path


# ── Physical constants ────────────────────────────────────────────────────────
MAX_FORCE    = 30.0   # ✅ CHANGED: was 10.0 → increased torque (better traction)
MAX_VELOCITY = 6.0    # ✅ CHANGED: was 15.0 → reduced speed (more stable learning)
WHEELBASE    = 0.50   # m — lateral distance between left and right wheels


class RoverController:
    """
    Wraps a loaded rover URDF and provides a simple drive interface.

    Usage:
        rover = RoverController(physics_client, rover_id)
        rover.drive(left_speed=1.0, right_speed=1.0)  # range [-1, 1]
    """

    # These are the joint names that get motors.
    # Must match the names in rover_urdf_generator.py exactly.
    DRIVEN_JOINT_NAMES = [
        "wheel_front_left_joint",
        "wheel_front_right_joint",
        "wheel_mid_left_joint",
        "wheel_mid_right_joint",
        "wheel_rear_left_joint",
        "wheel_rear_right_joint",
    ]

    LEFT_JOINTS  = [
        "wheel_front_left_joint",
        "wheel_mid_left_joint",
        "wheel_rear_left_joint",
    ]
    RIGHT_JOINTS = [
        "wheel_front_right_joint",
        "wheel_mid_right_joint",
        "wheel_rear_right_joint",
    ]

    def __init__(self, physics_client: int, rover_id: int):
        self.client    = physics_client
        self.rover_id  = rover_id
        self._joint_map = {}   # name → joint index
        self._build_joint_map()
        self._enable_motors()

    def _build_joint_map(self):
        """Scan all joints and build a name → index lookup."""
        n = p.getNumJoints(
            self.rover_id, physicsClientId=self.client)
        for i in range(n):
            info = p.getJointInfo(
                self.rover_id, i, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            self._joint_map[joint_name] = i

        # Validate all expected joints exist
        for name in self.DRIVEN_JOINT_NAMES:
            if name not in self._joint_map:
                print(f"  WARNING: joint '{name}' not found in URDF.")
                print(f"  Available: {list(self._joint_map.keys())}")

    def _enable_motors(self):
        """
        Enable velocity control on all driven joints.
        Start at zero velocity with zero force — this disables the
        default position-lock that PyBullet applies to new joints.
        """
        for name in self.DRIVEN_JOINT_NAMES:
            if name not in self._joint_map:
                continue
            idx = self._joint_map[name]
            p.setJointMotorControl2(
                self.rover_id, idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,            # start passive — no lock
                physicsClientId=self.client
            )

    def drive(self, left_speed: float, right_speed: float):
        """
        Set wheel velocities.

        Args:
            left_speed:  [-1, 1] — negative = reverse
            right_speed: [-1, 1] — negative = reverse

        The input is normalised then scaled to MAX_VELOCITY.
        Left and right sides are set independently for differential steering.
        """
        left_vel  = float(np.clip(left_speed,  -1, 1)) * MAX_VELOCITY
        right_vel = float(np.clip(right_speed, -1, 1)) * MAX_VELOCITY

        for name in self.LEFT_JOINTS:
            if name in self._joint_map:
                p.setJointMotorControl2(
                    self.rover_id,
                    self._joint_map[name],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=left_vel,
                    force=MAX_FORCE,
                    physicsClientId=self.client
                )

        for name in self.RIGHT_JOINTS:
            if name in self._joint_map:
                p.setJointMotorControl2(
                    self.rover_id,
                    self._joint_map[name],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=right_vel,
                    force=MAX_FORCE,
                    physicsClientId=self.client
                )

    def stop(self):
        """Emergency stop — zero velocity, hold position."""
        for name in self.DRIVEN_JOINT_NAMES:
            if name in self._joint_map:
                p.setJointMotorControl2(
                    self.rover_id,
                    self._joint_map[name],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=MAX_FORCE,
                    physicsClientId=self.client
                )

    def get_velocity(self) -> np.ndarray:
        """
        Returns the rover's linear velocity (vx, vy) in world frame.
        Computed from the base link velocity, not wheel odometry.
        """
        vel, ang = p.getBaseVelocity(
            self.rover_id, physicsClientId=self.client)
        return np.array([vel[0], vel[1]])

    def get_pose(self) -> tuple:
        """Returns (position_xyz, yaw_radians)."""
        pos, orn = p.getBasePositionAndOrientation(
            self.rover_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array(pos), float(yaw)

    def print_joint_info(self):
        """Debug helper — print all joints and their current states."""
        n = p.getNumJoints(self.rover_id, physicsClientId=self.client)
        print(f"\nRover has {n} joints:")
        for i in range(n):
            info  = p.getJointInfo(self.rover_id, i,
                                    physicsClientId=self.client)
            state = p.getJointState(self.rover_id, i,
                                     physicsClientId=self.client)
            jtype = {0:"REVOLUTE", 1:"PRISMATIC",
                     2:"SPHERICAL", 3:"PLANAR", 4:"FIXED"}.get(info[2], "?")
            print(f"  [{i:2}] {info[1].decode():40s} {jtype:10s}  "
                  f"pos={state[0]:+.3f}  vel={state[1]:+.3f}")


# ── Standalone preview ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating rover URDF...")
    urdf_path = get_rover_urdf_path()
    print(f"URDF: {urdf_path}")

    print("Starting PyBullet GUI preview...")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -3.72, physicsClientId=client)
    p.setRealTimeSimulation(0, physicsClientId=client)

    p.loadURDF("plane.urdf", physicsClientId=client)

    rover_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.3],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=client,
        flags=p.URDF_USE_SELF_COLLISION
    )

    rover = RoverController(client, rover_id)
    rover.print_joint_info()

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.2],
        physicsClientId=client
    )

    print("\nControls in terminal (press Ctrl+C to stop):")
    print("  The rover will drive forward for 3s, then turn for 2s, then stop.")

    sequences = [
        ("Driving forward...",  1.0,  1.0,  3.0),
        ("Turning right...",    1.0, -1.0,  2.0),
        ("Driving forward...",  1.0,  1.0,  2.0),
        ("Turning left...",    -1.0,  1.0,  2.0),
        ("Reversing...",       -0.7, -0.7,  2.0),
        ("Stopped.",            0.0,  0.0,  1.0),
    ]

    for label, left, right, duration in sequences:
        print(f"  {label}")
        dt = 1.0 / 240.0
        steps = int(duration / dt)
        for _ in range(steps):
            rover.drive(left, right)
            p.stepSimulation(physicsClientId=client)
            time.sleep(dt)

    pos, yaw = rover.get_pose()
    print(f"\nFinal position: x={pos[0]:.2f}  y={pos[1]:.2f}  yaw={np.degrees(yaw):.1f}°")
    print("Close the PyBullet window to exit.")

    while p.isConnected(client):
        p.stepSimulation(physicsClientId=client)
        time.sleep(1.0 / 240.0)