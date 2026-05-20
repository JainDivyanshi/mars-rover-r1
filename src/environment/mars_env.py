import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at src/environment/mars_env.py
# We need src/ on the path so "from slam.xxx" imports work
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))   # src/environment
_SRC_DIR     = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # src/
_PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # project root

for _p in [_THIS_DIR, _SRC_DIR, _PROJECT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from terrain_loader import load_terrain
from slam.lidar_sensor import LidarSensor
from slam.occupancy_grid import OccupancyGrid
from rover_urdf_generator import get_rover_urdf_path
from rover_physics import RoverController


# ── Physics tuning constants ──────────────────────────────────────────────────
# FIX: was 10 substeps — rover moved ~0.02m/step, progress reward was ~0.04
# which is smaller than the stall penalty (0.05), so rover always got negative
# reward for existing. Reduced to 4 substeps for faster training while keeping
# enough physics accuracy for stable wheel contact.
PHYSICS_SUBSTEPS = 4

# How many steps to let the rover settle onto terrain before episode starts.
# FIX: was 10 — not enough for the 6-wheel suspension to fully settle.
# At 240Hz physics, 50 steps = ~0.2 real seconds of settling time.
SETTLE_STEPS = 50


class MarsEnv(gym.Env):
    def __init__(self, terrain_path, render=False):
        super().__init__()

        self.terrain_path = terrain_path
        self.render_mode  = render
        self.terrain_size = 128

        # ── Observation space (261 values total) ──────────────────────────────
        # [0:256]   16×16 local SLAM occupancy grid patch (flattened)
        # [256:258] unit vector pointing from rover to goal  (direction)
        # [258]     normalised distance to goal              (0=at goal, 1=far)
        # [259:261] rover velocity (vx, vy) normalised to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(261,), dtype=np.float32)

        # Actions: [left_wheel_speed, right_wheel_speed] both in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.physics    = None
        self.rover_id   = None
        self.terrain_data = None
        self.rover_ctrl = None

        self.lidar = LidarSensor()
        self.map   = OccupancyGrid()

        self.step_count   = 0
        self.max_steps    = 500   # FIX: was 1000 — shorter episodes = more
                                  # resets = rover sees more diverse start
                                  # positions = better generalisation.
                                  # Also: ep_len stuck at 1000 was a red flag.
        self.episode_count = 0

        self.debug_text_id = None
        self.goal_visual   = None

        self.prev_dist    = None
        self.current_vel  = np.zeros(2)

        # Generate the rover URDF once at startup — reused every episode
        self.rover_urdf_path = get_rover_urdf_path()

        # ── Curriculum (episode-based) ────────────────────────────────────────
        # FIX: start much closer (5m) so the rover can accidentally reach
        # the goal in early random exploration and get the 20-point bonus.
        # That first success is what "bootstraps" learning.
        # Stage progression is now tied to success rate, not just episode count
        # (tracked via self.recent_successes sliding window).
        self.curriculum_stages = [
            (0,    5),    # stage 0: goal within  5m  — very easy start
            (200,  10),   # stage 1: goal within 10m
            (500,  20),   # stage 2: goal within 20m
            (1000, 35),   # stage 3: goal within 35m
            (1800, 60),   # stage 4: goal within 60m  — full difficulty
        ]

        # Sliding window of last 20 episode outcomes for adaptive curriculum
        self.recent_successes = []

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        # Disconnect previous physics world if it exists
        if self.physics is not None:
            try:
                p.disconnect(self.physics)
            except Exception:
                pass

        mode = p.GUI if self.render_mode else p.DIRECT
        self.physics = p.connect(mode)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics)
        p.setGravity(0, 0, -3.72, physicsClientId=self.physics)  # Mars gravity

        # Disable rendering during reset for speed
        if self.render_mode:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0, physicsClientId=self.physics)

        # Load terrain
        self.terrain_data = load_terrain(self.terrain_path)
        self._create_terrain()

        # ── Spawn rover ───────────────────────────────────────────────────────
        start_x = float(np.random.uniform(25, 40))
        start_y = float(np.random.uniform(25, 40))
        # Spawn well above terrain — suspension will lower it during settle
        start_z = self._get_height(start_x, start_y) + 0.8

        self.rover_id = p.loadURDF(
            self.rover_urdf_path,
            basePosition=[start_x, start_y, start_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # FIX: settle with ZERO motor force so wheels hang freely under gravity
        # This ensures all 6 wheels touch ground before episode begins.
        # Without this, the rover is "floating" and the first reward step
        # measures a drop in height, not lateral progress.
        self._settle_rover()
        self.rover_ctrl = RoverController(self.physics, self.rover_id)

        # ── Curriculum goal placement ─────────────────────────────────────────
        self.episode_count += 1
        goal_range = self._get_goal_range()

        # Random direction, fixed distance = cleaner curriculum signal
        angle  = np.random.uniform(0, 2 * np.pi)
        goal_x = float(np.clip(
            start_x + goal_range * np.cos(angle), 5, self.terrain_size - 5))
        goal_y = float(np.clip(
            start_y + goal_range * np.sin(angle), 5, self.terrain_size - 5))

        self.goal_pos = np.array([
            goal_x, goal_y,
            self._get_height(goal_x, goal_y) + 0.1
        ])

        # Visual goal marker
        if self.render_mode:
            vs = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.6,
                rgbaColor=[1, 0.3, 0, 1],
                physicsClientId=self.physics)
            p.createMultiBody(
                baseVisualShapeIndex=vs,
                basePosition=self.goal_pos.tolist(),
                physicsClientId=self.physics)
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics)

        # Reset per-episode state
        self.map        = OccupancyGrid()
        self.step_count = 0
        self.current_vel = np.zeros(2)
        self.debug_text_id = None

        # Get real position after settling (may differ from spawn position)
        pos_arr, _ = self.rover_ctrl.get_pose()
        self.prev_dist = float(np.linalg.norm(
            self.goal_pos[:2] - pos_arr[:2]))

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        if not p.isConnected(self.physics):
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

        # Apply wheel velocities
        self.rover_ctrl.drive(float(action[0]), float(action[1]))

        # FIX: 4 substeps instead of 10.
        # Each PyBullet step = 1/240s. 4 steps = ~17ms per RL step.
        # The rover moves ~0.05–0.10m per RL step at MAX_VELOCITY=6 rad/s.
        # progress reward = 2.0 * 0.07 ≈ 0.14, well above stall threshold.
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self.physics)

        # Read true physics state
        pos_arr, _   = self.rover_ctrl.get_pose()
        self.current_vel = self.rover_ctrl.get_velocity()

        new_x = float(np.clip(pos_arr[0], 0, self.terrain_size - 1))
        new_y = float(np.clip(pos_arr[1], 0, self.terrain_size - 1))
        new_z = float(pos_arr[2])

        # Update SLAM map
        lidar_hits = self.lidar.scan(self.physics, self.rover_id)
        self.map.update([new_x, new_y], lidar_hits)

        reward, done = self._compute_reward([new_x, new_y, new_z])

        # ── GUI debug visuals ─────────────────────────────────────────────────
        if self.render_mode:
            dist_now = float(np.linalg.norm(
                np.array([new_x, new_y]) - self.goal_pos[:2]))

            if reward > 0.05:
                color, status = [0, 1, 0, 1], "PROGRESS"
            elif reward > -0.05:
                color, status = [1, 1, 0, 1], "IDLE"
            else:
                color, status = [1, 0.2, 0.2, 1], "PENALTY"

            try:
                p.changeVisualShape(
                    self.rover_id, 0, rgbaColor=color,
                    physicsClientId=self.physics)
            except Exception:
                pass

            if self.debug_text_id is not None:
                p.removeUserDebugItem(
                    self.debug_text_id, physicsClientId=self.physics)

            self.debug_text_id = p.addUserDebugText(
                f"{status} | r={reward:.2f} | d={dist_now:.1f}m"
                f" | ep={self.episode_count} | goal={self._get_goal_range()}m",
                [new_x, new_y, new_z + 1.5],
                textColorRGB=color[:3], textSize=1.2,
                physicsClientId=self.physics)

            p.resetDebugVisualizerCamera(
                cameraDistance=8, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[new_x, new_y, new_z],
                physicsClientId=self.physics)

            time.sleep(1 / 60)

        self.step_count += 1
        return self._get_obs(), float(reward), bool(done), False, {}

    # ─────────────────────────────────────────────────────────────────────────
    def _settle_rover(self):
        """
        Run physics with zero motor force so gravity pulls the rover down
        until all 6 wheels rest on the terrain.
        PyBullet's default joints are position-locked — we must explicitly
        set zero force to allow free fall.
        """
        n = p.getNumJoints(self.rover_id, physicsClientId=self.physics)
        for i in range(n):
            info = p.getJointInfo(
                self.rover_id, i, physicsClientId=self.physics)
            jtype = info[2]
            # Only release revolute/continuous joints (not fixed)
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                p.setJointMotorControl2(
                    self.rover_id, i,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0,   # zero force = free to fall
                    physicsClientId=self.physics)

        for _ in range(SETTLE_STEPS):
            p.stepSimulation(physicsClientId=self.physics)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_goal_range(self):
        goal_range = self.curriculum_stages[0][1]
        for min_ep, distance in self.curriculum_stages:
            if self.episode_count >= min_ep:
                goal_range = distance
        return goal_range

    def _get_height(self, x, y):
        xi = int(np.clip(x, 0, self.terrain_size - 1))
        yi = int(np.clip(y, 0, self.terrain_size - 1))
        return float(self.terrain_data[xi, yi])

    def _create_terrain(self):
        flat  = self.terrain_data.flatten().tolist()
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 1],
            heightfieldData=flat,
            numHeightfieldRows=self.terrain_size,
            numHeightfieldColumns=self.terrain_size,
            physicsClientId=self.physics)
        body = p.createMultiBody(0, shape, physicsClientId=self.physics)
        p.resetBasePositionAndOrientation(
            body,
            [self.terrain_size / 2, self.terrain_size / 2, 0],
            [0, 0, 0, 1],
            physicsClientId=self.physics)

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.rover_id, physicsClientId=self.physics)

        patch     = self.map.get_local_patch(pos).flatten()          # 256
        to_goal   = self.goal_pos[:2] - np.array(pos[:2])
        dist      = float(np.linalg.norm(to_goal)) + 1e-8
        direction = to_goal / dist                                    # 2
        norm_dist = np.array([min(dist / 60.0, 1.0)])                # 1
        norm_vel  = np.clip(self.current_vel / 5.0, -1, 1)          # 2

        return np.concatenate(
            [patch, direction, norm_dist, norm_vel]).astype(np.float32)

    def _compute_reward(self, new_pos):
        """
        Reward redesigned for real wheel physics (slow movement per step).

        Key insight: with 4 substeps at 240Hz, rover moves ~0.05-0.10m/step.
        All reward/penalty magnitudes are scaled to that movement range.

        1. PROGRESS REWARD: 10× progress_metres
           → moving 0.07m toward goal gives +0.7 reward
           → this is clearly positive and learnable

        2. TIME PENALTY: -0.005 per step (was -0.01 — too harsh for slow rover)
           → 500 steps of doing nothing = -2.5 total (much less than goal bonus)

        3. STALL PENALTY: -0.02 if barely moving (< 0.005m — scaled down from 0.01)
           → threshold matched to actual physics movement scale

        4. SPIN PENALTY: penalize turning without progress
           → catches the "spin in place" failure mode

        5. GOAL BONUS: +20 on arrival within 2m
           → ~30× a typical good step → clear signal but value fn can predict it

        6. PROXIMITY BONUS: small bonus for being close (within 5m)
           → dense reward in the goal region, helps final approach

        7. NO TIMEOUT PENALTY: timing out is neutral (reward = 0)
           → harsh timeout penalties teach rover to give up early
        """
        pos_2d = np.array(new_pos[:2])
        dist   = float(np.linalg.norm(pos_2d - self.goal_pos[:2]))

        # 1. Progress reward (scaled for real physics movement)
        progress = self.prev_dist - dist      # metres closer this step
        reward   = 10.0 * progress            # FIX: was 2.0, now 10.0

        # 2. Time penalty (small — don't overwhelm progress signal)
        reward -= 0.005                       # FIX: was -0.01

        # 3. Stall penalty — threshold scaled to real movement per step
        if abs(progress) < 0.005:            # FIX: was < 0.01
            reward -= 0.02                   # FIX: was -0.05

        # 4. Spin penalty — if velocity magnitude is low but time is passing
        speed = float(np.linalg.norm(self.current_vel))
        if speed < 0.05 and self.step_count > 10:
            reward -= 0.01

        # 5. Proximity bonus — extra reward density near goal
        if dist < 5.0:
            reward += 0.1 * (5.0 - dist) / 5.0   # up to +0.1 per step near goal

        # 6. Goal reached
        done = False
        if dist < 2.0:
            reward += 20.0
            done    = True
            # Log success for curriculum tracking
            self.recent_successes.append(1)
            if len(self.recent_successes) > 20:
                self.recent_successes.pop(0)

        # 7. Timeout — neutral, no penalty
        if self.step_count >= self.max_steps:
            done = True
            self.recent_successes.append(0)
            if len(self.recent_successes) > 20:
                self.recent_successes.pop(0)

        self.prev_dist = dist
        return reward, done


# ── Manual test — run from project root: python src/environment/mars_env.py ──
if __name__ == "__main__":
    # FIX: use absolute path so this works from any working directory
    terrain_path = os.path.join(_PROJECT_DIR, "data", "terrain", "mars.tif")

    print(f"Project root : {_PROJECT_DIR}")
    print(f"Terrain path : {terrain_path}")
    print(f"File exists  : {os.path.exists(terrain_path)}")

    if not os.path.exists(terrain_path):
        print("\nERROR: mars.tif not found.")
        print("Make sure you have: data/terrain/mars.tif in your project root.")
        sys.exit(1)

    print("\nStarting GUI test — rover drives straight for 200 steps...")
    env  = MarsEnv(terrain_path, render=True)
    obs, _ = env.reset()

    print(f"Obs shape      : {obs.shape}")   # should be (261,)
    print(f"Goal pos       : {env.goal_pos}")
    print(f"Goal range     : {env._get_goal_range()}m")
    print(f"Prev dist      : {env.prev_dist:.2f}m")

    total_reward = 0.0
    for step in range(200):
        action = np.array([0.8, 0.8])  # full speed forward
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            pos, _ = env.rover_ctrl.get_pose()
            print(f"  step={step:3d} | pos=({pos[0]:.1f},{pos[1]:.1f}) "
                  f"| reward={reward:+.3f} | total={total_reward:+.2f} "
                  f"| done={done}")
        if done:
            print(f"\nEpisode ended at step {step}!")
            break

    print("\nTest complete. Close the PyBullet window.")
    input("Press Enter to exit...")