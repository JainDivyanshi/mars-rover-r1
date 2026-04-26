import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from terrain_loader import load_terrain
from slam.lidar_sensor import LidarSensor
from slam.occupancy_grid import OccupancyGrid
from rover_urdf_generator import save_urdf
from rover_physics import RoverPhysics


class MarsEnv(gym.Env):
    def __init__(self, terrain_path, render=False):
        super().__init__()

        self.terrain_path = terrain_path
        self.render_mode = render
        self.terrain_size = 128

        # ─── Observation space breakdown ──────────────────────────────────────
        # 256  = 16x16 local SLAM map patch (flattened)
        #   2  = unit vector pointing toward goal (direction)
        #   1  = normalised distance to goal
        #   2  = rover velocity (vx, vy) — NEW: helps agent understand momentum
        # ─────────────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(261,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)

        self.physics = None
        self.rover_id = None
        self.terrain_data = None

        self.lidar = LidarSensor()
        self.map = OccupancyGrid()

        self.step_count = 0
        self.max_steps = 1000
        self.episode_count = 0

        # Debug visual elements
        self.debug_text_id = None
        self.goal_visual = None

        # For velocity tracking
        self.prev_pos = None
        self.prev_dist = None

        # ── Curriculum thresholds (episodes, not steps) ──────────────────────
        # We space these out much wider so the rover actually masters each stage
        self.curriculum_stages = [
            (0,    10),   # stage 0: goal within 10m  (episodes 0-299)
            (300,  20),   # stage 1: goal within 20m  (episodes 300-699)
            (700,  35),   # stage 2: goal within 35m  (episodes 700-1199)
            (1200, 60),   # stage 3: goal within 60m  (episodes 1200+)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        if self.physics is not None:
            try:
                p.disconnect(self.physics)
            except Exception:
                pass

        mode = p.GUI if self.render_mode else p.DIRECT
        self.physics = p.connect(mode)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.physics)
        p.setGravity(0, 0, -3.72, physicsClientId=self.physics)

        # Disable rendering during reset for speed (GUI mode only)
        if self.render_mode:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0,
                physicsClientId=self.physics)

        self.terrain_data = load_terrain(self.terrain_path)
        self._create_terrain()

        # ── Spawn rover ───────────────────────────────────────────────────────
        # Keep rover away from edges (padding=20) to avoid boundary weirdness
        start_x = float(np.random.uniform(20, 40))
        start_y = float(np.random.uniform(20, 40))
        start_z = self._get_height(start_x, start_y) + 0.4

        self.rover_id = p.loadURDF(
            "r2d2.urdf",
            basePosition=[start_x, start_y, start_z],
            physicsClientId=self.physics
        )

        # ── Curriculum: pick goal distance based on episode count ─────────────
        self.episode_count += 1
        goal_range = self._get_goal_range()

        # Sample goal direction randomly, place at exactly goal_range distance
        # Using a fixed distance (not uniform range) makes curriculum cleaner
        angle = np.random.uniform(0, 2 * np.pi)
        goal_x = start_x + goal_range * np.cos(angle)
        goal_y = start_y + goal_range * np.sin(angle)
        goal_x = float(np.clip(goal_x, 5, self.terrain_size - 5))
        goal_y = float(np.clip(goal_y, 5, self.terrain_size - 5))

        self.goal_pos = np.array([
            goal_x, goal_y,
            self._get_height(goal_x, goal_y) + 0.1
        ])

        # Visual goal marker (GUI only)
        if self.render_mode:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.6,
                rgbaColor=[1, 0.3, 0, 1],  # orange — more visible
                physicsClientId=self.physics
            )
            self.goal_visual = p.createMultiBody(
                baseVisualShapeIndex=visual_shape,
                basePosition=self.goal_pos.tolist(),
                physicsClientId=self.physics
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1,
                physicsClientId=self.physics)

        # Reset SLAM map and tracking variables
        self.map = OccupancyGrid()
        self.step_count = 0
        self.prev_pos = np.array([start_x, start_y])
        self.prev_dist = float(np.linalg.norm(
            self.goal_pos[:2] - self.prev_pos))
        self.debug_text_id = None

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action):
        if not p.isConnected(self.physics):
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

        left_vel, right_vel = action * 5.0

        rover_pos, rover_orn = p.getBasePositionAndOrientation(
            self.rover_id, physicsClientId=self.physics)
        yaw = p.getEulerFromQuaternion(rover_orn)[2]

        # Differential drive kinematics
        linear_vel = (left_vel + right_vel) / 2.0
        angular_vel = (right_vel - left_vel) / 0.3  # wheel base = 0.3m

        vx = linear_vel * np.cos(yaw)
        vy = linear_vel * np.sin(yaw)

        new_x = float(rover_pos[0] + vx * 0.1)
        new_y = float(rover_pos[1] + vy * 0.1)
        new_x = np.clip(new_x, 0, self.terrain_size - 1)
        new_y = np.clip(new_y, 0, self.terrain_size - 1)
        new_z = self._get_height(new_x, new_y) + 0.3

        # New yaw from differential drive
        new_yaw = yaw + angular_vel * 0.1
        new_quat = p.getQuaternionFromEuler([0, 0, float(new_yaw)])

        p.resetBasePositionAndOrientation(
            self.rover_id,
            [new_x, new_y, new_z],
            new_quat,
            physicsClientId=self.physics
        )
        p.stepSimulation(physicsClientId=self.physics)

        # Update SLAM
        lidar_hits = self.lidar.scan(self.physics, self.rover_id)
        self.map.update([new_x, new_y], lidar_hits)

        # Track velocity for observation
        self.current_vel = np.array([vx, vy])
        self.prev_pos = np.array([new_x, new_y])

        reward, done = self._compute_reward([new_x, new_y, new_z])

        # ── GUI debug visuals ─────────────────────────────────────────────────
        if self.render_mode:
            if reward > 0.1:
                color = [0, 1, 0, 1]
                status = "PROGRESS"
            elif reward > -0.05:
                color = [1, 1, 0, 1]
                status = "IDLE"
            else:
                color = [1, 0.2, 0.2, 1]
                status = "PENALTY"

            p.changeVisualShape(
                self.rover_id, -1, rgbaColor=color,
                physicsClientId=self.physics)

            if self.debug_text_id is not None:
                p.removeUserDebugItem(
                    self.debug_text_id,
                    physicsClientId=self.physics)

            dist_to_goal = np.linalg.norm(
                np.array([new_x, new_y]) - self.goal_pos[:2])
            self.debug_text_id = p.addUserDebugText(
                f"{status} | r={reward:.2f} | d={dist_to_goal:.1f}m | ep={self.episode_count}",
                [new_x, new_y, new_z + 2.5],
                textColorRGB=color[:3],
                textSize=1.2,
                physicsClientId=self.physics
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=45,
                cameraPitch=-40,
                cameraTargetPosition=[new_x, new_y, new_z],
                physicsClientId=self.physics
            )
            time.sleep(1 / 60)

        self.step_count += 1
        obs = self._get_obs()
        return obs, float(reward), bool(done), False, {}

    # ──────────────────────────────────────────────────────────────────────────
    def _get_goal_range(self):
        """Return the current curriculum goal distance."""
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
        flat = self.terrain_data.flatten().tolist()
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 1],
            heightfieldData=flat,
            numHeightfieldRows=self.terrain_size,
            numHeightfieldColumns=self.terrain_size,
            physicsClientId=self.physics
        )
        terrain_body = p.createMultiBody(
            0, shape, physicsClientId=self.physics)
        p.resetBasePositionAndOrientation(
            terrain_body,
            [self.terrain_size / 2, self.terrain_size / 2, 0],
            [0, 0, 0, 1],
            physicsClientId=self.physics
        )

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.rover_id, physicsClientId=self.physics)

        # 16x16 local SLAM patch (flattened = 256 values)
        patch = self.map.get_local_patch(pos).flatten()

        # Direction to goal (unit vector, 2 values)
        to_goal = self.goal_pos[:2] - np.array(pos[:2])
        dist = float(np.linalg.norm(to_goal)) + 1e-8
        direction = to_goal / dist

        # Normalised distance (1 value) — clamped to [0,1]
        norm_dist = np.array([min(dist / 60.0, 1.0)])

        # Rover velocity estimate (2 values) — normalised
        vel = getattr(self, 'current_vel', np.zeros(2))
        norm_vel = np.clip(vel / 5.0, -1, 1)

        obs = np.concatenate([patch, direction, norm_dist, norm_vel])
        return obs.astype(np.float32)

    def _compute_reward(self, new_pos):
        """
        Reward design principles:
        1. Progress reward: dense signal pulling rover toward goal
        2. Small time penalty: encourages efficiency
        3. Stall penalty: discourages spinning in place
        4. Goal bonus: big but proportionate (not 200x larger than step rewards)
        5. NO negative terminal reward: we don't punish timeout harshly
           (harsh timeout penalties cause the rover to give up early)
        """
        pos_2d = np.array(new_pos[:2])
        dist = float(np.linalg.norm(pos_2d - self.goal_pos[:2]))

        # ── 1. Progress reward ────────────────────────────────────────────────
        # How much closer did we get this step?
        progress = self.prev_dist - dist  # positive = moved toward goal
        reward = 2.0 * progress           # scaled so max ~1.0 per step

        # ── 2. Time penalty ───────────────────────────────────────────────────
        reward -= 0.01  # tiny cost per step → encourages speed

        # ── 3. Stall penalty ─────────────────────────────────────────────────
        # If barely moving (|progress| < 0.01m) add extra penalty
        if abs(progress) < 0.01:
            reward -= 0.05

        # ── 4. Goal bonus ─────────────────────────────────────────────────────
        # 20.0 is ~10-20x a typical good step reward — significant but not
        # so massive that the value function can't predict it
        done = False
        if dist < 2.0:
            reward += 20.0
            done = True

        # ── 5. Timeout ────────────────────────────────────────────────────────
        if self.step_count >= self.max_steps:
            done = True
            # Small proportional reward for getting close even if not reached
            reward += max(0, (1.0 - dist / self.prev_dist) * 2.0)

        self.prev_dist = dist
        return reward, done


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = MarsEnv("data/terrain/mars.tif", render=True)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")  # should be (261,)
    print(f"Goal: {env.goal_pos}")
    print(f"Curriculum stage: {env._get_goal_range()}m")

    step = 0
    while True:
        # Drive straight forward
        action = np.array([0.8, 0.8])
        obs, reward, done, _, _ = env.step(action)
        step += 1
        if step % 100 == 0:
            print(f"Step {step} | reward={reward:.3f} | done={done}")
        if done:
            print("Episode done, resetting...")
            obs, _ = env.reset()
            step = 0


            #tensorboard --logdir tensorboard_logs