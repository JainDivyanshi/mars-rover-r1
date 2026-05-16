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
from rover_urdf_generator import get_rover_urdf_path
from rover_physics import RoverController


class MarsEnv(gym.Env):
    def __init__(self, terrain_path, render=False):
        super().__init__()

        self.terrain_path = terrain_path
        self.render_mode = render
        self.terrain_size = 128

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

        self.debug_text_id = None
        self.goal_visual = None

        self.prev_pos = None
        self.prev_dist = None

        self.rover_urdf_path = get_rover_urdf_path()
        self.rover_ctrl = None

        self.curriculum_stages = [
            (0,    10),
            (300,  20),
            (700,  35),
            (1200, 60),
        ]

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

        if self.render_mode:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0,
                physicsClientId=self.physics)

        self.terrain_data = load_terrain(self.terrain_path)
        self._create_terrain()

        start_x = float(np.random.uniform(20, 40))
        start_y = float(np.random.uniform(20, 40))
        start_z = self._get_height(start_x, start_y) + 0.4

        self.rover_id = p.loadURDF(
            self.rover_urdf_path,
            basePosition=[start_x, start_y, start_z + 0.2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics,
            flags=p.URDF_USE_SELF_COLLISION
        )

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics)

        self.rover_ctrl = RoverController(self.physics, self.rover_id)

        self.episode_count += 1
        goal_range = self._get_goal_range()

        angle = np.random.uniform(0, 2 * np.pi)
        goal_x = start_x + goal_range * np.cos(angle)
        goal_y = start_y + goal_range * np.sin(angle)
        goal_x = float(np.clip(goal_x, 5, self.terrain_size - 5))
        goal_y = float(np.clip(goal_y, 5, self.terrain_size - 5))

        self.goal_pos = np.array([
            goal_x, goal_y,
            self._get_height(goal_x, goal_y) + 0.1
        ])

        if self.render_mode:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.6,
                rgbaColor=[1, 0.3, 0, 1],
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

        self.map = OccupancyGrid()
        self.step_count = 0
        self.prev_pos = np.array([start_x, start_y])
        self.prev_dist = float(np.linalg.norm(
            self.goal_pos[:2] - self.prev_pos))
        self.debug_text_id = None

        return self._get_obs(), {}

    def step(self, action):
        if not p.isConnected(self.physics):
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

        self.rover_ctrl.drive(float(action[0]), float(action[1]))

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics)

        pos_arr, yaw = self.rover_ctrl.get_pose()
        new_x = float(np.clip(pos_arr[0], 0, self.terrain_size - 1))
        new_y = float(np.clip(pos_arr[1], 0, self.terrain_size - 1))
        new_z = float(pos_arr[2])
        self.current_vel = self.rover_ctrl.get_velocity()
        self.prev_pos = np.array([new_x, new_y])

        lidar_hits = self.lidar.scan(self.physics, self.rover_id)
        self.map.update([new_x, new_y], lidar_hits)

        reward, done = self._compute_reward([new_x, new_y, new_z])

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

            try:
                p.changeVisualShape(
                    self.rover_id, 0, rgbaColor=color,
                    physicsClientId=self.physics)
            except Exception:
                pass

            if self.debug_text_id is not None:
                p.removeUserDebugItem(
                    self.debug_text_id,
                    physicsClientId=self.physics)

            dist_to_goal = np.linalg.norm(
                np.array([new_x, new_y]) - self.goal_pos[:2])
            self.debug_text_id = p.addUserDebugText(
                f"{status} | r={reward:.2f} | d={dist_to_goal:.1f}m | ep={self.episode_count}",
                [new_x, new_y, new_z + 1.5],
                textColorRGB=color[:3],
                textSize=1.2,
                physicsClientId=self.physics
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=8,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[new_x, new_y, new_z],
                physicsClientId=self.physics
            )
            time.sleep(1 / 60)

        self.step_count += 1
        obs = self._get_obs()
        return obs, float(reward), bool(done), False, {}

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

        patch = self.map.get_local_patch(pos).flatten()

        to_goal = self.goal_pos[:2] - np.array(pos[:2])
        dist = float(np.linalg.norm(to_goal)) + 1e-8
        direction = to_goal / dist

        norm_dist = np.array([min(dist / 60.0, 1.0)])

        vel = getattr(self, 'current_vel', np.zeros(2))
        norm_vel = np.clip(vel / 5.0, -1, 1)

        obs = np.concatenate([patch, direction, norm_dist, norm_vel])
        return obs.astype(np.float32)

    # ✅ ONLY MODIFIED PART
    def _compute_reward(self, new_pos):
        pos_2d = np.array(new_pos[:2])
        dist = float(np.linalg.norm(pos_2d - self.goal_pos[:2]))

        progress = self.prev_dist - dist
        reward = 2.0 * progress

        reward -= 0.01

        # ✅ FIXED stall penalty
        if abs(progress) < 0.005:
            reward -= 0.02

        # ✅ small survival reward
        reward += 0.002

        done = False
        if dist < 2.0:
            reward += 20.0
            done = True

        if self.step_count >= self.max_steps:
            done = True
            reward += max(0, (1.0 - dist / self.prev_dist) * 2.0)

        self.prev_dist = dist
        return reward, done