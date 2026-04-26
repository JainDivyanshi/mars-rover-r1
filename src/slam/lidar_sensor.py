import pybullet as p
import numpy as np


class LidarSensor:
    def __init__(self, num_rays=36, max_range=15.0):
        self.num_rays = num_rays
        self.max_range = max_range

    def scan(self, physics_client, rover_id):
        pos, orn = p.getBasePositionAndOrientation(
            rover_id,
            physicsClientId=physics_client
        )

        yaw = p.getEulerFromQuaternion(orn)[2]

        hits = []

        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        for angle in angles:
            world_angle = angle + yaw

            dx = np.cos(world_angle) * self.max_range
            dy = np.sin(world_angle) * self.max_range

            ray_start = [pos[0], pos[1], pos[2] + 0.2]
            ray_end = [pos[0] + dx, pos[1] + dy, pos[2] + 0.2]

            result = p.rayTest(
                ray_start,
                ray_end,
                physicsClientId=physics_client
            )

            hit_fraction = result[0][2]

            if hit_fraction < 1.0:
                # Obstacle detected
                hit_x = pos[0] + dx * hit_fraction
                hit_y = pos[1] + dy * hit_fraction
                hits.append((hit_x, hit_y, True))
            else:
                # Free space
                hits.append((ray_end[0], ray_end[1], False))

        return hits