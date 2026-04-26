import numpy as np
import matplotlib.pyplot as plt


class OccupancyGrid:
    def __init__(self, width=128, height=128, resolution=1.0):
        self.width = width
        self.height = height
        self.res = resolution

        self.grid = np.ones((height, width)) * 0.5
        self.rover_path = []

    def update(self, rover_pos, lidar_hits):
        rx, ry = int(rover_pos[0]), int(rover_pos[1])
        self.rover_path.append((rx, ry))

        for hit_x, hit_y, is_obstacle in lidar_hits:
            hx, hy = int(hit_x), int(hit_y)

            if 0 <= hx < self.width and 0 <= hy < self.height:
                if is_obstacle:
                    self.grid[hy, hx] = 1.0
                else:
                    self.grid[hy, hx] = 0.0

    def get_local_patch(self, rover_pos, size=16):
        rx, ry = int(rover_pos[0]), int(rover_pos[1])
        half = size // 2

        # Always create fixed-size patch
        patch = np.ones((size, size)) * 0.5

        for i in range(size):
            for j in range(size):
                x = rx - half + j
                y = ry - half + i

                if 0 <= x < self.width and 0 <= y < self.height:
                    patch[i, j] = self.grid[y, x]

        return patch

    def visualize(self, save_path=None):
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(self.grid, cmap='gray_r', vmin=0, vmax=1, origin='lower')

        if self.rover_path:
            xs = [p[0] for p in self.rover_path]
            ys = [p[1] for p in self.rover_path]

            ax.plot(xs, ys, 'r-', linewidth=1)
            ax.plot(xs[-1], ys[-1], 'ro')

        ax.set_title("SLAM Occupancy Map")

        if save_path:
            plt.savefig(save_path, dpi=100)

        plt.show()