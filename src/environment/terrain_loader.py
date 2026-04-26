import rasterio
import numpy as np


def load_terrain(filepath, size=128):
    with rasterio.open(filepath) as src:
        elevation = src.read(1).astype(np.float32)

    # Crop center region
    h, w = elevation.shape
    cx, cy = h // 2, w // 2
    half = size // 2

    tile = elevation[cx-half:cx+half, cy-half:cy+half]

    # Normalize heights to 0–5 meters
    tile -= tile.min()
    tile /= (tile.max() + 1e-8)
    tile *= 5.0

    return tile


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tile = load_terrain("data/terrain/your_file.tif")

    plt.imshow(tile, cmap="terrain")
    plt.colorbar(label="Height (m)")
    plt.title("Mars Terrain Tile")

    plt.savefig("terrain_preview.png")
    print("Saved terrain_preview.png — open it to see your Mars map!")