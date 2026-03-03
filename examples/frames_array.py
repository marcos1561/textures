"""
Computing texture and its derivatives using the FramesArray calculator.

This example demonstrates how to analyze texture properties from point trajectories
using the FramesArray calculator. It generates random point trajectories across two
frames and computes the texture and its derivatives.

The script:
1. Creates a regular grid with 5x5 cells over a 10x10 region
2. Generates 30 random points with 20 frames of movement data
3. Creates two sets of frames (`frames1` and `frames2`) with small random displacements.
    `frames2[i]` should be interpreted as the `frames1[i]` after a small time interval dt.
4. Calculates texture descriptors using Voronoi linking with a distance cutoff of grid_size/3.
5. Visualizes the computed matrices.

Optional Features:
- Interactive slider to visualize point links and trajectories frame by frame if `show_points_links=True`
"""
import os
import matplotlib
matplotlib.use(os.environ.get("MATPLOTLIB_BACKEND", "TkAgg"))

import numpy as np
import matplotlib.pyplot as plt

import grids
import textures as tx

# =
# Setup
# =

np.random.seed(42)
show_points_links = False

num_frames = 20
num_points = 30
grid_size = 10
dl = 0.1

grid = grids.RegularGrid(
    length=grid_size, height=grid_size,
    num_cols=5, num_rows=5,
)

# ==
# Creating random trajectories
# ==

frames1 = np.empty((num_frames, num_points, 2))
frames2 = np.empty((num_frames, num_points, 2))

frames1[0] = (np.random.random((num_points, grid.num_dims)) - 1/2) * grid_size

theta = np.random.random(num_points) * 2 * np.pi
frames2[0] = frames1[0] + np.array([np.cos(theta), np.sin(theta)]).T * dl
for idx in range(1, num_frames):
    theta = np.random.random(num_points) * 2 * np.pi
    frames1[idx] = frames2[idx-1] + np.array([np.cos(theta), np.sin(theta)]).T * dl
    theta = np.random.random(num_points) * 2 * np.pi
    frames2[idx] = frames1[idx] + np.array([np.cos(theta), np.sin(theta)]).T * dl

# ==
# Calculating texture and its derivatives
# ==

calc = tx.calculators.FramesArray(
    frames1, frames2, grid, 
    links_cfg=tx.links.VoronoiLink(grid_size/3),
    dt=0.01,
)
results = calc.calculate()

# =
# Plotting results
# =

fig, ax = plt.subplots(2, 3, tight_layout=None if show_points_links else True)

ax[0, 0].set_title("M")
tx.display.draw_matrices(ax[0, 0], calc.grid, results.M)

ax[0, 1].set_title("B")
tx.display.draw_matrices(ax[0, 1], calc.grid, results.B)

ax[0, 2].set_title("T")
tx.display.draw_matrices(ax[0, 2], calc.grid, results.T)

ax[1, 0].set_title("V")
tx.display.draw_matrices(ax[1, 0], calc.grid, results.V)

ax[1, 1].set_title("P")
tx.display.draw_matrices(ax[1, 1], calc.grid, results.P)

ax[1, 2].set_title("Omega")
tx.display.draw_count_2D(ax[1, 2], calc.grid, results.omega)

if show_points_links:
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = plt.Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        frame_idx = int(slider.val)
        ax[0, 0].clear()
        ax[0, 0].set_title("M")
        tx.display.draw_points_links(ax[0, 0], frames1[frame_idx], tx.links.VoronoiLink(grid_size/3).link_func(frames1[frame_idx]))
        tx.display.draw_matrices(ax[0, 0], calc.grid, results.M)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

plt.show()