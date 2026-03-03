"""
Texture Calculation Example

This example demonstrates how to calculate and visualize texture data using Voronoi
linking and regular grid binning.

The example:
1. Generates 100 random points in a 2D space
2. Creates a 10x10 regular grid covering the unit square [0,1] x [0,1]
3. Links points to grid cells using Voronoi tessellation with a maximum distance of 0.1
4. Bins and aggregates texture values from points into grid cells
5. Computes mean texture values for each grid cell
6. Displays the results including:
    - Point locations and their links to grid cells
    - Texture heatmap overlaid on the grid
    - Grid structure with point scatter plot
"""

import numpy as np
import matplotlib.pyplot as plt

import grids
import textures as tx

np.random.seed(42)

num_points = 100
points = np.random.random((num_points, 2))

grid = grids.RegularGrid(
    length=1, height=1,
    num_cols=10, num_rows=10,
    center=(0.5, 0.5),
)

links = tx.links.VoronoiLink(max_dist=0.1).link_func(points)
texture_sum, texture_count = tx.bin_texture_sum(points, links, grid)
texture = tx.grid_data_mean(texture_sum, texture_count)

fig, ax = plt.subplots()

tx.display.draw_points_links(ax, points, links, points_kw={"label": "Points"}, links_kw={"label": "Links"})
tx.display.draw_matrices(ax, grid, texture)

ax.scatter(*points.T)
grid.plot_grid(ax)

ax.legend()

plt.show()