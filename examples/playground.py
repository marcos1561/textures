
"""
Example usage of the PlayGround class from the textures.playground module.

This script demonstrates how to initialize and run a PlayGround instance with two sets of points
and a specified matrix type. The PlayGround visualizes geometric relationships between the points
and displays unique identifiers for each point.
"""

import os
import matplotlib
matplotlib.use(os.environ.get("MATPLOTLIB_BACKEND", "TkAgg"))

from textures.playground import PlayGround, MatrixType

playground = PlayGround(
    init_points_1=[
        [-0.5, 0],
        [0.5, 0],
        [0, 1],
        [0, -1],
    ],
    init_points_2=[
        [-1, 0],
        [1, 0],
        [0, 0.5],
        [0, -0.5],
    ],
    matrix_type=MatrixType.geometry,
    show_uids=True,
)
playground.run()