# Textures
A Python package to compute quantify, from a set of points and links between them, a texture and its time evolution.

Implementation of F. Graner, B. Dollet, C. Raufaste, and P. Marmottant, Discrete rearranging disordered patterns, part I: Robust statistical tools in two or three dimensions* **Eur. Phys. J. E** 25, 349-369 (2008) DOI [10.1140/epje/i2007-10298-8](https://doi.org/10.1140/epje/i2007-10298-8)

## Installation
```bash
pip install -e "git+https://github.com/marcos1561/textures.git/#egg=textures"
```

## Calculating the texture and its derivatives in grid cells
Given a set of points (array with shape (n# of points, n# of coordinates)), and a `Grid` from the package [grids](https://github.com/marcos1561/grids), one can compute the texture and its derivatives in each grid element in the following way:

1. Compute the links (see text below).
2. Use the appropriate functions to compute the desired quantity.  

### Computing links
Links are computed creating a `LinkCfg` object, then using the `link_cfg.link_func(points)`, where `points` is the array with the points:

```python
import textures as tx

points = ...

# Computing links using Voronoi tesselation
links_cfg = tx.links.VoronoiLink(max_dist=0.1) 
links_ids = links_cfg.link_func(points)
```
`links_ids` is an array with shape (n# of links, 2), and the i-th link can be constructed
as follow:
```python
id1, id2 = links_ids[i]
link_i = points[id2] - points[id1]
```

### Computing the texture
With the links in hands, we can compute the texture
```python
import texture as tx
texture_sum, texture_count = tx.bin_texture_sum(points, links_ids, grid)

# Averaging the results in each grid cell 
texture = tx.grid_data_mean(texture_sum, texture_count)
```
see the functions stating with `bin_` to compute other quantities.

## Calculators
One can use core functions (such as `bin_texture_sum()` in the section above) to calculate tools, but this is not convenient. To provide
a better user interface, calculators are provided inside the module `textures.calculators`.

### FramesArray Calculator
If you have a list of frames (a frame is a list of points) and want to do an average between all frames, `FramesArray` is the calculator for you. In the following example, all tools are calculated for a list of frames (doing an average over all frames), for every grid element, and the resulting texture is shown. 
```python
import matplotlib.pyplot as plt
import grids
from textures import calculators, links, display

# Suppose I have loaded frames1 and frames2 here.

grid = grids.RegularRectGridCfg(
    length=10, height=10,
    num_cols=5, num_rows=5,
).get_grid()

calc = calculators.FramesArray(
    frames1, frames2, grid, 
    links.VoronoiLink(),
    dt=0.01,
)
r = calc.calculate()

display.draw_matrices(plt.gca(), calc.grid, r.M)
plt.show()
```
see also the example [playground.py](./examples/playground.py).

## Playground
The playground is an application to play with the texture and its derivatives in an interactive way. It consists
of two frames, where the user can add points clicking with the mouse, or move existing points also with the mouse. At each frame,
links will be calculated on the fly and the respective selected quantity (M, B or T) will be shown on the frame as an ellipse.

The fallowing example initializes the playground with some points in both frames, configured to show the topological derivative:

```python
from textures import playground

app = playground.PlayGround(
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
    matrix_type=playground.MatrixType.topology,
    show_uids=True,
)
app.run()
```

After running this code, you should see the following

![Playground](docs/images/playground.png)
