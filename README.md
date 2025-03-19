# Sweep And Prune (SAP)

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-NumPy-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-Numba-orange)

An optimized Python implementation of [Sweep and Prune](https://www.cs.princeton.edu/courses/archive/spr01/cs598b/papers/cohen95.pdf) (SAP) algorithm for fast broad-phase collision detection in 2D and 3D simulations.

## Description
The core algorithm efficiently detects potential collisions between objects by projecting their bounding boxes onto coordinate axes, sorting them, and identifying overlapping intervals—essentially "sweeping" through sorted object boundaries to "prune" impossible collisions.

This version presents several optimizations:
- Contiguous sorted array ordering for improved memory locality and reduced cache misses
- Direct pairwise comparisons instead of vectorized boolean masks for better Numba performance
- Thread-local storage with parallel strided processing to minimize synchronization overhead
- Early pruning via sorted x-axis traversal to eliminate unnecessary intersection tests
- Dynamic buffer allocation based on problem size and thread count

## Example

The main `query_pairs_2d` and `query_pairs_3d` functions take a single vectorized numpy array containing all AABBs (Axis-Aligned Bounding Boxes). Each AABB is represented as `[min_x, max_x, min_y, max_y]` for a 2d bounding rectangle or as `[min_x, max_x, min_y, max_y, min_z, max_z]` for a 3d bounding box. The algorithm then operates on the entire dataset at once:

```python
# Bulk extraction of all bounding volumes
bounds = bodies.get_AABB()

# Perform batch query to find all unique colliding pairs
pairs = query_pairs_2d(*bounds)
```
Note: The 2D and 3D implementations have been deliberately separated to eliminate runtime dimension checking and ensure optimal performance in high-frequency queries.
  
## Test
Can maintain 60fps up to 400,000 unique pairs of colliding rectangles from a dense set of **50,000** moving AABBs. In both 2d and 3d.

*Please note that the above test was carried out without displaying any primitives and only approximately reflects the performance of pure numerical collision detection calculations.
Rendering all bounding rectangles would logically only lower the frame rate.*

Tested on a 2020 MSI laptop with Intel Core i7-8750H CPU @ 2.20GHz and 32Gb RAM
  
## Dependancies

##### Visualization:
- py5 – Main graphical environment for testing and visualization.  
- render_utils (included) – Fast primitive rendering in Py5.  

##### Core dependencies:
- NumPy 
- Numba 
