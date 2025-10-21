# Generate two meshes: (1) union of two crossed circles, (2) single circle (center (0,0))
# Save each mesh as two CSV files: points and triangles.
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def generate_mesh_for_region(inside_func, bbox, num_samples=2500, boundary_samples=200, extra_points=None):
    xmin, xmax, ymin, ymax = bbox
    points = []
    tries = 0
    # rejection sample
    while len(points) < num_samples and tries < num_samples * 20:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if inside_func(np.array([x, y])):
            points.append((x, y))
        tries += 1
    points = np.array(points)
    # boundary_samples should be added by caller if needed
    if extra_points is not None:
        points = np.vstack([points, extra_points])
    # Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    # filter triangles by centroid inside region
    centroids = points[triangles].mean(axis=1)
    mask = np.array([inside_func(c) for c in centroids])
    triangles = triangles[mask]
    return points, triangles


c1 = np.array([-0.4, 0.0]);
r1 = 1
c2 = np.array([0.4, 0.0]);
r2 = 1


def inside_crossed(p):
    d1 = np.sum((p - c1) ** 2)
    d2 = np.sum((p - c2) ** 2)
    return (d1 <= r1 ** 2) or (d2 <= r2 ** 2)


xmin = min(c1[0] - r1, c2[0] - r2) - 0.05
xmax = max(c1[0] + r1, c2[0] + r2) + 0.05
ymin = min(c1[1] - r1, c2[1] - r2) - 0.05
ymax = max(c1[1] + r1, c2[1] + r2) + 0.05
bbox_crossed = (xmin, xmax, ymin, ymax)

# prepare boundary samples for both circles
theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
b1 = c1 + np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
b2 = c2 + np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))
extra_crossed = np.vstack([b1, b2, c1, c2])

points_crossed, triangles_crossed = generate_mesh_for_region(inside_crossed, bbox_crossed, num_samples=2500,
                                                             extra_points=extra_crossed)

# Save crossed mesh CSVs
pcsv_crossed = "mesh_crossed_points.csv"
tcsv_crossed = "mesh_crossed_triangles.csv"
pd.DataFrame(points_crossed, columns=["x", "y"]).to_csv(pcsv_crossed, index=False)
pd.DataFrame(triangles_crossed, columns=["p1", "p2", "p3"]).to_csv(tcsv_crossed, index=False)

# --- Single circle (new circle at (0,0), r=0.8) ---
c_single = np.array([0.0, 0.0]);
r_single = 0.8


def inside_single(p):
    return np.sum((p - c_single) ** 2) <= r_single ** 2


xmin_s = c_single[0] - r_single - 0.05
xmax_s = c_single[0] + r_single + 0.05
ymin_s = c_single[1] - r_single - 0.05
ymax_s = c_single[1] + r_single + 0.05
bbox_single = (xmin_s, xmax_s, ymin_s, ymax_s)

# boundary points for single circle
b_single = c_single + np.column_stack((r_single * np.cos(theta), r_single * np.sin(theta)))
extra_single = np.vstack([b_single, c_single])

points_single, triangles_single = generate_mesh_for_region(inside_single, bbox_single, num_samples=2500,
                                                           extra_points=extra_single)

# Save single mesh CSVs
pcsv_single = "mesh_single_points.csv"
tcsv_single = "mesh_single_triangles.csv"
pd.DataFrame(points_single, columns=["x", "y"]).to_csv(pcsv_single, index=False)
pd.DataFrame(triangles_single, columns=["p1", "p2", "p3"]).to_csv(tcsv_single, index=False)

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title('Delaunay triangulation for circle')

# Plot only the filtered triangles (mesh)
ax.triplot(points_single[:,0], points_single[:,1], triangles_single, linewidth=0.6)

# Plot the point cloud lightly
ax.scatter(points_single[:,0], points_single[:,1], s=6)

# Outline the two circles for reference
circle1 = plt.Circle(c_single, r_single, fill=False, linewidth=1.0)
ax.add_patch(circle1)

# Save image
out_path = "Delaunay_single_circle.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)

fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.set_aspect('equal')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_title('Delaunay triangulation for two circles')

# Plot only the filtered triangles (mesh)
ax1.triplot(points_crossed[:,0], points_crossed[:,1], triangles_crossed, linewidth=0.6)

# Plot the point cloud lightly
ax1.scatter(points_crossed[:,0], points_crossed[:,1], s=6)

# Outline the two circles for reference
circle1 = plt.Circle(c1, r1, fill=False, linewidth=1.0)
circle2 = plt.Circle(c2, r2, fill=False, linewidth=1.0)
ax1.add_patch(circle1)
ax1.add_patch(circle2)

# Save image
out_path = "Delaunay_crossed_circles.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.show()
