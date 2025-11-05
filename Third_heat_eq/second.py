import math
import numpy as np
import matplotlib.pyplot as plt
import triangle  # pip install triangle


def circle_boundary_points(R=1.0, num_points=64, center=(0, 0)):
    """Generate equally spaced points on the boundary of a circle."""
    cx, cy = center
    return [
        (cx + R * math.cos(2 * math.pi * i / num_points),
         cy + R * math.sin(2 * math.pi * i / num_points))
        for i in range(num_points)
    ]


def triangulate_circle_with_triangle(R=1.0, num_boundary=64, area_max=None, center=(0, 0)):
    """
    Perform Delaunay triangulation inside a circle using the 'triangle' library.
    """
    # Generate boundary points and segments
    boundary = circle_boundary_points(R, num_boundary, center)
    vertices = np.array(boundary)
    segments = np.array([[i, (i + 1) % num_boundary] for i in range(num_boundary)])

    # Define PSLG (Planar Straight Line Graph)
    A = dict(vertices=vertices, segments=segments)

    # Triangulation options: p=PSLG, q30=quality constraint, a=area constraint
    options = "pq30"
    if area_max is not None:
        options += f"a{area_max}"

    # Run triangulation
    B = triangle.triangulate(A, options)

    verts = B["vertices"]
    tris = B["triangles"]

    return verts, tris


def triangulate_two_intersecting_circles_union(R1=1.0, R2=0.8, distance=1.2, num_boundary=64, area_max=None):
    """
    Perform Delaunay triangulation of the UNION of two intersecting circles.
    """
    # Generate boundary points for both circles
    center1 = (-distance / 2, 0)
    center2 = (distance / 2, 0)

    boundary1 = circle_boundary_points(R1, num_boundary, center1)
    boundary2 = circle_boundary_points(R2, num_boundary, center2)

    # Combine all vertices
    all_vertices = boundary1 + boundary2

    # Create segments for both circles
    segments1 = [[i, (i + 1) % num_boundary] for i in range(num_boundary)]
    segments2 = [[i + num_boundary, (i + 1) % num_boundary + num_boundary] for i in range(num_boundary)]
    all_segments = segments1 + segments2

    # Create a convex hull that contains both circles
    # Find the bounding box of both circles
    min_x = min(center1[0] - R1, center2[0] - R2)
    max_x = max(center1[0] + R1, center2[0] + R2)
    min_y = min(center1[1] - R1, center2[1] - R2)
    max_y = max(center1[1] + R1, center2[1] + R2)

    # Add some padding
    padding = 0.5
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    # Create bounding box vertices
    bbox_vertices = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]

    # Add bounding box to vertices
    all_vertices.extend(bbox_vertices)

    # Create segments for bounding box
    bbox_start_idx = len(boundary1) + len(boundary2)
    bbox_segments = [
        [bbox_start_idx, bbox_start_idx + 1],
        [bbox_start_idx + 1, bbox_start_idx + 2],
        [bbox_start_idx + 2, bbox_start_idx + 3],
        [bbox_start_idx + 3, bbox_start_idx]
    ]

    all_segments.extend(bbox_segments)

    # Convert to numpy arrays
    vertices = np.array(all_vertices)
    segments = np.array(all_segments)

    # Define PSLG (Planar Straight Line Graph)
    A = dict(vertices=vertices, segments=segments)

    # Triangulation options: p=PSLG, q30=quality constraint, a=area constraint
    options = "pq30"
    if area_max is not None:
        options += f"a{area_max}"

    # Run triangulation
    B = triangle.triangulate(A, options)

    verts = B["vertices"]
    tris = B["triangles"]

    return verts, tris, center1, center2


def is_point_in_circle(point, center, radius):
    """Check if a point is inside a circle."""
    return math.dist(point, center) <= radius


def filter_triangles_to_union(vertices, triangles, center1, R1, center2, R2):
    """
    Filter triangles to only keep those inside the union of the two circles.
    """
    filtered_triangles = []

    for tri in triangles:
        # Calculate centroid of the triangle
        tri_vertices = vertices[tri]
        centroid = np.mean(tri_vertices, axis=0)

        # Check if centroid is inside either circle
        if (is_point_in_circle(centroid, center1, R1) or
                is_point_in_circle(centroid, center2, R2)):
            filtered_triangles.append(tri)

    return np.array(filtered_triangles)


def mesh_score(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """
    Compute mesh quality score based on uniformity of triangle edge lengths.
    Score = average(min_edge / max_edge). 1.0 = perfect equilateral mesh.
    """
    scores = []
    for tri in triangles:
        pts = vertices[tri]
        d1 = math.dist(pts[0], pts[1])
        d2 = math.dist(pts[1], pts[2])
        d3 = math.dist(pts[2], pts[0])
        mn, mx = min(d1, d2, d3), max(d1, d2, d3)
        if mx > 0:
            scores.append(mn / mx)
    return sum(scores) / len(scores) if scores else 0.0


def plot_triangle_mesh(vertices, triangles, title="Triangle library mesh", centers=None, R1=1.0, R2=1.0):
    plt.figure(figsize=(10, 8))

    # Plot triangles
    for tri in triangles:
        pts = vertices[tri]
        plt.fill([pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0]],
                 [pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1]],
                 'lightblue', alpha=0.6, edgecolor='black', linewidth=0.5)

    # Plot circle boundaries for reference
    if centers is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle1_x = centers[0][0] + R1 * np.cos(theta)
        circle1_y = centers[0][1] + R1 * np.sin(theta)
        circle2_x = centers[1][0] + R2 * np.cos(theta)
        circle2_y = centers[1][1] + R2 * np.sin(theta)

        plt.plot(circle1_x, circle1_y, 'r--', linewidth=1, alpha=0.7, label='Circle boundaries')
        plt.plot(circle2_x, circle2_y, 'r--', linewidth=1, alpha=0.7)

        plt.plot(centers[0][0], centers[0][1], 'ro', markersize=5, label='Centers')
        plt.plot(centers[1][0], centers[1][1], 'ro', markersize=5)
        plt.legend()

    plt.axis("equal")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Single circle triangulation
    print("Single circle triangulation:")
    verts, tris = triangulate_circle_with_triangle(R=1.0, num_boundary=25, area_max=0.01)
    score = mesh_score(verts, tris)
    print(f"Generated {len(tris)} triangles. Mesh score: {score:.4f}")
    plot_triangle_mesh(verts, tris, title=f"Single Circle - Mesh score = {score:.4f}")

    # Two intersecting circles triangulation (UNION)
    print("\nTwo intersecting circles triangulation (UNION):")
    R1, R2, distance = 1.0, 0.8, 1.2
    center1 = (-distance / 2, 0)
    center2 = (distance / 2, 0)

    verts2, tris2, center1, center2 = triangulate_two_intersecting_circles_union(
        R1=R1, R2=R2, distance=distance, num_boundary=40, area_max=0.005
    )

    # Filter triangles to only keep those in the union of circles
    tris2_filtered = filter_triangles_to_union(verts2, tris2, center1, R1, center2, R2)

    score2 = mesh_score(verts2, tris2_filtered)
    print(f"Generated {len(tris2_filtered)} triangles. Mesh score: {score2:.4f}")
    plot_triangle_mesh(verts2, tris2_filtered,
                       title=f"Two Intersecting Circles (Union) - Mesh score = {score2:.4f}",
                       centers=[center1, center2], R1=R1, R2=R2)