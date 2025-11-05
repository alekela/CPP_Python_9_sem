"""
heat_delta_center.py

Solve u_t = alpha * Laplace(u) + S * delta(x-x0) on a unit circle with u=0 on boundary,
using P1 finite elements on a triangular mesh and backward Euler (implicit) time stepping.

- Delta source placed at center (0,0). The total source strength S (W) is
  distributed to the 3 nodes of the triangle containing the center using barycentric coords.
- Dirichlet BC (u=0) enforced strongly by replacing rows/cols in system.

Requires: numpy, scipy, matplotlib, triangle, and second.py (from your upload).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import math
import os
import imageio
from imageio.v2 import imread
import second  # your uploaded file that defines triangulate_circle_with_triangle


def assemble_matrices(vertices, triangles):
    """
    Assemble global Mass (M) and Stiffness (K) matrices for P1 triangles.
    vertices: (nverts,2)
    triangles: (ntri,3) with indices into vertices
    Returns sparse CSR matrices M, K
    """
    nverts = vertices.shape[0]
    I = []
    J = []
    Mvals = []
    Kvals = []

    for tri in triangles:
        idx = tri.astype(int)
        pts = vertices[idx]  # shape (3,2)

        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]

        # area
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        A = 0.5 * abs(det)
        if A <= 0:
            continue  # degenerate?

        # Gradients of barycentric basis functions (constant on triangle)
        # Using formula: grad phi1 = 1/(2A) * [y2 - y3, x3 - x2], cyclic
        b1 = np.array([y2 - y3, x3 - x2]) / (2 * A)
        b2 = np.array([y3 - y1, x1 - x3]) / (2 * A)
        b3 = np.array([y1 - y2, x2 - x1]) / (2 * A)
        grads = np.vstack([b1, b2, b3])  # shape (3,2)

        # local stiffness: Ke_ij = A * grad(phi_i) . grad(phi_j)
        Ke = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Ke[i, j] = A * np.dot(grads[i], grads[j])

        # local mass matrix (consistent): Me = (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
        Me = (A / 12.0) * (np.ones((3, 3)) + np.eye(3))

        # assemble
        for a in range(3):
            for b in range(3):
                I.append(idx[a])
                J.append(idx[b])
                Mvals.append(Me[a, b])
                Kvals.append(Ke[a, b])

    M = sp.csr_matrix((Mvals, (I, J)), shape=(nverts, nverts))
    K = sp.csr_matrix((Kvals, (I, J)), shape=(nverts, nverts))
    return M, K


def find_triangle_containing_point(vertices, triangles, p):
    """
    Return (tri_index, barycentric_coords) for triangle that contains point p.
    If none found, returns (None, None).
    """
    px, py = p
    for t_idx, tri in enumerate(triangles):
        idx = tri.astype(int)
        pts = vertices[idx]
        x1, y1 = pts[0];
        x2, y2 = pts[1];
        x3, y3 = pts[2]
        # Compute barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-14:
            continue
        l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        l3 = 1.0 - l1 - l2
        # Accept small negative tolerance to deal with boundary
        if l1 >= -1e-10 and l2 >= -1e-10 and l3 >= -1e-10:
            return t_idx, np.array([l1, l2, l3])
    return None, None


def boundary_node_mask(vertices, R=1.0, tol=1e-3):
    """
    Identify nodes on the boundary circle by distance from origin ~ R.
    """
    radii = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    return np.abs(radii - R) < tol


def enforce_dirichlet_system(A, b, dirichlet_nodes):
    """
    Modify sparse matrix A and RHS b to enforce u[dirichlet_nodes]=0 (strong Dirichlet).
    Replaces rows of A by identity on those nodes and set b[node]=0.
    """
    A = A.tolil()
    for n in dirichlet_nodes:
        A.rows[n] = [n]
        A.data[n] = [1.0]
        b[n] = 0.0
    return A.tocsr(), b


def solve_heat_on_circle(alpha=1.0, S=1.0, R=1.0, num_boundary=60, area_max=0.001,
                         dt=1e-3, t_end=0.05, plot_every=10, source=1):
    # 1) generate mesh using your triangulation routine
    verts, tris = second.triangulate_circle_with_triangle(R=R, num_boundary=num_boundary, area_max=area_max)
    verts = np.asarray(verts)
    tris = np.asarray(tris)

    # 2) Assemble FEM matrices
    print("Assembling matrices...")
    M, K = assemble_matrices(verts, tris)
    nverts = verts.shape[0]
    print(f"Vertices: {nverts}, Triangles: {len(tris)}")

    # 3) Find triangle that contains center (0,0) and compute barycentric weights
    tri_idx, bary = find_triangle_containing_point(verts, tris, (0.0, 0.0))
    if tri_idx is None:
        # fallback: find node closest to center and put entire source there
        nod = np.argmin(np.sum(verts ** 2, axis=1))
        F = np.zeros(nverts)
        F[nod] = S
        print("Center not inside any triangle — lumping delta at nearest node", nod)
    else:
        tri_nodes = tris[tri_idx].astype(int)
        F = np.zeros(nverts)
        # total source S distributed to the three nodes of containing triangle
        # physically the right-hand side entry should be integral of delta * phi_i = phi_i(x0)
        # which equals barycentric coordinate l_i
        for i_local, gi in enumerate(tri_nodes):
            F[gi] += S * bary[i_local]
        print("Delta source distributed to triangle", tri_idx, "nodes", tri_nodes, "bary", bary)

    # 4) Time stepping (backward Euler):
    # (M + dt * alpha * K) u^{n+1} = M u^n + dt * F
    A = M + dt * alpha * K

    # detect boundary nodes and enforce Dirichlet u=0
    bmask = boundary_node_mask(verts, R=R, tol=1e-3)
    dirichlet_nodes = np.where(bmask)[0]
    print("Dirichlet nodes:", len(dirichlet_nodes))

    # Pre-factor A but we must modify A to enforce Dirichlet first
    # We'll modify A and keep a factorized solver; the factorization stays valid because BCs are constant.
    # Build a sample RHS to modify system before factorization:
    # (we will factorize once)
    Ar, _ = enforce_dirichlet_system(A.copy(), np.zeros(nverts), dirichlet_nodes)
    print("Factorizing system matrix...")
    solver = spla.factorized(Ar.tocsc())

    # initial condition u0 = 0 everywhere
    u = np.zeros(nverts)
    u[tri_nodes] = source / len(tri_nodes)

    times = np.arange(0, t_end + 0.5 * dt, dt)
    snapshots = []
    snap_times = []
    snapshots.append(u.copy())
    snap_times.append(0)
    for k, t in enumerate(times[1:], start=1):
        rhs = M.dot(u) + dt * F
        # enforce Dirichlet in rhs (zero)
        rhs[dirichlet_nodes] = 0.0
        # solve
        u = solver(rhs)
        if k % plot_every == 0 or k == len(times) - 1:
            snapshots.append(u.copy())
            snap_times.append(t)
            print(f"t = {t:.4f} (step {k}/{len(times) - 1})")

    # Plot final solution and a few snapshots
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)

    for i, (uu, tt) in enumerate(zip(snapshots, snap_times)):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Heat equation with center delta source (P1 FEM, Backward Euler)")
        tpc = ax.tripcolor(triang, uu, shading='flat')
        ax.set_title(f"t = {tt:.4f}")
        ax.set_aspect('equal')
        plt.colorbar(tpc, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join("Pics", f"t={tt:.4f}.png"))
        plt.close("all")

    steps = []
    for file in os.listdir("Pics"):
        steps.append(float(file[:-4].split('=')[1]))
    steps.sort()
    images = []
    for step in steps:
        images.append(imread(os.path.join(f"Pics", f"t={step:.4f}.png")))
    imageio.mimsave(f"Res.gif", images)

    # Return solution arrays in case caller wants numeric data
    return verts, tris, times, snapshots


if __name__ == "__main__":
    # parameters (you can change these)
    alpha = 1.0  # diffusivity
    S = 0  # total strength of delta source
    source = 1.

    R = 1.0 # max side of triangle for meshing
    num_boundary = 80 # number of points on the boundary
    area_max = 0.0008 # max area for meshing

    dt = 1e-3
    t_end = 0.2
    plot_every = 5

    verts, tris, times, snapshots = solve_heat_on_circle(
        alpha=alpha, S=S, R=R, num_boundary=num_boundary, area_max=area_max,
        dt=dt, t_end=t_end, plot_every=plot_every, source=source
    )
