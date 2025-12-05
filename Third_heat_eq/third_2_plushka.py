import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from second import triangulate_two_intersecting_circles_union, filter_triangles_to_union


def remove_unused_vertices(verts, tris):
    used = np.unique(tris.flatten())
    reindex = -np.ones(len(verts), dtype=int)
    reindex[used] = np.arange(len(used))
    verts_new = verts[used]
    tris_new = reindex[tris]
    return verts_new, tris_new


def assemble_matrices(verts, tris):
    n = len(verts)
    rows, cols, Mdata, Kdata = [], [], [], []
    vertex_area = np.zeros(n)

    for tri in tris:
        i, j, k = tri
        p = verts[[i, j, k]]

        x1, y1 = p[0];
        x2, y2 = p[1];
        x3, y3 = p[2]
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        if area < 1e-15: continue

        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        grad = np.vstack((b, c)).T / (2 * area)

        Ke = area * (grad @ grad.T)
        Me = (area / 12) * (np.ones((3, 3)) + np.eye(3))

        idx = [i, j, k]
        for a in range(3):
            vertex_area[idx[a]] += area / 3
            for b in range(3):
                rows.append(idx[a])
                cols.append(idx[b])
                Mdata.append(Me[a, b])
                Kdata.append(Ke[a, b])

    M = sp.csr_matrix((Mdata, (rows, cols)), shape=(n, n))
    K = sp.csr_matrix((Kdata, (rows, cols)), shape=(n, n))
    return M, K, vertex_area


def configure_initial_state_delta(verts, vertex_area, center=(0, 0)):
    d = np.linalg.norm(verts - np.array(center), axis=1)
    idx = np.argmin(d)
    u0 = np.zeros(len(verts))
    u0[idx] = 1 / vertex_area[idx]
    return u0


def theoretical_solution(verts, t, D=1.0):
    if t == 0:
        return np.zeros(len(verts))
    r2 = np.sum(verts ** 2, axis=1)
    return np.exp(-r2 / (4 * D * t)) / (4 * np.pi * D * t)


def solve_heat_CN(verts, tris, u0, dt, tmax, D=1.0, save_every=1):
    M, K, vertex_area = assemble_matrices(verts, tris)

    A = M + 0.5 * dt * D * K
    B = M - 0.5 * dt * D * K

    # regularization (kills singularity safely)
    eps = 1e-12
    A = A + eps * sp.identity(A.shape[0])

    A = spla.factorized(A.tocsc())

    u = u0.copy()
    times = [0]
    solutions = [u.copy()]

    pics = []
    Nt = int(tmax / dt)

    times.append(0)
    solutions.append(u.copy())
    pic = save_plot(verts, tris, u, theoretical_solution(verts, 0), 0, 0)
    pics.append(pic)

    for k in range(1, Nt + 1):
        u = A(B @ u)
        t = k * dt
        if k % save_every == 0 or k == Nt:
            times.append(t)
            solutions.append(u.copy())
            pic = save_plot(verts, tris, u, theoretical_solution(verts, t), t, k)
            pics.append(pic)

    return times, solutions, vertex_area, pics


def save_plot(verts, tris, u, u_th, t, k, folder="Pics_plushka"):
    import matplotlib.tri as tri
    triang = tri.Triangulation(verts[:, 0], verts[:, 1], tris)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    c0 = ax[0].tricontourf(triang, u, 30)
    fig.colorbar(c0, ax=ax[0])
    ax[0].set_title("Numerical")

    c1 = ax[1].tricontourf(triang, u_th, 30)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title("Theoretical")

    c2 = ax[2].tricontourf(triang, abs(u - u_th), 30)
    fig.colorbar(c2, ax=ax[2])
    ax[2].set_title("Error")

    for a in ax:
        a.set_aspect("equal")

    os.makedirs(folder, exist_ok=True)
    fname = f"{folder}/step_{k:04}.png"
    plt.suptitle(f"t = {t:.5e}")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def make_gif(pics, name="heat_two_circles.gif"):
    imgs = [imageio.imread(p) for p in pics]
    imageio.mimsave(name, imgs, fps=10)


verts_all, tris_all, c1, c2 = triangulate_two_intersecting_circles_union(
    R1=1.0, R2=0.8, distance=1.2, num_boundary=80, area_max=0.002
)
tris = filter_triangles_to_union(verts_all, tris_all, c1, 1.0, c2, 0.8)
verts, tris = remove_unused_vertices(verts_all, tris)
print("Mesh cleaned")

M, K, v_area = assemble_matrices(verts, tris)
u0 = configure_initial_state_delta(verts, v_area)
print("Initial condition prepared")

times, solutions, v_area, pics = solve_heat_CN(verts, tris, u0, dt=1e-4, tmax=0.01, save_every=1)
print("Solved")

make_gif(pics)
print("Gif and pics saved")
