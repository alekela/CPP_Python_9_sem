import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from second import triangulate_circle_with_triangle, triangulate_two_intersecting_circles_union


def assemble_matrices(verts, tris):
    n = verts.shape[0]
    rows, cols, M_data, K_data = [], [], [], []
    vertex_area = np.zeros(n)

    for tri in tris:
        i, j, k = tri
        p = verts[[i, j, k]]

        x1, y1 = p[0]
        x2, y2 = p[1]
        x3, y3 = p[2]

        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        if area < 1e-15: continue

        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        grads = np.vstack((b, c)).T / (2 * area)

        Ke = area * grads @ grads.T
        Me = (area / 12) * (np.ones((3, 3)) + np.eye(3))

        idx = [i, j, k]
        for a in range(3):
            vertex_area[idx[a]] += area / 3
            for b in range(3):
                rows.append(idx[a])
                cols.append(idx[b])
                M_data.append(Me[a, b])
                K_data.append(Ke[a, b])

    M = sp.csr_matrix((M_data, (rows, cols)), shape=(n, n))
    K = sp.csr_matrix((K_data, (rows, cols)), shape=(n, n))
    return M, K, vertex_area


def configure_initial_state_delta(verts, vertex_area):
    u0 = np.zeros(len(verts))
    idx = np.argmin(np.linalg.norm(verts, axis=1))
    u0[idx] = 1 / vertex_area[idx]  # unit mass
    return u0


def theoretical_solution(verts, t, D=1.0):
    if t == 0:
        return np.zeros(len(verts))

    r2 = verts[:, 0] ** 2 + verts[:, 1] ** 2
    return np.exp(-r2 / (4 * D * t)) / (4 * np.pi * D * t)


def solve_heat_CN(verts, tris, u0, dt, tmax, D=1.0):
    M, K, vertex_area = assemble_matrices(verts, tris)

    A = M + 0.5 * dt * D * K
    B = M - 0.5 * dt * D * K

    A = spla.factorized(A.tocsc())

    u = u0.copy()
    times = [0]
    solutions = [u.copy()]

    Nt = int(tmax / dt)
    for n in range(1, Nt + 1):
        u = A(B @ u)
        t = n * dt
        times.append(t)
        solutions.append(u.copy())

    return times, solutions, vertex_area


def plot_and_save(verts, tris, u, u_theory, t, k, folder="Pics"):
    import matplotlib.tri as tri

    triang = tri.Triangulation(verts[:, 0], verts[:, 1], tris)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    c0 = ax[0].tricontourf(triang, u, 30)
    ax[0].set_title("Numerical")
    fig.colorbar(c0, ax=ax[0])

    c1 = ax[1].tricontourf(triang, u_theory, 30)
    ax[1].set_title("Theory")
    fig.colorbar(c1, ax=ax[1])

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("y")

    fig.suptitle(f"t = {t:.5f}")

    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/step_{k:04}.png"
    plt.savefig(filename)
    plt.close()
    return filename


def make_gif(folder="Pics", name="heat.gif", fps=8):
    files = sorted([f"{folder}/{f}" for f in os.listdir(folder) if f.endswith(".png")])
    images = [imageio.imread(f) for f in files]
    imageio.mimsave(name, images, fps=fps)
    print(f"GIF saved as {name}")


R = 1.0
dt = 1e-4
tmax = 0.01
D = 1.0

verts, tris = triangulate_circle_with_triangle(R=R, num_boundary=80, area_max=0.0025)

M, K, vertex_area = assemble_matrices(verts, tris)

u0 = configure_initial_state_delta(verts, vertex_area)
print("Initial condition prepared")

times, solutions, vertex_area = solve_heat_CN(verts, tris, u0, dt, tmax, D)
print("Solved")

picture_files = []
for k, (t, u) in enumerate(zip(times, solutions)):
    u_th = theoretical_solution(verts, t, D)
    fname = plot_and_save(verts, tris, u, u_th, t, k)
    picture_files.append(fname)

make_gif("Pics", "heat.gif", fps=10)
print("Gif and pics saved")

# print("Check mass conservation:")
# for t, u in zip(times, solutions):
#     mass = np.sum(u * vertex_area)
#     print(f"t={t:.5e}, mass={mass:.6e}")
