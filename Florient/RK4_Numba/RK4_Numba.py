import numpy as np
import galaxy_generator
import visualizer3d_vbo
import time

import pylab as plt
import numba

DT = 0.01
G = 1.560339e-13


@numba.njit(parallel=True)
def compute_acce_numba(positions, masses):
    N = positions.shape[0]
    acc = np.zeros((N, 3))
    for i in numba.prange(N):  # parallel range
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(N):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]

                dist_sq = dx * dx + dy * dy + dz * dz + 1e-8  # here dist_sq is a scalar, then I cannot use a mask
                inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))

                ax += G * masses[j] * dx * inv_dist3
                ay += G * masses[j] * dy * inv_dist3
                az += G * masses[j] * dz * inv_dist3
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    return acc


def update(positions, masses, velocities, dt):

   # Étape 1
    start = time.time()
    a1 = compute_acce_numba(positions, masses)


    v1 = velocities
    p1 = positions

    # Étape 2
    a2 = compute_acce_numba(p1 + 0.5 * dt * v1, masses)

    v2 = v1 + 0.5 * dt * a1
    p2 = p1 + 0.5 * dt * v1

    # Étape 3
    a3 = compute_acce_numba(p2 + 0.5 * dt * v2, masses)

    v3 = v1 + 0.5 * dt * a2
    p3 = p1 + 0.5 * dt * v3

    # Étape 4
    a4 = compute_acce_numba(p3 + 0.5 * dt * v3, masses)

    v4 = v1 + dt * a3
    p4 = p1 + dt * v3

    # Mis à jour
    positions += (dt / 6 ) * (v1 + 2 * v2 + 2 * v3 + v4)
    velocities += (dt / 6 ) * (a1 + 2 * a2 + 2 * a3 + a4)


    t = time.time() - start


    print("Compute time:", t)

    return positions.astype(np.float32)


if __name__ == '__main__':

    N_ETOILES = 2000
    masses, positions, velocities, colors = galaxy_generator.generate_galaxy(n_stars=N_ETOILES)

    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    velocities = np.array(velocities, dtype=np.float64)
    colors_array = np.array(colors, dtype=np.float32)
    luminosities = np.ones(len(masses), dtype=np.float32)

    if len(positions) > 0:
        max_coord = np.max(np.abs(positions)) * 2.0
    else:
        max_coord = 10.0

    bounds = [(-max_coord, max_coord),
              (-max_coord, max_coord),
              (-max_coord, max_coord)]
    visualizer = visualizer3d_vbo.Visualizer3D(positions.astype(np.float32), colors_array, luminosities, bounds)
    visualizer.run(updater=lambda dt: update(positions, masses, velocities, dt), dt=DT)