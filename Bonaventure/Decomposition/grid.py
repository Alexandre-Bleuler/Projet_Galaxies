import numpy as np
import galaxy_generator
import visualizer3d_vbo
import time
import numba

DT = 0.01
G = 1.560339e-13


GRID_RESOLUTION_X = 50
GRID_RESOLUTION_Y = 50
GRID_RESOLUTION_Z = 5

@numba.njit
def compute_bounds(positions):
    n = positions.shape[0]
    min_vals = positions[0].copy()
    max_vals = positions[0].copy()
    for i in range(1, n):
        for j in range(3):
            if positions[i, j] < min_vals[j]:
                min_vals[j] = positions[i, j]
            if positions[i, j] > max_vals[j]:
                max_vals[j] = positions[i, j]
    margin = 0.05 * (max_vals - min_vals)
    min_vals -= margin
    max_vals += margin
    return min_vals, max_vals

@numba.njit
def build_grid(positions, masses, bounds):
    """
    Builds a regular grid with resolution (nx, ny, nz).
    Returns:
        cell_particles: 4D array of particle indices per cell
        cell_counts: number of particles per cell
        cell_i, cell_j, cell_k: cell indices for each particle
        cell_size: cell size (dx, dy, dz)
        min_bounds: min bounds of the grid
    """
    nx = GRID_RESOLUTION_X
    ny = GRID_RESOLUTION_Y
    nz = GRID_RESOLUTION_Z
    min_bounds, max_bounds = bounds

    
    dx = (max_bounds[0] - min_bounds[0]) / nx
    dy = (max_bounds[1] - min_bounds[1]) / ny
    dz = (max_bounds[2] - min_bounds[2]) / nz
    cell_size = np.array([dx, dy, dz])

    N = positions.shape[0]

    
    cell_counts = np.zeros((nx, ny, nz), dtype=np.int32)
    cell_particles = -np.ones((nx, ny, nz, N), dtype=np.int32)

    cell_i = np.zeros(N, dtype=np.int32)
    cell_j = np.zeros(N, dtype=np.int32)
    cell_k = np.zeros(N, dtype=np.int32)

    for idx in range(N):
        pos = positions[idx]

        i = int((pos[0] - min_bounds[0]) / dx)
        j = int((pos[1] - min_bounds[1]) / dy)
        k = int((pos[2] - min_bounds[2]) / dz)

        
        if i < 0: i = 0
        if j < 0: j = 0
        if k < 0: k = 0
        if i >= nx: i = nx - 1
        if j >= ny: j = ny - 1
        if k >= nz: k = nz - 1

        cell_i[idx] = i
        cell_j[idx] = j
        cell_k[idx] = k

        slot = cell_counts[i, j, k]
        cell_particles[i, j, k, slot] = idx
        cell_counts[i, j, k] += 1

    return cell_particles, cell_counts, cell_i, cell_j, cell_k, cell_size, min_bounds

@numba.njit
def compute_cell_properties(cell_particles, cell_counts, positions, masses):
    """
    Calculates for each cell: total mass and center of mass.
    Returns two 3D arrays: cell_mass, cell_cm (shape (nx, ny, nz, 3))
    """
    nx, ny, nz, max_particles = cell_particles.shape
    cell_mass = np.zeros((nx, ny, nz), dtype=np.float64)
    cell_cm = np.zeros((nx, ny, nz, 3), dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                count = cell_counts[i, j, k]
                if count == 0:
                    continue
                m_tot = 0.0
                cm_x = 0.0
                cm_y = 0.0
                cm_z = 0.0
                for p in range(count):
                    idx = cell_particles[i, j, k, p]
                    m = masses[idx]
                    m_tot += m
                    cm_x += positions[idx, 0] * m
                    cm_y += positions[idx, 1] * m
                    cm_z += positions[idx, 2] * m
                
                cell_mass[i, j, k] = m_tot
                cell_cm[i, j, k, 0] = cm_x / m_tot
                cell_cm[i, j, k, 1] = cm_y / m_tot
                cell_cm[i, j, k, 2] = cm_z / m_tot

    return cell_mass, cell_cm

@numba.njit(parallel=True)
def compute_acce_grid(positions, masses):
    N = positions.shape[0]
    acc = np.zeros((N, 3))

    
    bounds = compute_bounds(positions)
    cell_particles, cell_counts, cell_i, cell_j, cell_k, cell_size, min_bounds = build_grid(
        positions, masses, bounds
    )

    
    cell_mass, cell_cm = compute_cell_properties(cell_particles, cell_counts, positions, masses)

    nx, ny, nz, _ = cell_particles.shape

    
    max_cell_size = max(cell_size[0], cell_size[1], cell_size[2])
    threshold = 2.0 * max_cell_size  

    for i in numba.prange(N):
        ax = 0.0
        ay = 0.0
        az = 0.0

        pos_i = positions[i]

        
        for ci in range(nx):
            for cj in range(ny):
                for ck in range(nz):
                    count = cell_counts[ci, cj, ck]
                    if count == 0:
                        continue

                    
                    cm_x = cell_cm[ci, cj, ck, 0]
                    cm_y = cell_cm[ci, cj, ck, 1]
                    cm_z = cell_cm[ci, cj, ck, 2]

                    
                    dx_cm = cm_x - pos_i[0]
                    dy_cm = cm_y - pos_i[1]
                    dz_cm = cm_z - pos_i[2]
                    dist_cm_sq = dx_cm*dx_cm + dy_cm*dy_cm + dz_cm*dz_cm + 1e-8
                    dist_cm = np.sqrt(dist_cm_sq)

                    if dist_cm > threshold:
                        
                        inv_dist3 = 1.0 / (dist_cm_sq * dist_cm)
                        ax += G * cell_mass[ci, cj, ck] * dx_cm * inv_dist3
                        ay += G * cell_mass[ci, cj, ck] * dy_cm * inv_dist3
                        az += G * cell_mass[ci, cj, ck] * dz_cm * inv_dist3
                    else:
                        
                        for p in range(count):
                            j = cell_particles[ci, cj, ck, p]
                            if j == i:
                                continue
                            dx = positions[j, 0] - pos_i[0]
                            dy = positions[j, 1] - pos_i[1]
                            dz = positions[j, 2] - pos_i[2]
                            dist_sq = dx*dx + dy*dy + dz*dz + 1e-8
                            inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
                            ax += G * masses[j] * dx * inv_dist3
                            ay += G * masses[j] * dy * inv_dist3
                            az += G * masses[j] * dz * inv_dist3

        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az

    return acc

def update():
    global positions, velocities, acc

    start = time.time()

    positions += velocities * DT + 0.5 * acc * DT**2
    new_acc = compute_acce_grid(positions, masses)
    velocities += 0.5 * (acc + new_acc) * DT
    acc = new_acc

    print("Compute time:", time.time() - start)
    return positions.astype(np.float32)

if __name__ == "__main__":
    N_ETOILES = 3000
    masses, positions, velocities, colors = galaxy_generator.generate_galaxy(
        n_stars=N_ETOILES
    )

    masses = np.array(masses, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    velocities = np.array(velocities, dtype=np.float64)

    acc = compute_acce_grid(positions, masses)

    colors_array = np.array(colors, dtype=np.float32)
    luminosities = np.ones(len(masses), dtype=np.float32)

    if len(positions) > 0:
        max_coord = np.max(np.abs(positions)) * 2.0
    else:
        max_coord = 10.0

    bounds = [
        (-max_coord, max_coord),
        (-max_coord, max_coord),
        (-max_coord, max_coord),
    ]

    visualizer = visualizer3d_vbo.Visualizer3D(
        positions.astype(np.float32),
        colors_array,
        luminosities,
        bounds,
    )

    visualizer.run(updater=lambda dt: update(), dt=DT)