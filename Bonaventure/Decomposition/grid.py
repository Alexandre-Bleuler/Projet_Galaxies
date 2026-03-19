import numpy as np
import galaxy_generator
import visualizer3d_vbo
import time
import numba

DT = 0.001
G = 1.560339e-13


GRID_RESOLUTION_X = 20
GRID_RESOLUTION_Y = 20
GRID_RESOLUTION_Z = 2

@numba.njit(parallel=True)
def compute_bounds(positions):
    """
    A function to compute the bounds of the grid based on the radial extent of positions.

    Args:
        positions: the (number_of_bodies, 3)-array containing the coordinates of the bodies.

    Return:
        min_vals: array of minimum bounds [xmin, ymin, min_z]
        max_vals: array of maximum bounds [xmax, ymax, max_z]
    """
    n = positions.shape[0]
    
    max_r = 0.0
    min_z = positions[0, 2]
    max_z = positions[0, 2]
    
    for i in numba.prange(n):
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        
        r = (x*x + y*y)**0.5
        if r > max_r:
            max_r = r
        
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z
    
    xmin = -max_r
    xmax = max_r
    ymin = -max_r
    ymax = max_r
    
    margin_xy = 0.05 * max_r
    margin_z = 0.05 * (max_z - min_z)
    
    xmin -= margin_xy
    xmax += margin_xy
    ymin -= margin_xy
    ymax += margin_xy
    min_z -= margin_z
    max_z += margin_z
    
    min_vals = np.array([xmin, ymin, min_z])
    max_vals = np.array([xmax, ymax, max_z])
    
    return min_vals, max_vals

@numba.njit(parallel=True)
def build_grid(positions, masses, bounds):
    """
    Build a regular grid with resolution (nx, ny, nz).

    Args:
        positions: (N, 3) array of particle positions.
        masses: (N,) array of particle masses.
        bounds: tuple (min_bounds, max_bounds) defining the grid extents.

    Return:
        row_ptr: row pointers for cells (size num_cells + 1)
        col_indices: column indices (particle indices, size N)
        cell_counts: number of particles per cell (1D array, size num_cells)
        cell_i, cell_j, cell_k: cell indices for each particle
        cell_size: cell size (dx, dy, dz)
        min_bounds: min bounds of the grid
        num_cells: total number of cells
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
    num_cells = nx * ny * nz

    
    cell_counts = np.zeros(num_cells, dtype=np.int32)

    cell_i = np.zeros(N, dtype=np.int32)
    cell_j = np.zeros(N, dtype=np.int32)
    cell_k = np.zeros(N, dtype=np.int32)

    # First pass: count particles per cell
    for idx in numba.prange(N):
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

        cell_index = i + j * nx + k * nx * ny
        cell_counts[cell_index] += 1

    #Arrange the particles according to a table
    row_ptr = np.zeros(num_cells + 1, dtype=np.int32)
    cumsum = 0
    for cell in range(num_cells):
        row_ptr[cell] = cumsum
        cumsum += cell_counts[cell]
    row_ptr[num_cells] = cumsum

    
    col_indices = np.zeros(N, dtype=np.int32)
    cell_counters = np.zeros(num_cells, dtype=np.int32)  

    for idx in range(N):
        i = cell_i[idx]
        j = cell_j[idx]
        k = cell_k[idx]
        cell_index = i + j * nx + k * nx * ny
        local_idx = cell_counters[cell_index]
        col_indices[row_ptr[cell_index] + local_idx] = idx
        cell_counters[cell_index] += 1

    return row_ptr, col_indices, cell_counts, cell_i, cell_j, cell_k, cell_size, min_bounds, num_cells

@numba.njit(parallel=True)
def compute_cell_properties(row_ptr, col_indices, cell_counts, positions, masses, num_cells):
    """
    Calculate for each grid cell the total mass and center of mass.

    Args:
        row_ptr: CSR row pointers for cells.
        col_indices: CSR column indices (particle indices).
        cell_counts: number of particles per cell.
        positions: (N, 3) array of particle positions.
        masses: (N,) array of particle masses.
        num_cells: total number of cells.

    Return:
        cell_mass: (num_cells,) array of total mass per cell.
        cell_cm: (num_cells, 3) array of cell center-of-mass positions.
    """
    cell_mass = np.zeros(num_cells, dtype=np.float64)
    cell_cm = np.zeros((num_cells, 3), dtype=np.float64)

    for cell in numba.prange(num_cells):
        start = row_ptr[cell]
        end = row_ptr[cell + 1]
        count = cell_counts[cell]
        if count == 0:
            continue
        m_tot = 0.0
        cm_x = 0.0
        cm_y = 0.0
        cm_z = 0.0
        for p in range(start, end):
            idx = col_indices[p]
            m = masses[idx]
            m_tot += m
            cm_x += positions[idx, 0] * m
            cm_y += positions[idx, 1] * m
            cm_z += positions[idx, 2] * m
        
        cell_mass[cell] = m_tot
        cell_cm[cell, 0] = cm_x / m_tot
        cell_cm[cell, 1] = cm_y / m_tot
        cell_cm[cell, 2] = cm_z / m_tot

    return cell_mass, cell_cm

@numba.njit(parallel=True)
def compute_acce_grid(positions, masses):
    """
    Compute the gravitational acceleration on each body using a grid-based approximation.

    Args:
        positions: (N, 3) array of body positions.
        masses: (N,) array of body masses.

    Return:
        acc: (N, 3) array of accelerations for each body.
    """
    N = positions.shape[0]
    acc = np.zeros((N, 3))
    bounds = compute_bounds(positions)
    row_ptr, col_indices, cell_counts, cell_i, cell_j, cell_k, cell_size, min_bounds, num_cells = build_grid(
        positions, masses, bounds
    )

    
    cell_mass, cell_cm = compute_cell_properties(row_ptr, col_indices, cell_counts, positions, masses, num_cells)
    
    nx = GRID_RESOLUTION_X
    ny = GRID_RESOLUTION_Y
    nz = GRID_RESOLUTION_Z

    
    max_cell_size = max(cell_size[0], cell_size[1], cell_size[2])
    threshold = 2.0 * max_cell_size  

    for i in numba.prange(N):
        ax = 0.0
        ay = 0.0
        az = 0.0

        pos_i = positions[i]

        
        for cell in range(num_cells):
            start = row_ptr[cell]
            end = row_ptr[cell + 1]
            count = cell_counts[cell]
            if count == 0:
                continue

            
            cm_x = cell_cm[cell, 0]
            cm_y = cell_cm[cell, 1]
            cm_z = cell_cm[cell, 2]

            
            dx_cm = cm_x - pos_i[0]
            dy_cm = cm_y - pos_i[1]
            dz_cm = cm_z - pos_i[2]
            dist_cm_sq = dx_cm*dx_cm + dy_cm*dy_cm + dz_cm*dz_cm + 1e-8
            dist_cm = np.sqrt(dist_cm_sq)

            if dist_cm > threshold:
                
                inv_dist3 = 1.0 / (dist_cm_sq * dist_cm)
                ax += G * cell_mass[cell] * dx_cm * inv_dist3
                ay += G * cell_mass[cell] * dy_cm * inv_dist3
                az += G * cell_mass[cell] * dz_cm * inv_dist3
            else:
                
                for p in range(start, end):
                    j = col_indices[p]
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
    """
    Update positions and velocities using a Verlet-like time integration step.

    This function updates global state variables (`positions`, `velocities`, `acc`) and
    returns the updated positions as float32.

    Return:
        positions: updated (N, 3) array of body positions.
    """
    global positions, velocities, acc

    start = time.time()

    positions += velocities * DT + 0.5 * acc * DT**2
    new_acc = compute_acce_grid(positions, masses)
    velocities += 0.5 * (acc + new_acc) * DT
    acc = new_acc
    print("Compute time:", time.time() - start)
    return positions.astype(np.float32)

if __name__ == "__main__":
    N_ETOILES = 1000
    masses, positions, velocities, colors = galaxy_generator.generate_galaxy(
        n_stars = N_ETOILES
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