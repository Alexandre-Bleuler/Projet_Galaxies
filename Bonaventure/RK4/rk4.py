import numpy as np 
import galaxy_generator
import visualizer3d_vbo
import time

import pylab as plt
import numba
DT = 0.01
G = 1.560339e-13

@numba.njit(parallel = True)
def compute_acce_numba(positions: np.ndarray, masses: np.ndarray)-> np.ndarray:
    N = positions.shape[0]
    acc = np.zeros((N,3))
    for i in numba.prange(N): #parallel range
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(N):
            if i != j :
                
                dx = positions[j, 0] - positions[i, 0]  
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]

                dist_sq = dx*dx + dy*dy + dz*dz + 1e-8 # here dist_sq is a scalar, then I cannot use a mask
                inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))

                ax += G * masses[j]*dx*inv_dist3
                ay += G * masses[j]*dy*inv_dist3
                az += G * masses[j]*dz*inv_dist3
        acc[i,0] = ax
        acc[i,1] = ay
        acc[i,2] = az
    return acc

@numba.njit(parallel = True)
def rk4_step(positions, velocities, masses, dt)-> np.ndarray:
    # k1
    k1_v = compute_acce_numba(positions, masses)  # dv/dt = a(x)
    k1_x = velocities                             # dx/dt = v

    # k2
    pos_k2 = positions + 0.5 * dt * k1_x
    vel_k2 = velocities + 0.5 * dt * k1_v
    k2_v = compute_acce_numba(pos_k2, masses)
    k2_x = vel_k2

    # k3
    pos_k3 = positions + 0.5 * dt * k2_x
    vel_k3 = velocities + 0.5 * dt * k2_v
    k3_v = compute_acce_numba(pos_k3, masses)
    k3_x = vel_k3

    # k4
    pos_k4 = positions + dt * k3_x
    vel_k4 = velocities + dt * k3_v
    k4_v = compute_acce_numba(pos_k4, masses)
    k4_x = vel_k4

    
    positions_new = positions + (dt/6.0) * (k1_x + 2.0*k2_x + 2.0*k3_x + k4_x)
    velocities_new = velocities + (dt/6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)

    return positions_new, velocities_new

def update():
    global positions, velocities

    start = time.time()
    positions, velocities = rk4_step(positions, velocities, masses, DT)
    print("Compute time:", time.time() - start)

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
    visualizer = visualizer3d_vbo.Visualizer3D(positions.astype(np.float32),colors_array,luminosities,bounds)
    visualizer.run(updater=lambda dt: update(), dt=DT)