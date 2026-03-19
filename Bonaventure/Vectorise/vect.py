import numpy as np 
import galaxy_generator
import visualizer3d_vbo
import time

DT = 0.001
G = 1.560339e-13

def compute_acce(positions, masses):
    N = len(masses)
    acc = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        diff = positions - positions[i]        
        dist_sq = np.sum(diff**2, axis =1) 

        mask = dist_sq < 10E-8 
        dist_sq[mask]
        dist_sq[i] = 1.0
        inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        inv_dist3[i] = 0.0
        acc[i] = np.sum(G * masses[:, None] * diff * inv_dist3[:, None], axis=0)
    return acc

def update():
    global positions, velocities

    start = time.time()
    acc = compute_acce(positions, masses)
    print("Compute time:", time.time() - start)

    positions += velocities*DT + 0.5*acc*DT**2
    velocities +=acc*DT
    return positions.astype(np.float32)

def update_auto(DT, positions, velocities, masses):

    start = time.time()
    acc = compute_acce(positions, masses)
    print("Compute time:", time.time() - start)

    positions += velocities*DT + 0.5*acc*DT**2
    velocities +=acc*DT
    return positions.astype(np.float32)

if __name__ == '__main__':
    
    N_ETOILES = 200
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
