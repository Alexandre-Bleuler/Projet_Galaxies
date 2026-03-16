import sys
import os

# Setting the root of the project in sys.path

script_dir=os.path.dirname(__file__)
project_dir=os.path.dirname(script_dir)
sys.path.append(project_dir)

import numpy as np 
import galaxy_generator
import visualizer3d_vbo
import time

G = 1.560339e-13

def compute_acce(positions, masses):
    """
    A function to compute a vectorized version of the acceleration
    using forward Euler scheme. 

    Args: 
    positions: the (number_of_bodies,3)-array containing on one line the the
    coordinates of the corresponding body.
    masses: the masses of the bodies.

    Return:
    Tha array contaning the accelerations in the same style as positions.

    """
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

def update(delta_t):
    """
    Update the positions and velocities of the galaxy's bodies given the 
    choosen time step.

    Args:
        delta_t: the choosen time step. 

    Return:
        positions: the (number_of_bodies,3)-array containing on one line the the
        coordinates of the corresponding body.
    """
    global positions, velocities

    start = time.time()
    acc = compute_acce(positions, masses)
    print("Compute time:", time.time() - start)

    positions += velocities*delta_t + 0.5*acc*delta_t**2
    velocities +=acc*delta_t
    return positions.astype(np.float32)

def update_stats(delta_t, positions, velocities, masses):
    """
    Compute the the new positions and velocities of the bodies and measure the time needed to do the computations.
    
    Args:
        delta_t: te choosen time step.  
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        veloctities: the (number_of_bodies, 3) ndarray containing the velocities of the galaxy's bodies.
        masses: the array containing the masses of the galaxy's bodies.

    Return:
        elapsed_update_time: the time needed to do the computations.
        positions: the positions actualized.
    """
    time_begin= time.time()
    acc = compute_acce(positions, masses)
    positions += velocities*delta_t + 0.5*acc*delta_t**2
    velocities +=acc*delta_t
    elapsed_update_time=time.time()-time_begin
    return elapsed_update_time, positions.astype(np.float32)

if __name__ == '__main__':
    
    DT = 0.001
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