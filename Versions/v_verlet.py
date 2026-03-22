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
import numba


G = 1.560339e-13

@numba.njit(parallel = True)
def compute_acce_numba(positions: np.ndarray, masses: np.ndarray)-> np.ndarray:
    """
    A function to compute a the acceleration
    using forward Verlet scheme and the package numba. 

    Args: 
    positions: the (number_of_bodies,3)-array containing on one line the the
    coordinates of the corresponding body.
    masses: the masses of the bodies.

    Return:
    The array contaning the accelerations in the same style as positions.
    """

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

    global positions, velocities, acc
    start = time.time()
    positions += velocities * delta_t + 0.5 * acc * delta_t**2
    new_acc = compute_acce_numba(positions, masses)
    velocities += 0.5 * (acc + new_acc) * delta_t
    acc = new_acc
    print("Compute time:", time.time() - start)
    return positions.astype(np.float32)

 
def initialize_acc(positions, masses):
    """ 
    A function to compute the initial acceleration

    Args:
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        masses: the array containing the masses of the galaxy's bodies.

    Return:
        The accecleration array in the same style as positions.
    """
    return compute_acce_numba(positions, masses)

def update_stats(delta_t, positions, velocities, masses, acceleration):
    """
    Compute the the new positions and velocities of the bodies and measure the time needed to do the computations.
    
    Args:
        delta_t: te choosen time step.  
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        veloctities: the (number_of_bodies, 3) ndarray containing the velocities of the galaxy's bodies.
        masses: the array containing the masses of the galaxy's bodies.
        acceleration: the (number_of_bodies, 3) ndarray containing the acceleartions of the galaxy's bodies.

    Return:
        elapsed_update_time: the time needed to do the computations.
        positions: the positions actualized.
    """

    time_begin= time.time()
    positions += velocities * delta_t + 0.5 * acceleration * delta_t**2
    new_acc = compute_acce_numba(positions, masses)
    velocities += 0.5 * (acc + new_acc) * delta_t
    acceleration = new_acc
    elapsed_update_time=time.time()-time_begin
    return elapsed_update_time, positions.astype(np.float32)

if __name__ == '__main__':
    
    DT = 0.01
    N_ETOILES = 1000
    masses, positions, velocities, colors = galaxy_generator.generate_galaxy(n_stars=N_ETOILES)

    masses = np.array(masses, dtype=np.float64)            
    positions = np.array(positions, dtype=np.float64)          
    velocities = np.array(velocities, dtype=np.float64)   
    acc = compute_acce_numba(positions, masses)  # acceleration calculation     
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