import numpy as np
from naive import Corps, NCorps

import galaxy_generator
import visualizer3d_vbo
import time


N_ETOILES = 200
DT = 0.01

masses, positions, velocities, colors = galaxy_generator.generate_galaxy(n_stars=N_ETOILES)

corps_list = [Corps(masses[i], positions[i], velocities[i], colors[i]) for i in range(len(masses))]

ncorps = NCorps(corps_list)

points = np.array([c.position for c in corps_list], dtype=np.float32)
colors_array = np.array([c.couleur for c in corps_list], dtype=np.float32)  

if len(points) > 0:
    max_coord = np.max(np.abs(points)) * 2.0 # for adaptation
else:
    max_coord = 10.0

bounds = [(-max_coord, max_coord), (-max_coord, max_coord), (-max_coord, max_coord)]


luminosities = np.ones(len(corps_list), dtype=np.float32)

start_time = time.time() 
frame = 0
def update(dummy):
    global frame
    frame += 1
    
    iteration_start_time = time.time()
    accelerations = []

    for i in range(len(corps_list)):
        a = ncorps.force_attraction(i)
        accelerations.append(a)

    
    for i in range(len(corps_list)):
        corps_list[i].update(DT, accelerations[i])
    
    

    
    new_points = np.array([c.position for c in corps_list], dtype=np.float32)
    iteration_time = time.time() - iteration_start_time
    print(f"Iteration time : {iteration_time}")
    #debug
    #if frame % 50 == 0:
    #    print(f"Frame {frame}, positions: {new_points[0]}")
    if frame % 10 == 0:
        total_time = time.time() - start_time  
        fps = frame / total_time 
        print(f"Frame {frame}, FPS: {fps:.2f}")
    return new_points


visualizer = visualizer3d_vbo.Visualizer3D(points, colors_array, luminosities, bounds)
print("\n Visualisation")
visualizer.run(updater=update, dt=DT)