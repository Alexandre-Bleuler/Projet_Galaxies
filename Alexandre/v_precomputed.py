import sys
import os

# Setting the root of the project in sys.path

script_dir=os.path.dirname(__file__)
project_dir=os.path.dirname(script_dir)
sys.path.append(project_dir)


import numpy as np 
import galaxy_generator as gg
import visualizer3d_vbo as vbo
import time
import numba


gravity_constant= 1.560339E-13

@numba.njit(parallel = True)
def get_max_dist(positions):
    """
    Get the maximal distance to the black hole in the Oxy plane and along z axis.

    Args:
        positions: the (number_of_bodie, 3) ndarray containing the positions of the galaxy's bodies.
    Return:
        max_xy, max_z: bound_xy the maximal distance in the Oxy plane and bound_z along the z-axis
    """

    max_xy = 0
    max_z = 0
    for i in numba.prange(1,np.shape(positions)[0]):
        body = positions[i,:]
        dist_xy = np.linalg.norm(body[:2], 2)
        dist_z  =np.abs(body[2])
        if dist_xy>max_xy:
            max_xy = dist_xy
        if dist_z>max_z:
            max_z=dist_z
    return max_xy, max_z

@numba.njit(parallel = True)
def get_bodies_box(positions,  bounds,  dp, bodies_indexes=None):
    """
    Get the indexes of the box where each body is.
    Args:
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        bounds: the (2-3)-array where the first line is the minimal bounds for each axis,
        and the second line the maximum ones for each axis.
        dp: the 3-array containing the boxes' spatial steps along each axis.
        bodies_indexes: the (number_of_bodies, 3) ndarray containing at a given line the (lower) indexes along
        each axix of the box where the corresponding body is. 
    Return:
        bodies_indexes: the actualized indexes.
    """

    if bodies_indexes==None:
        bodies_indexes=np.zeros((np.shape(positions)[0],3))
    for i in numba.prange(np.shape(positions)[0]):
        x,y,z=positions[i,:]
        bodies_indexes[i,0]=np.floor((x-bounds[0,0])/dp[0])
        bodies_indexes[i,1]=np.floor((y-bounds[0,1])/dp[1])
        bodies_indexes[i,2]=np.floor((z-bounds[0,2])/dp[2])    
    return bodies_indexes


@numba.njit(parallel = True)
def get_boxes_data(bodies_indexes, masses, positions, number_of_boxes):
    """
    Get the total mass and the center of gravity of a give box
    Args:
        bodies_box_indexes: the (number_of_bodies, 3) ndarray containing at a given line the (lower) indexes along
        each axix of the box where the corresponding body is. 
        masses: the array containing the masses of the galaxy's bodies. 
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        bounds: the (2-3)-array where the first line is the minimal bounds for each axis,
        and the second line the maximum ones for each axis.
        dp: the 3-array containing the boxes' spatial steps along each axis. 
        number_of_boxes: the 3-array containing the number of boxes along each axis.
    Return:
        boxes_data: the 4D-ndarray where the first three dimensions are the spatial indexes of the boxes
        and the last one contains the data of the corresponding box, namely its total mass and the three coordinates 
        of its center of gravity.
    """
    
    boxes_data=np.zeros((number_of_boxes[0], number_of_boxes[1], number_of_boxes[2], 4))

    for i in numba.prange(number_of_boxes[0]):
        for j in numba.prange(number_of_boxes[1]):
            for k in numba.prange(number_of_boxes[2]):
                for b in numba.prange(np.shape(masses)[0]):
                    b_indexes=bodies_indexes[b,:]
                    if i==b_indexes[0] and j==b_indexes[1] and k==b_indexes[2]:
                        body_mass=masses[b]
                        boxes_data[i,j,k,0]+=body_mass
                        boxes_data[i,j,k,1:]+=body_mass*positions[b,:]
                box_mass=boxes_data[i,j,k,0]
                if(box_mass>0):
                    boxes_data[i,j,k,1:]/=box_mass
    return boxes_data    


@numba.njit(parallel = True)
def compute_acce_numba(bodies_indexes, masses, positions, boxes_data, dp):
    """
    Compute the acceletation of the bodies in the current positions
    Args:
        bodies_box_indexes: the (number_of_bodies, 3) ndarray containing at a given line the (lower) indexes along
        each axix of the box where the corresponding body is. 
        masses: the array containing the masses of the galaxy's bodies. 
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        bounds: the (2-3)-array where the first line is the minimal bounds for each axis,
        and the second line the maximum ones for each axis.
        dp: the 3-array containing the boxes' spatial steps along each axis. 
        boxes_data: the 4D-ndarray where the first three dimensions are the spatial indexes of the boxes
        and the last one contains the data of the corresponding box, namely its total mass and the three coordinates 
        of its center of gravity.
        number_of_boxes: the 3-array containing the number of boxes along each axis.
    Return:
        acc: the nd-arrays containing in each line the acceleration of the corresponding body.
    """

    criteria=2*np.linalg.norm(dp,2)
    number_of_boxes=np.shape(boxes_data)[:3]
    number_of_bodies=np.shape(positions)[0]
    acc = np.zeros((number_of_bodies,3))

    # Acceleration of each body
    for b in numba.prange(number_of_bodies):
        body_position=positions[b,:]

        # Computing acceleration box by box
        for i in numba.prange(number_of_boxes[0]):
            for j in numba.prange(number_of_boxes[1]):
                for k in numba.prange(number_of_boxes[2]):
                    box_mass=boxes_data[i,j,k,0]
                    if box_mass==0:
                        continue
                    box_gravity_center=boxes_data[i,j,k,1:]
                    body_to_center=box_gravity_center-body_position
                    dist_to_center=np.linalg.norm(body_to_center)
                    if(dist_to_center>criteria):
                        # Approximation with gravity center since the box is sufficiently far away
                        acc[b,:]+=(gravity_constant*box_mass/dist_to_center**3)*body_to_center
                    else:
                        # Computing body by body of the box
                        for otherb in numba.prange(number_of_bodies):
                            otherb_indexes=bodies_indexes[b,:]
                            if otherb!=b and i==otherb_indexes[0] and j==otherb_indexes[1] and k==otherb_indexes[2]:
                                body_to_other=positions[otherb,:]-body_position
                                dist_to_other=np.linalg.norm(body_to_other)
                                acc[b,:]+=(gravity_constant*masses[otherb]/dist_to_other**3)*body_to_other

    return acc 


def update(delta_t, masses, positions, velocities, bounds, dp, number_of_boxes):
    """
    Compute the the new positions and velocities of the bodies.
    Args:
        delta_t: te choosen time step. 
        masses: the array containing the masses of the galaxy's bodies. 
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        veloctities: the (number_of_bodies, 3) ndarray containing the velocities of the galaxy's bodies.
        bounds: the (2-3)-array where the first line is the minimal bounds for each axis,
        and the second line the maximum ones for each axis.
        dp: the 3-array containing the boxes' spatial steps along each axis.
        number_of_boxes: the 3-array containing the number of boxes along each axis.
    Return:
        positions: the positions actualized.
    """
    start = time.time()
    bodies_indexes=get_bodies_box(positions,  bounds,  dp)
    boxes_data=get_boxes_data(bodies_indexes, masses, positions, number_of_boxes)
    new_acc = compute_acce_numba(bodies_indexes, masses, positions, boxes_data, dp)
    positions += velocities * delta_t + 0.5 * new_acc * delta_t**2
    bodies_indexes=get_bodies_box(positions,  bounds,  dp)
    boxes_data=get_boxes_data(bodies_indexes, masses, positions, number_of_boxes)
    new_acc2 = compute_acce_numba(bodies_indexes, masses, positions, boxes_data, dp)
    velocities += 0.5 * (new_acc2 + new_acc) * delta_t
    print("Compute time:", time.time() - start)
    return positions.astype(np.float32)

def update_stats(delta_t, masses, positions, velocities, bounds, dp, number_of_boxes):
    """
    Compute the the new positions and velocities of the bodies and measure the time needed to do the computations.
    Args:
        delta_t: te choosen time step. 
        masses: the array containing the masses of the galaxy's bodies. 
        positions: the (number_of_bodies, 3) ndarray containing the positions of the galaxy's bodies.
        veloctities: the (number_of_bodies, 3) ndarray containing the velocities of the galaxy's bodies.
        bounds: the (2-3)-array where the first line is the minimal bounds for each axis,
        and the second line the maximum ones for each axis.
        dp: the 3-array containing the boxes' spatial steps along each axis.
        number_of_boxes: the 3-array containing the number of boxes along each axis.
    Return:
        elapsed_update_time: the time needed to do the computations.
        positions: the positions actualized.
    """
    time_begin= time.time()
    bodies_indexes=get_bodies_box(positions,  bounds,  dp)
    boxes_data=get_boxes_data(bodies_indexes, masses, positions, boxes_data, number_of_boxes)
    new_acc = compute_acce_numba(positions, masses)
    positions += velocities * delta_t + 0.5 * new_acc * delta_t**2
    bodies_indexes=get_bodies_box(positions,  bounds,  dp)
    boxes_data=get_boxes_data(bodies_indexes, masses, positions, boxes_data, number_of_boxes)
    new_acc2 = compute_acce_numba(positions, masses)
    velocities += 0.5 * (new_acc2 + new_acc) * delta_t
    elapsed_update_time=time.time()-time_begin
    return elapsed_update_time, positions.astype(np.float32)

if __name__ == '__main__':

    delta_t = 0.01
    number_of_boxes=np.array([50,50,5])
    
    data= np.loadtxt("DATA/galaxies_data/galaxy_1000", dtype=np.double)
    masses=data[:,0]
    colors=np.empty((len(masses),3))
    for k,mass in enumerate(masses):
        colors[k,:]=gg.generate_star_color(mass)
    positions=np.column_stack([data[:,1],data[:,2],data[:,3]])
    velocities=np.column_stack([data[:,4],data[:,5],data[:,6]])

    luminosities = np.ones(np.shape(masses))
    
    max_xy,max_z=get_max_dist(positions)
    bounds=1.3*np.array([[-max_xy,-max_xy,-max_z],
                         [max_xy,max_xy,max_z]])

    dp=(bounds[1,:]-bounds[0,:])/number_of_boxes

    updater= lambda delta_t : update(delta_t, masses, positions, velocities, bounds, dp, number_of_boxes)

    vizualizer_bounds=((bounds[0,0],bounds[1,0]), (bounds[0,1],bounds[1,1]), (bounds[0,2],bounds[1,2]))

    visualizer = vbo.Visualizer3D(positions,colors,luminosities,vizualizer_bounds)
    visualizer.run(updater, dt=delta_t)