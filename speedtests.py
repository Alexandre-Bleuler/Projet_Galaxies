import sys
import os
sys.path.append(os.getcwd())

### WARNING: the working directory must be the root of the project

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import visualizer3d_vbo as vbo
import galaxy_generator as gg

import Alexandre.v_naive as A_naive
import Bonaventure.Vectorise.vect as B_vect


# Command line arguments

output_name=sys.argv[1]
is_class= bool(int(sys.argv[2]))
delta_t=float(sys.argv[3])

# For exemple :
# output_name="Alexandre_Naive"
# isclasse=1 for true or 0 for false


def get_data_file_names(max_number_of_bodies=1000):
    """
    A function that get the file name in DATA/galaxies_data

    Args :
    max_number_of_bodies : The maximum number of bodies considered 
    compatible with files of DATA/galaxies_data

    Return : 

    The list of the file names in increasing order of the number of bodies 
    """
    name_list=[]
    number_of_bodies=0
    while number_of_bodies < max_number_of_bodies:
        number_of_bodies+=50
        name_list.append(f"DATA/galaxies_data/galaxy_{number_of_bodies}")
    return name_list




if __name__ == "__main__":
    
    # Initialising statistical object 

    max_number_of_bodies=1000
    number_of_bodies=np.arange(50,max_number_of_bodies+50,50)
    data_file_names=get_data_file_names(max_number_of_bodies)
    average_time=np.zeros((len(number_of_bodies),2))
    average_time[:,0]=number_of_bodies

    number_of_updates=10

    # Loop through all the galaxies of DATA/galaxies_data

    for i,name in enumerate(data_file_names):

        print(f"\nTesting time of execution with {(i+1)*50} bodies")

        # Gathering galaxy data

        data=np.loadtxt(name, dtype=np.double)
        masses=data[:,0]
        colors=np.empty((len(masses),3))
        for k,mass in enumerate(masses):
            colors[k,:]=gg.generate_star_color(mass)
        positions=np.column_stack([data[:,1],data[:,2],data[:,3]])
        velocities=np.column_stack([data[:,4],data[:,5],data[:,6]])

        if is_class:
            ncorps=A_naive.NBodies(name)
            

        # Génération de luminosités
        
        luminosities = np.ones(np.shape(positions)[0]).astype(np.float32)

        # Définition des limites de l'espace
        
        bounds = ((-100, 100), (-100, 100), (-100, 100))

        if is_class:         
            visualizer = vbo.Visualizer3D(positions, colors, luminosities, bounds)
            average_time[i,1]=visualizer.run_stats(ncorps.update, delta_t,  number_of_updates)

        else: 
            visualizer = vbo.Visualizer3D(positions, colors, luminosities, bounds)
            updater= lambda delta_t : B_vect.update_auto(delta_t, positions, velocities, masses)
            average_time[i,1]=visualizer.run_stats(updater, delta_t,  number_of_updates)
        
    # Saving average time data
    
    np.savetxt("DATA/speedtests_data/" + output_name +f"_dt{delta_t}", average_time, fmt=["%d","%f"])




    

    
    
    """plt.figure()
    plt.plot(stars_list, avg_fps_list)
    plt.xlabel("Nombre d'étoiles")
    plt.ylabel("FPS moyen")
    plt.title("Evolution des perfomances")
    plt.show()"""



### OLD main
"""repeats = 3 # number of times a series of tests will be repeated
    steps = 10 # represents the number of iterations that the simulation performs during the test phase
    delta_t = 0.1
    stars_list = []
    avg_fps_list = []


    print("\nStarting")
    for n_stars in Liste:
        file_path = os.path.join(output_dir, f"galaxy_{n_stars}")
        if not os.path.exists(file_path):
            print(f"File not find: {file_path}")
            continue
        print(f"\n== {n_stars} stars: {file_path} ==")
        fps_list = []
        for r in range(1, repeats + 1):
            ncorps = NBodies(file_path)
            elapsed, fps = run_headless(ncorps, delta_t, steps=steps)
            fps_list.append(fps)
            print(f"Run {r}/{repeats}: {steps} steps -> {elapsed:.4f} s, {fps:.2f} FPS")
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
        stars_list.append(n_stars)
        avg_fps_list.append(avg_fps)
        print(f"Average FPS for {n_stars} stars: {avg_fps:.2f}\n")

    print("\n End: The numbers of stars increase the iteration time.")"""