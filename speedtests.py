import sys
import os

# Setting the root of the project in sys.path

script_dir=os.path.dirname(__file__)
sys.path.append(script_dir)


import numpy as np
import numba

import visualizer3d_vbo as vbo
import galaxy_generator as gg

import Versions.v_naive as v_naive
import Versions.v_vect as v_vect
import Versions.v_numba as v_numba
import Versions.v_rk4 as v_rk4
import Versions.v_verlet as v_verlet



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

    # Asking to the user the value of some parameters:

    print("""\nEnter the name of the file where the results will be saved: 
(however be aware that the value of the time step and number of iterations will be added to the name)""")
    output_name=input()
    print(
    """\nThe available versions are:\n
- v_naive, number=0\n
- v_vect, number=1\n
- v_numba, number=2\n
- v_rk4, number=3\n
- v_verlet, number=4.\n
Enter the number associated with the version you want to use:"""
    )
    version=int(input())
    print("\nEnter the value of the time step you want to use:")
    delta_t=float(input())
    print("\nEnter the number of iterations you want to make:")
    number_of_updates=int(input())

    # Initialising statistical objects

    max_number_of_bodies=1000
    number_of_bodies=np.arange(50,max_number_of_bodies+50,50)
    data_file_names=get_data_file_names(max_number_of_bodies)
    average_time=np.zeros((len(number_of_bodies),2))
    average_time[:,0]=number_of_bodies

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

        # Génération de luminosités
        
        luminosities = np.ones(np.shape(positions)[0]).astype(np.float32)

        # Définition des limites de l'espace
        
        bounds = ((-100, 100), (-100, 100), (-100, 100))

        # Setting the visualizer
        
        visualizer = vbo.Visualizer3D(positions, colors, luminosities, bounds)

        match version:
            case 0: 
                ncorps=v_naive.NBodies(name)
                average_time[i,1]=visualizer.run_stats(ncorps.update_stats, delta_t,  number_of_updates)
            case 1: 
                updater=lambda delta_t : v_vect.update_stats(delta_t, positions, velocities, masses)
                average_time[i,1]=visualizer.run_stats(updater, delta_t,  number_of_updates)
            case 2: 
                updater=lambda delta_t : v_numba.update_stats(delta_t, positions, velocities, masses)
                average_time[i,1]=visualizer.run_stats(updater, delta_t,  number_of_updates)
            case 3: 
                updater=lambda delta_t : v_rk4.update_stats(delta_t, positions, velocities, masses)
                average_time[i,1]=visualizer.run_stats(updater, delta_t,  number_of_updates)
            case 4: 
                updater=lambda delta_t : v_verlet.update_stats(delta_t, positions, velocities, masses)
                average_time[i,1]=visualizer.run_stats(updater, delta_t,  number_of_updates)
            case _:
                raise ValueError("Error: not the number of an existing version!")
        
    # Saving average time data
    
    np.savetxt("DATA/speedtests_data/" + output_name +f"_dt{delta_t}_iter{number_of_updates}", average_time, fmt=["%d","%f"])

