import matplotlib.pyplot as plt
import numpy as np

# GAthering speedtests' data

data_naive=np.loadtxt("DATA/speedtests_data/v_naive_dt0.01_iter10", dtype=np.double)
data_vect=np.loadtxt("DATA/speedtests_data/v_vect_dt0.01_iter10", dtype=np.double)
data_numba=np.loadtxt("DATA/speedtests_data/v_numba_dt0.01_iter10", dtype=np.double)
data_rk4=np.loadtxt("DATA/speedtests_data/v_rk4_dt0.01_iter10", dtype=np.double)
data_verlet=np.loadtxt("DATA/speedtests_data/v_verlet_dt0.01_iter10", dtype=np.double)
data_precond=np.loadtxt("DATA/speedtests_data/v_precond_dt0.01_iter10", dtype=np.double)

number_of_bodies=data_naive[:,0]
time_naive=data_naive[:,1]
time_vect=data_vect[:,1]
time_numba=data_numba[:,1]
time_rk4=data_rk4[:,1]
time_verlet=data_verlet[:,1]
time_precond=data_precond[:,1]

# Making graphics


plt.figure()
plt.plot(number_of_bodies, time_naive, color='b')
plt.plot(number_of_bodies, time_vect, color='r')
plt.plot(number_of_bodies, time_numba, color='g')
plt.plot(number_of_bodies, time_rk4, color='k')
plt.plot(number_of_bodies, time_verlet, color='m')
plt.plot(number_of_bodies, time_precond, color='y')
plt.yscale("log")
plt.xlabel("Number of bodies")
plt.ylabel("Mean compute time for an iteration (seconds)")
plt.legend(["naive", "vect", "numba", "rk4", "verlet", "precond"])
plt.show()

