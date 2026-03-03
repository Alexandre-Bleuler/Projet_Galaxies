import numpy as np
from naive import Corps, NCorps  


corps1 = Corps(masse=5.972e24, position=[0, 0, 0], vitesse=[0, 0, 0], couleur="blue") 
corps2 = Corps(masse=7.348e22, position=[384400000, 0, 0], vitesse=[0, 1.6e3, 0], couleur="gray")  
collect = [corps1, corps2]
ncorps = NCorps(collect)
force_sur_terre = ncorps.force_attraction(0)
print("Force d'attraction sur la Terre:", force_sur_terre)
dt = 3600
force_sur_terre  = force_sur_terre[0]
accel_terre = force_sur_terre / corps1.masse
corps1.update(dt, accel_terre)
print("Nouvelle position de la Terre:", corps1.position)
print("Nouvelle vitesse de la Terre:", corps1.vitesse)

