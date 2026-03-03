import numpy as np
from numpy import linalg

class Corps():

    def __init__(self, masse, position, vitesse, couleur):
        self.masse = masse
        self.position = np.array(position, dtype=np.float64)
        self.vitesse = np.array(vitesse, dtype=np.float64)
        self.couleur = couleur


        
    def update(self, dt, acceleration):
        self.vitesse = self.vitesse + dt*acceleration
        self.position = self.position + dt * self.vitesse + (1/2)*(dt**2)*acceleration

    def distance(self, other):
        return np.linalg.norm(self.position - other.position)
    
                                       
class NCorps():
    def __init__(self, collect): 
        self.collect = collect
            
    def force_attraction(self, index):
        n = len(self.collect)
        G = 1.560339e-13
        P = self.collect
        S = np.zeros(3, dtype=np.float64)  
        for i in range(n):  
            
            if i == index:
                continue
            num = G * P[i].masse * P[index].masse
            den = (np.linalg.norm(-P[index].position + P[i].position))**3 + 1e-6 # to avoid divisions by 0 
            term = -P[index].position + P[i].position
            Si = (num / den) * term          
            S += Si
        S = S / P[index].masse
        return S       