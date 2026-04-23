import numpy as np
import matplotlib.pyplot as plt
from Ising_Model_2d import simulate_ising_2d

def Mag_Vs_T():
    T = np.linspace(0.5,4, 20)
    M = np.zeros_like(T)
    J = 1.0
    N = 500
    Lx , Ly = (50,50)
    
    for i in range(len(T)):
        spins, mags, Es = simulate_ising_2d(Lx, Ly, J, T[i], N)

        M[i] = np.mean(np.abs(mags[0.5*len(mags):]))

    return T, M

T, M = Mag_Vs_T()

plt.plot(T, M)
plt.show()

