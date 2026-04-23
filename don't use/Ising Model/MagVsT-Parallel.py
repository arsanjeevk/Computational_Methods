import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from Ising_Model_2d import simulate_ising_2d

# Global parameters for your system
J = 1.0
N = 5000
Lx, Ly = 50, 50
T_values = np.linspace(0.5, 4 , 10)

def simulate_for_T(T):
    spins, mags, Es = simulate_ising_2d(Lx, Ly, J, T, N)
    M = np.mean(np.abs(mags[int(0.5*len(mags)):]))
    return M

if __name__ == "__main__":
    with Pool() as pool:
        M_values = pool.map(simulate_for_T, T_values)
    plt.plot(T_values, M_values)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magentization (M)")
    plt.title("2d Ising Model : Magenetization Vs Temperature")
    plt.grid()
    plt.show()
