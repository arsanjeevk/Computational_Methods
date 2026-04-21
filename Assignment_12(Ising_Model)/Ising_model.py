import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#Set seed
np.random.seed(1)

def initialize_spins(Lx, Ly):
    '''Create a 2d array of spin'''
    return np.random.choice([1, -1], size=(Lx, Ly))

def calculate_energy(spins, J):
    '''Calculation of Energy of the 2d lattice using periodic boundary condition'''
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    return -J * np.sum(spins * convolve2d(spins, kernel, mode="same", boundary="wrap")) / 2

def metropolis_step(spins, B, J):
    '''Perform single metropolis update on 2d lattice at random location'''
    Lx, Ly = spins.shape

    for _ in range(Lx * Ly):
        i = np.random.randint(Lx)
        j = np.random.randint(Ly)

        delta_E = 2 * J * spins[i, j] * (
            spins[(i-1)%Lx, j] + spins[(i+1)%Lx, j] +
            spins[i, (j+1)%Ly] + spins[i, (j-1)%Ly]
        )

        if delta_E <= 0 or np.random.rand() < np.exp(-B * delta_E):
            spins[i, j] *= -1

    return spins

def simulate_ising_2d(Lx, Ly, J, T, n_steps):
    """Run the 2D Ising model simulation."""
    B = 1.0 / T
    spins = initialize_spins(Lx, Ly)
    magnetizations = []
    energies = []

    for step in range(n_steps):
        spins = metropolis_step(spins, B, J)
        magnetizations.append(np.mean(spins))
        energies.append(calculate_energy(spins, J))

    return spins, magnetizations, energies


#single temprature analysis
def main():
    L = 50
    T = 2.27
    J = 1.0
    n_steps = 5000  

    final_spins, mags, Es = simulate_ising_2d(L, L, J, T, n_steps)

    #Magnetization vs time
    plt.figure()
    plt.plot(mags)
    plt.xlabel('Monte Carlo step')
    plt.ylabel('Magnetisation')
    plt.title(f'Magnetisation (T = {T})')
    plt.grid(True)

    #Energy vs time
    plt.figure()
    plt.plot(Es)
    plt.xlabel('Monte Carlo step')
    plt.ylabel('Energy')
    plt.title(f'Energy vs Time (T = {T})')
    plt.grid(True)

    #Final spin configuration
    plt.figure()
    plt.imshow(final_spins, cmap='coolwarm')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Spin Configuration at T = {T}')
    plt.colorbar()
    plt.show()


#Magnetization Vs Temprature
def Mag_Vs_T():
    T = np.linspace(1, 4, 20)
    M = np.zeros_like(T)

    J = 1.0
    Lx, Ly = (50, 50)

    equil_steps = 2000
    measure_steps = 3000

    for i in range(len(T)):
        spins = initialize_spins(Lx, Ly)
        B = 1.0 / T[i]

        mags = []

        #Equilibration
        for _ in range(equil_steps):
            spins = metropolis_step(spins, B, J)

        #Measurement
        for _ in range(measure_steps):
            spins = metropolis_step(spins, B, J)
            mags.append(np.mean(spins))

        M[i] = np.mean(np.abs(mags))

    return T, M


#Energy Fluctuation
def energy_fluctuation_vs_T():
    T = np.linspace(1, 4, 20)
    fluctuations = np.zeros_like(T)

    J = 1.0
    Lx, Ly = (50, 50)

    equil_steps = 2000
    measure_steps = 3000

    for i in range(len(T)):
        spins = initialize_spins(Lx, Ly)
        B = 1.0 / T[i]

        energies = []

        #Equilibration
        for _ in range(equil_steps):
            spins = metropolis_step(spins, B, J)

        #Measurement
        for _ in range(measure_steps):
            spins = metropolis_step(spins, B, J)
            energies.append(calculate_energy(spins, J))

        fluctuations[i] = np.var(energies)

    return T, fluctuations


#Representative spin states for the 5 different Temprature Values
def plot_representative_states():
    T_values = [1.0, 2.0, 2.27, 3.0, 4.0]
    L = 50
    J = 1.0
    steps = 5000

    plt.figure(figsize=(12, 8))

    for i, T in enumerate(T_values):
        spins, _, _ = simulate_ising_2d(L, L, J, T, steps)

        plt.subplot(2, 3, i+1)
        plt.imshow(spins, cmap='coolwarm')
        plt.title(f"T = {T}")
        plt.axis('off')

    plt.suptitle("Representative Spin Configurations")
    plt.show()



if __name__ == "__main__":
    main()

    #Magnetization vs Temperature
    T_vals, M_vals = Mag_Vs_T()
    plt.figure()
    plt.plot(T_vals, M_vals, 'o-')
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs Temperature")
    plt.grid()
    plt.show()

    #Energy Fluctuation
    T_vals2, F_vals = energy_fluctuation_vs_T()
    plt.figure()
    plt.plot(T_vals2, F_vals, 'o-')
    plt.xlabel("Temperature")
    plt.ylabel("Energy Fluctuation")
    plt.title("Energy Fluctuation vs Temperature")
    plt.grid()
    plt.show()

    #Spin configurations
    plot_representative_states()