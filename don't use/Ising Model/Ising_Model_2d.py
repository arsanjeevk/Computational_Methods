import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def initialize_spins(Lx, Ly):
    '''Create a 2d array of spin'''
    np.random.seed(1)
    return np.random.choice([1,-1], size=(Lx,Ly))

def calculate_energy(spins, J):
    '''Calcuation of Energy of the 2d lattice using periodic boundary condition'''
    #Multiplying the spins of nearest spin states
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    return -J * np.sum(spins*convolve2d(spins, kernel , mode="same", boundary="wrap")) / 2

def metropolis_step(spins, B, J):
    '''Perform single metropolis update on 2d lattice at random location'''

    Lx, Ly = spins.shape

    for _ in range(Lx*Ly):
        i = np.random.randint(Lx)
        j = np.random.randint(Ly)

        delta_E = 2 * J * spins[i, j] * (spins[(i-1)%Lx, j] + spins[(i+1)%Lx, j] + spins[i, (j+1)%Ly] + spins[i, (j-1)%Ly])
        # %Lx and %Ly is for periodic boundary condition

        if delta_E <= 0 or np.random.rand() < np.exp(- B * delta_E):
            spins[i, j] *= -1
    return spins

def simulate_ising_2d(Lx, Ly, J, T, n_steps):
    """Run the 1D Ising model simulation."""
    B = 1.0 / T      # B --> beta
    spins = initialize_spins(Lx, Ly)
    magnetizations = []
    energies = []

    for step in range(n_steps):
        spins = metropolis_step(spins, B, J)
        magnetization = np.mean(spins)
        magnetizations.append(magnetization)
        energy = calculate_energy(spins, J)
        energies.append(energy)

    return spins, magnetizations, energies

def main():

    #Parameters
    L = 50            # Size of the the Square lattice
    T = 2.27           # Temperature in Kelvin
    J = 1.0           # Interaction Strength
    n_steps = 500    # Number of steps   

    # Run simulation
    final_spins, mags, Es = simulate_ising_2d(L, L, J, T, n_steps)


    # Plot magnetization over time
    plt.figure(1)
    plt.plot(mags)
    plt.xlabel('Monte Carlo step')
    plt.ylabel('Magnetisation')
    plt.title(f'Magnetisation (Temperature = {T})')
    plt.grid(True)
    
    # Plot Energy over time
    plt.figure(2)
    plt.plot(Es)
    plt.xlabel('Monte Carlo step')
    plt.ylabel('E')
    plt.title(f'Energy Fluctuation (Temperature = {T})')
    plt.grid(True)
    
    # Plot Energy over time
    plt.figure(1)
    plt.imshow(final_spins)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Spin of the System at T = {T})')
    plt.colorbar()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()



def Mag_Vs_T():
    T = np.linspace(0.5,4, 5)
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
