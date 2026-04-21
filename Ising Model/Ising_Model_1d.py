import numpy as np
import matplotlib.pyplot as plt

def initialize_spins(N):
    """Initialize a 1D spin chain with N spins randomly set to +1 or -1."""
    return np.random.choice([-1, 1], size=N)

def calculate_energy(spins, J):
    """Calculate the total energy of the 1D spin chain."""
    return -J * np.sum(spins * np.roll(spins, 1))  # periodic boundary

def metropolis_step(spins, beta, J):
    """Perform a single Metropolis update on the spin chain."""
    N = len(spins) 
    for _ in range(N):
        i = np.random.randint(N)
        delta_E = 2 * J * spins[i] * (spins[(i-1)%N] + spins[(i+1)%N])
        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i] *= -1
    return spins

def simulate_ising_1d(N, J, T, n_steps):
    """Run the 1D Ising model simulation."""
    beta = 1.0 / T
    spins = initialize_spins(N)
    magnetizations = []
    energies = []

    for step in range(n_steps):
        spins = metropolis_step(spins, beta, J)
        magnetization = np.sum(spins) / N
        magnetizations.append(magnetization)
        energy = calculate_energy(spins, J)
        energies.append(energy)

    return spins, magnetizations, energies

# Parameters
N = 100             # Number of spins
J = 1.0             # Interaction strength
T = 2.0             # Temperature
n_steps = 1000      # Number of Metropolis steps

# Run simulation
final_spins, mags, Es = simulate_ising_1d(N, J, T, n_steps)


# Plot magnetization over time
plt.plot(mags)
plt.xlabel('Monte Carlo step')
plt.ylabel('Magnetization')
plt.title('1D Ising Model (Metropolis-Hastings)')
plt.grid(True)
plt.show()

# Plot Energy over time
plt.plot(Es)
plt.xlabel('Monte Carlo step')
plt.ylabel('E')
plt.title('1D Ising Model (Metropolis-Hastings)')
plt.grid(True)
plt.show()

# Plot Energy over time
plt.plot(final_spins)
plt.xlabel('Lattice Site')
plt.ylabel('s')
plt.title('1D Ising Model (Metropolis-Hastings)')
plt.grid(True)
plt.show()