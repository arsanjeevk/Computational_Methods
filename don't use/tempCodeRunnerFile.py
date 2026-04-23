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

