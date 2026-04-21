import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from Ising_model import simulate_ising_2d, initialize_spins, metropolis_step, calculate_energy


# ================================
# GLOBAL PARAMETERS
# ================================
J = 1.0
Lx, Ly = 50, 50
T_values = np.linspace(1, 4, 20)

equil_steps = 2000
measure_steps = 3000


# ================================
# PARALLEL FUNCTION (MAGNETIZATION)
# ================================
def simulate_for_T(T):
    np.random.seed()   # IMPORTANT: different seed per process

    spins = initialize_spins(Lx, Ly)
    B = 1.0 / T

    mags = []

    # Equilibration
    for _ in range(equil_steps):
        spins = metropolis_step(spins, B, J)

    # Measurement
    for _ in range(measure_steps):
        spins = metropolis_step(spins, B, J)
        mags.append(np.mean(spins))

    return np.mean(np.abs(mags))


# ================================
# PARALLEL FUNCTION (ENERGY FLUCTUATION)
# ================================
def energy_for_T(T):
    np.random.seed()

    spins = initialize_spins(Lx, Ly)
    B = 1.0 / T

    energies = []

    for _ in range(equil_steps):
        spins = metropolis_step(spins, B, J)

    for _ in range(measure_steps):
        spins = metropolis_step(spins, B, J)
        energies.append(calculate_energy(spins, J))

    return np.var(energies)


# ================================
# REPRESENTATIVE STATES
# ================================
def plot_representative_states():
    T_vals = [1.0, 2.0, 2.27, 3.0, 4.0]

    plt.figure(figsize=(12, 8))

    for i, T in enumerate(T_vals):
        spins, _, _ = simulate_ising_2d(Lx, Ly, J, T, 5000)

        plt.subplot(2, 3, i+1)
        plt.imshow(spins, cmap='coolwarm')
        plt.title(f"T = {T}")
        plt.axis('off')

    plt.suptitle("Representative Spin Configurations")
    plt.show()


# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":

    # 🔹 Parallel Magnetization
    with Pool() as pool:
        M_values = pool.map(simulate_for_T, T_values)

    plt.figure()
    plt.plot(T_values, M_values, 'o-')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization (M)")
    plt.title(f"Magnetization vs Temperature (T = {T_values[0]:.1f} to {T_values[-1]:.1f})")
    plt.grid()
    plt.show()

    # 🔹 Parallel Energy Fluctuation
    with Pool() as pool:
        F_values = pool.map(energy_for_T, T_values)

    plt.figure()
    plt.plot(T_values, F_values, 'o-')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy Fluctuation")
    plt.title(f"Energy Fluctuation vs Temperature (T = {T_values[0]:.1f} to {T_values[-1]:.1f})")
    plt.grid()
    plt.show()

    # 🔹 Spin configurations
    plot_representative_states()