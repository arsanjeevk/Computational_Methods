import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  RK4 SOLVER
# ══════════════════════════════════════════════════════════════════════════════
def rk4(f, x0, y0, h, n):
    x = x0
    y = np.array(y0)
    x_values = [x0]
    y_values = [y0]
    for i in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2)*k1)
        k3 = f(x + h/2, y + (h/2)*k2)
        k4 = f(x + h,   y + h*k3)
        y  = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x += h
        x_values.append(x)
        y_values.append(y.copy())
    return np.array(x_values), np.array(y_values)


# ══════════════════════════════════════════════════════════════════════════════
#  OU PROCESS GENERATOR  (Euler-Maruyama)
#
#  Equation : dx/dt = -(1/tau)*x(t) + sqrt(2/tau)*W(t)
#  Discrete  : eta[i+1] = eta[i]
#                       - (1/tau)*eta[i]*dt
#                       + sigma*sqrt(2/tau)*sqrt(dt)*N(0,1)
# ══════════════════════════════════════════════════════════════════════════════
def generate_OU(tau, sigma, h, n):
    eta         = np.zeros(n + 1)
    eta[0]      = 0.0
    noise_coeff = sigma * np.sqrt(2.0 / tau) * np.sqrt(h)
    for i in range(n):
        eta[i+1] = eta[i] - (1.0 / tau) * eta[i] * h \
                   + noise_coeff * np.random.randn()
    return eta


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOCORRELATION  (normalised)
#
#  Theoretical OU autocorrelation : R(lag) = exp(-lag*h / tau)
# ══════════════════════════════════════════════════════════════════════════════
def autocorrelation(signal, max_lag):
    signal = signal - np.mean(signal)
    var    = np.var(signal)
    result = [1.0]
    for lag in range(1, max_lag + 1):
        c = np.mean(signal[:len(signal)-lag] * signal[lag:]) / var
        result.append(c)
    return np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
m       = 1
k       = 20
c       = 1                          # underdamped : c < 2*sqrt(m*k) ≈ 8.94
omega_0 = np.sqrt(k / m)
gamma   = c / (2 * m)
omega_d = np.sqrt(omega_0**2 - gamma**2)

y0 = [1.0, 0.0]                     # initial conditions : x(0)=1, v(0)=0
t0 = 0.0
h  = 0.05
n  = 600                             # 600 steps → t in [0, 30]

# ══════════════════════════════════════════════════════════════════════════════
#  THREE (tau, sigma) CASES
# ══════════════════════════════════════════════════════════════════════════════
tau_sigma_cases = {
    "tau=0.05, sigma=2":       (0.05, 2.0),
    "tau=0.5,  sigma=sqrt(2)": (0.5,  np.sqrt(2)),
    "tau=2.0,  sigma=1":       (2.0,  1.0),
}

# CSV column names
columns = [
    "time", "position", "velocity", "acceleration",
    "KE", "PE", "Total_E",
    "F_spring", "F_damping", "F_driving", "F_total", "F_OU"
]

# autocorrelation settings
max_lag   = 200
lag_times = np.arange(max_lag + 1) * h


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP  — one full run per (tau, sigma) case
# ══════════════════════════════════════════════════════════════════════════════
for label, (tau, sigma) in tau_sigma_cases.items():

    print(f"\n{'='*60}")
    print(f"  Underdamped | OU Forcing | {label}")
    print(f"  tau={tau},  sigma={sigma:.4f}")
    print(f"  gamma={gamma:.4f},  omega_d={omega_d:.4f}")
    print(f"{'='*60}")

    # ── 1. Generate OU driving force ──────────────────────────────────────────
    F_OU = generate_OU(tau, sigma, h, n)

    # ── 2. Define ODE and solve with RK4 ─────────────────────────────────────
    #       m*x'' + c*x' + k*x = F_OU(t)
    def f(t_curr, y, F_OU=F_OU, h=h):
        idx     = min(int(round(t_curr / h)), len(F_OU) - 1)
        F_drive = F_OU[idx]
        y1, y2  = y
        return np.array([y2, (F_drive - c*y2 - k*y1) / m])

    t, y = rk4(f, t0, y0, h, n)

    # ── 3. Extract quantities ─────────────────────────────────────────────────
    position     = y[:, 0]
    velocity     = y[:, 1]
    F_driving    = F_OU
    acceleration = (F_driving - c*velocity - k*position) / m

    KE      = 0.5 * m * velocity**2
    PE      = 0.5 * k * position**2
    Total_E = KE + PE

    F_spring  = -k * position
    F_damping = -c * velocity
    F_total   = m * acceleration

    # ── 4. Save CSV ───────────────────────────────────────────────────────────
    data_matrix = np.column_stack([
        t, position, velocity, acceleration,
        KE, PE, Total_E,
        F_spring, F_damping, F_driving, F_total, F_OU
    ])
    df         = pd.DataFrame(data_matrix, columns=columns)
    safe_label = (label.replace(' ', '').replace(',', '')
                       .replace('=', '').replace('(', '')
                       .replace(')', '').replace('/', ''))
    csv_name   = f"Underdamped_OU_{safe_label}.csv"
    df.to_csv(csv_name, index=False)
    print(df.head())
    print(f"  CSV saved → {csv_name}\n")

    # ── 5. Plot : Position, Velocity, Acceleration (3 separate subplots) ──────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Underdamped Oscillator — OU Forcing\n{label}", fontsize=13)

    axes[0].plot(t, position, color='steelblue', lw=1.2)
    axes[0].set_ylabel("Position $x(t)$")
    axes[0].set_title("Position vs Time")
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, velocity, color='darkorange', lw=1.2)
    axes[1].set_ylabel("Velocity $\\dot{x}(t)$")
    axes[1].set_title("Velocity vs Time")
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(t, acceleration, color='crimson', lw=1.2)
    axes[2].set_xlabel("Time $t$")
    axes[2].set_ylabel("Acceleration $\\ddot{x}(t)$")
    axes[2].set_title("Acceleration vs Time")
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # ── 6. Plot : Energy ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, KE,      label="KE")
    ax.plot(t, PE,      label="PE")
    ax.plot(t, Total_E, label="Total E")
    ax.set_title(f"Energy — OU Forcing | {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── 7. Plot : Forces ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, F_spring,  '--', label="Spring $F_s$")
    ax.plot(t, F_damping,       label="Damping $F_d$")
    ax.plot(t, F_driving,  ':', label="OU Driving $F_{OU}$", alpha=0.7)
    ax.plot(t, F_total,         label="Total $F$")
    ax.set_title(f"Forces — OU Forcing | {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Force")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── 8. Plot : Phase Space ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(position, velocity, lw=0.8, color='purple')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Phase Space — OU Forcing | {label}")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── 9. Plot : OU Force signal ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, F_OU, color='teal', lw=0.8, alpha=0.85)
    ax.set_title(f"OU Process (Driving Force) | {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\eta(t)$")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── 10. Plot : Autocorrelation of OU force ────────────────────────────────
    #        Numerical  vs  Theoretical : R(dt) = exp(-dt / tau)
    acf_numerical   = autocorrelation(F_OU, max_lag)
    acf_theoretical = np.exp(-lag_times / tau)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_times, acf_numerical,   color='steelblue', lw=1.5,
            label="Numerical ACF")
    ax.plot(lag_times, acf_theoretical, color='black',     lw=1.5,
            ls='--', label=f"Theoretical : $e^{{-\\Delta t/\\tau}}$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("Lag time $\\Delta t$")
    ax.set_ylabel("Autocorrelation $R(\\Delta t)$")
    ax.set_title(f"Autocorrelation of OU Driving Force | {label}")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()