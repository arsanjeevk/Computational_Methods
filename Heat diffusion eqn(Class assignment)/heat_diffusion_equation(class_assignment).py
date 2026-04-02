import numpy as np
import matplotlib.pyplot as plt

#parameters
xi, xf = 0, 1
D = 0.001

#grid
x = np.linspace(xi, xf, 100)
dx = x[1] - x[0]
dt = 0.5 * dx**2 / D
nt = int(1 / dt)


#temperature grid
T = np.zeros((len(x), nt))

#initial condition
T[1:-1,0] = 1


#time evolution
for j in range(nt - 1):
    for i in range(1, len(x) - 1):
        T[i, j+1] = T[i, j] + (D * dt / dx**2) * (
            T[i+1, j] - 2*T[i, j] + T[i-1, j]
        )

# Plot
plt.figure()
plt.plot(x, T[:, 0], label='t=0.00')
plt.plot(x, T[:, int(0.25*nt)], label='t=0.25')
plt.plot(x, T[:, int(0.5*nt)], label='t=0.50')
plt.plot(x, T[:, int(0.75*nt)], label='t=0.75')
plt.plot(x, T[:, -1], label='t=1.00')
plt.title("D = 0.001")
plt.xlabel("x")
plt.ylabel("T")
plt.legend()
plt.grid()
plt.show()
