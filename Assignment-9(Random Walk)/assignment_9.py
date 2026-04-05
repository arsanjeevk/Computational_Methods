#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#parameters
N = 100
l = 1
p = 0.5
walkers = 10000

#1d random walk simulation
def random_walk_simulation_1d(walkers, steps, p):
    #final positions of each walker
    m = []  
    for i in range(walkers):
        x = 0
        for j in range(steps):
            if np.random.rand() < p:
                x -= l
            else:
                x += l
        m.append(x)
    return m

#Calculating mean and mean square of final positions
def statistics_calculations(m, N):
    m = np.array(m)
    mean_x = np.mean(m)
    mean_x2 = np.mean(m**2)
    error = abs(mean_x2 - N)
    #print results
    print("⟨x⟩ =", mean_x)
    print("⟨x²⟩ =", mean_x2)
    print("Theoretical ⟨x⟩ = 0")
    print("Theoretical ⟨x²⟩ =", N)
    print("Error in ⟨x²⟩ =", error)

#question1

m1 = random_walk_simulation_1d(walkers, N, p)

#histogram of final positions
plt.hist(m1, bins=50, edgecolor='black')
plt.xlabel("Final Position")
plt.ylabel("Frequency")
plt.title("Distribution of Final Positions")
plt.show()

#results
results = statistics_calculations(m1, N)


#question2
# dob = 06/07/2006
ddd = 187
pod = ddd/365

m2 = random_walk_simulation_1d(walkers, N , pod)
result = statistics_calculations(m2, N)

#error analysis for different ensemble sizes
ensemble_sizes = [10, 100, 500, 1000, 5000, 10000]
errors = []

for walkers in ensemble_sizes:
    m = random_walk_simulation_1d(walkers, N, p)
    mean_x2 = np.mean(np.array(m)**2)
    error = abs(mean_x2 - N)
    errors.append(error)

# PLOT
plt.plot(ensemble_sizes, errors, marker='o')
plt.xscale("log")
plt.xlabel("Ensemble Size")
plt.ylabel("Error in ⟨x²⟩")
plt.title("Error vs Ensemble Size (p = 0.5)")
plt.grid(True)
plt.show()


#setup 2

#random walk 2d simulation setup
def random_walk_simulation_2d(walkers, steps):
    for w in range(walkers):
        x, y = 0, 0
        x_traj = [x]
        y_traj = [y]

        for i in range(N):
            r = np.random.rand()

            if r < 0.25:    #left
                x -= 1
            elif r < 0.5:   #right
                x += 1
            elif r < 0.75:  #down
                y -= 1
            else:           #up
                y += 1

            x_traj.append(x)
            y_traj.append(y)

        #plot trajectory
        plt.plot(x_traj, y_traj, marker='o', markersize=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Random Walk Trajectories")
plt.grid(True)
plt.axis("equal")
plt.show()
