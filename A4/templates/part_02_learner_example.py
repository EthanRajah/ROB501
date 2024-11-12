import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import time
import matplotlib.pyplot as plt

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
#C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
#t_init = np.array([[-0.2, 0.3, -5.0]]).T
C_init = dcm_from_rpy([np.pi/3, -np.pi/16, -np.pi/6])
t_init = np.array([[-0.01, 10, -2]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 0.1

# Sanity check the controller output if desired.
# ...

# Run simulation - use known depths.
gain_tuning = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4]
delta_t = np.zeros(len(gain_tuning))
for i, gain in enumerate(gain_tuning):
    sim_start = time.time()
    ibvs_simulation(Twc_init, Twc_last, pts, K, gain, False)
    sim_end = time.time()
    delta = sim_end - sim_start
    delta_t[i] = delta
    print("Simulation time for gain = ", gain, " is: ", delta, " seconds.")
    plt.savefig("A4/results/final_ibvs_gain_{}_known.png".format(gain))
plt.clf()
plt.plot(gain_tuning, delta_t)
plt.xlabel("Gain")
plt.ylabel("Simulation time (s)")
plt.title("Simulation Convergence Time vs Gain - Known Depths")
plt.savefig("A4/results/final_ibvs_time_known.png")
plt.show()

# Maximum iterations reached for gain =  0.001
# Simulation time for gain =  0.001  is:  324.94776821136475  seconds.
# Simulation time for gain =  0.01  is:  267.87153911590576  seconds.
# Simulation time for gain =  0.05  is:  413.82194900512695  seconds.
# Simulation time for gain =  0.1  is:  120.58914089202881  seconds.
# Simulation time for gain =  0.5  is:  3.1354057788848877  seconds.
# Simulation time for gain =  1  is:  1.4706659317016602  seconds.
# Simulation time for gain =  1.1  is:  1.8882770538330078  seconds.
# Simulation time for gain =  1.2  is:  2.3279731273651123  seconds.
# Simulation time for gain =  1.3  is:  3.135294198989868  seconds.
# Maximum iterations reached for gain =  1.4
# Simulation time for gain =  1.4  is:  324.66734290122986  seconds.