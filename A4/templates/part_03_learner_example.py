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
# C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
# t_init = np.array([[-0.2, 0.3, -5.0]]).T
C_init = dcm_from_rpy([np.pi/3, -np.pi/16, -np.pi/6])
t_init = np.array([[-0.01, 10, -2]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 0.1

# Sanity check the controller output if desired.
# ...

# Run simulation - estimate depths.
gain_tuning = [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 1]
delta_t = np.zeros(len(gain_tuning))
for i, gain in enumerate(gain_tuning):
    sim_start = time.time()
    ibvs_simulation(Twc_init, Twc_last, pts, K, gain, True)
    sim_end = time.time()
    delta = sim_end - sim_start
    delta_t[i] = delta
    print("Simulation time for gain = ", gain, " is: ", delta, " seconds.")
    plt.savefig("A4/results/final_ibvs_gain_{}_estimate.png".format(gain))
plt.clf()
plt.plot(gain_tuning, delta_t)
plt.xlabel("Gain")
plt.ylabel("Simulation time (s)")
plt.title("Simulation Convergence Time vs Gain - Estimated Depths")
plt.savefig("A4/results/final_ibvs_time_estimate.png")
plt.show()

# Maximum iterations reached for gain =  0.001
# Simulation time for gain =  0.001  is:  208.8726568222046  seconds.
# Simulation time for gain =  0.01  is:  148.74938797950745  seconds.
# Simulation time for gain =  0.05  is:  36.00089383125305  seconds.
# Simulation time for gain =  0.1  is:  20.698467016220093  seconds.
# Simulation time for gain =  0.5  is:  5.258511066436768  seconds.
# Simulation time for gain =  0.7  is:  4.421133041381836  seconds.
# Simulation time for gain =  0.9  is:  3.7940430641174316  seconds.
# Maximum iterations reached for gain =  1
# Simulation time for gain =  1  is:  209.01924777030945  seconds.