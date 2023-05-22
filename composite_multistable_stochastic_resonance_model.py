import numpy as np
import matplotlib.pyplot as plt

def composite_potential(x):
    return -0.1 * np.sin(4 * x) + 0.2 * np.sin(8 * x) + 0.3 * np.sin(12 * x)

def composite_snr_model(x, noise_amplitude, input_signal):
    dx = -composite_potential(x) + input_signal
    dx += noise_amplitude * np.random.normal(size=1)
    return dx

# Parameters of the model
noise_amplitude = 0.1
input_signal = 0.2

# Simulation
time = np.arange(0, 10, 0.01)
x = np.zeros(len(time))
x[0] = 0.1  # Initial condition

for i in range(1, len(time)):
    dx = composite_snr_model(x[i-1], noise_amplitude, input_signal)
    x[i] = x[i-1] + dx * 0.01  # Numerical integration (time step = 0.01)

# Plot the result
plt.plot(time, x)
plt.xlabel('Time')
plt.ylabel('System State')
plt.title('Composite Multistable Stochastic Resonance Model')
plt.show()
