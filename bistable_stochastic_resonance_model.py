import numpy as np
import matplotlib.pyplot as plt

def bistable_potential(x):
    return x**4 - x**2

def bistable_snr_model(x, noise_amplitude, input_signal):
    dx = -bistable_potential(x) + input_signal
    dx += noise_amplitude * np.random.normal(size=1)
    return dx

# Parâmetros do modelo
noise_amplitude = 0.1
input_signal = 0.2

# Simulação
time = np.arange(0, 10, 0.01)
x = np.zeros(len(time))
x[0] = 0.1  # Condição inicial

for i in range(1, len(time)):
    dx = bistable_snr_model(x[i-1], noise_amplitude, input_signal)
    x[i] = x[i-1] + dx * 0.01  # Integração numérica (passo de tempo = 0.01)

# Plot do resultado
plt.plot(time, x)
plt.xlabel('Tempo')
plt.ylabel('Estado do sistema')
plt.title('Modelo de ressonância estocástica bistável')
plt.show()
