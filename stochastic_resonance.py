import numpy as np
import cv2
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

# Processamento de imagem usando a ressonância estocástica com os parâmetros da baleia
original_img = cv2.imread('ohma.jpg', 0)  # Grayscale
noise_img = cv2.imread('kita.jpeg', 0)  # Grayscale

processed_img = np.zeros_like(original_img)
for i in range(original_img.shape[0]):
    for j in range(original_img.shape[1]):
        x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
        dx = bistable_snr_model(x, noise_amplitude, input_signal)
        processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)

# Exibir as imagens
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Imagem original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_img, cmap='gray')
plt.title('Imagem processada')
plt.axis('off')

plt.show()
