import numpy as np
import matplotlib.pyplot as plt
import cv2

def wiener_filter(noisy_img, kernel, K):
    F = np.fft.fft2(noisy_img)
    H = np.fft.fft2(kernel, s=noisy_img.shape)
    G = (np.conj(H) / (H * np.conj(H) + K)) * F
    denoised_img = np.fft.ifft2(G)
    denoised_img = np.abs(denoised_img)
    denoised_img = np.uint8(denoised_img)
    return denoised_img

def median_filter(noisy_img, kernel_size):
    noisy_img = np.uint8(noisy_img)  # Converter para np.uint8
    denoised_img = cv2.medianBlur(noisy_img, kernel_size)
    return denoised_img


def bistable_potential(x):
    return x ** 4 - x ** 2

def bistable_snr_model(x, noise_amplitude, input_signal):
    dx = -bistable_potential(x) + input_signal
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def bistable_model(original_img, noise_amplitude, input_signal):
    processed_img = np.zeros_like(original_img, dtype=float)
    
    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = bistable_snr_model(x, noise_amplitude, input_signal)
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_img.astype(np.uint8)

def new_potential_well(x, threshold):
    return np.where(x >= threshold, 1.0, -1.0)

def new_snr_model(x, noise_amplitude, input_signal):
    dx = new_potential_well(x, input_signal)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def new_model(original_img, noise_amplitude, input_signal):
    processed_img = np.zeros_like(original_img, dtype=float)
    
    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = new_snr_model(x, noise_amplitude, input_signal)
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_img.astype(np.uint8)

def bistable_potential(x):
    return x ** 4 - x ** 2

def bistable_snr_model(x, noise_amplitude, input_signal):
    dx = -bistable_potential(x) + input_signal
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def composite_potential(x):
    return x + x ** 3 - x ** 5

def composite_snr_model(x, noise_amplitude, input_signal):
    dx = -composite_potential(x) + input_signal
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def composite_model(original_img, noise_amplitude, input_signal):
    processed_img = np.zeros_like(original_img, dtype=float)
    
    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            
            if x <= 0.5:
                dx = bistable_snr_model(x, noise_amplitude, input_signal)
            else:
                dx = composite_snr_model(x, noise_amplitude, input_signal)
            
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_img.astype(np.uint8)

lena_img = cv2.imread('lena.png', 0)

# Adicionar ruído à imagem Lena
noise_amplitude = 0.1
noise = np.random.normal(loc=0, scale=noise_amplitude, size=lena_img.shape)
noisy_img = lena_img + noise

# Parâmetros do filtro de Wiener
kernel_size = 3
K = 0.01

# Parâmetros do filtro de mediana
median_kernel_size = 3

# Parâmetros do modelo bistável
bistable_noise_amplitude = 0.1
bistable_input_signal = 0.5

# Aplicar os métodos de denoização
denoised_wiener = wiener_filter(noisy_img, cv2.getGaussianKernel(kernel_size, 0), K)
denoised_median = median_filter(noisy_img, median_kernel_size)
denoised_bistable = bistable_model(noisy_img, bistable_noise_amplitude, bistable_input_signal)
denoised_new_well = new_model(noisy_img, bistable_noise_amplitude, bistable_input_signal)
denoised_composite_multistable = composite_model(noisy_img, bistable_noise_amplitude, bistable_input_signal)

# Exibir as imagens
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
fig.suptitle("Comparison of Denoising Methods")
axs[0, 0].imshow(noisy_img, cmap='gray')
axs[0, 0].set_title("Noisy Image")
axs[0, 1].imshow(denoised_wiener, cmap='gray')
axs[0, 1].set_title("Wiener Filter")
axs[1, 0].imshow(denoised_median, cmap='gray')
axs[1, 0].set_title("Median Filter")
axs[0, 2].imshow(denoised_bistable, cmap='gray')
axs[0, 2].set_title("Bistable Model")
axs[1, 1].set_title("New Potential Well Model")
axs[1, 1].imshow(denoised_new_well, cmap='gray')
axs[1,2].set_title("Composite Multistable Model")
axs[1,2].imshow(denoised_composite_multistable, cmap='gray')
plt.show()
