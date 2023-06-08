import numpy as np
import matplotlib.pyplot as plt
import cv2

def add_salt_and_pepper_noise(image, noise_ratio, K):
    noisy_image = image.copy()
    height, width = image.shape[:2]
    num_pixels = height * width
    num_noise_pixels = int(noise_ratio * num_pixels)

    # Gerar coordenadas aleatórias para ruído
    noise_indices = np.random.choice(num_pixels, num_noise_pixels, replace=False)
    noisy_image = noisy_image.reshape(-1)  # Achatar a imagem para um vetor unidimensional

    # Adicionar ruído salt and pepper
    noisy_image[noise_indices] = 0  # Salt noise (preto)
    noisy_image[noise_indices + 1] = 255  # Pepper noise (branco)

    # Redimensionar a imagem de volta à forma original
    noisy_image = noisy_image.reshape(height, width)
    return noisy_image.astype(np.uint8)

# Função do modelo bistável
def bistable_model(image, noise_amplitude, input_signal):
    processed_image = np.zeros_like(image, dtype=float)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = bistable_snr_model(x, noise_amplitude, input_signal)
            processed_image[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_image

def bistable_snr_model(x, noise_amplitude, K):
    dx = -bistable_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def bistable_potential(x):
    return x**4 - x**2

# Função do novo modelo de poço de potencial
def new_potential_well_model(image, noise_amplitude, input_signal):
    processed_image = np.zeros_like(image, dtype=float)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = new_potential_well_snr_model(x, noise_amplitude, input_signal)
            processed_image[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_image

def new_potential_well_snr_model(x, noise_amplitude, K):
    dx = -new_potential_well_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def new_potential_well_potential(x):
    return x**3 - x

# Função do modelo composto multistável
def composite_multistable_model(image, noise_amplitude, input_signal):
    processed_image = np.zeros_like(image, dtype=float)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = composite_multistable_snr_model(x, noise_amplitude, input_signal)
            processed_image[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)
    
    return processed_image

def composite_multistable_snr_model(x, noise_amplitude, K):
    dx = -composite_multistable_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx

def composite_multistable_potential(x):
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.2 * np.sin(5 * x)


# Função para calcular a correlação cruzada
def calculate_cross_correlation(original_image, processed_image):
    cross_correlation = np.sum(original_image * processed_image) / np.sqrt(np.sum(original_image ** 2) * np.sum(processed_image ** 2))
    return cross_correlation

# Função para calcular o PSNR
def calculate_psnr(original_image, processed_image):
    mse = np.mean((original_image - processed_image) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

# Carregar imagem Lena
lena_img = cv2.imread('lena.png', 0)

# Parâmetros
kernel_size = 3
K = 0.01
noise_amplitude = 0.2
input_signal = 0.5
# Parâmetros para adicionar ruído gaussiano
mean = 0
stddev = 50

# Adicionar ruído gaussiano à imagem Lena
noisy_img = add_salt_and_pepper_noise(lena_img, mean, stddev)

# Inicializar arrays para armazenar resultados
methods = [bistable_model, new_potential_well_model, composite_multistable_model]
psnr_results = []

# Aplicar os métodos de processamento de imagem e calcular o PSNR
for method in methods:
    denoised_img = method(noisy_img, noise_amplitude, input_signal)
    psnr = calculate_psnr(lena_img, denoised_img)
    psnr_results.append(psnr)

# Criar o gráfico de linhas
plt.plot(range(len(methods)), psnr_results, marker='o')
plt.xticks(range(len(methods)), ['Bistable Model', 'New Potential Well Model', 'Composite Multistable Model'])
plt.xlabel('Models')
plt.ylabel('PSNR')
plt.title('Comparison of PSNR of Processed Images')
plt.grid(True)
plt.show()
