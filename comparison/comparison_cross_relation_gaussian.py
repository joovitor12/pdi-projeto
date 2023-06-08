import numpy as np
import matplotlib.pyplot as plt
import cv2



def add_gaussian_noise(image, mean, stddev):
    noisy_image = image.copy()
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# Função de filtragem de mediana
def median_filter(image, kernel_size, K):
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

# Função de filtragem de Wiener
def wiener_filter(image, kernel_size, K):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    denoised_image = cv2.filter2D(image, -1, kernel)
    return denoised_image


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

# Função para calcular a variância
def calculate_variance(image):
    variance = np.var(image)
    return variance

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
noisy_img = add_gaussian_noise(lena_img, mean, stddev)

# Inicializar arrays para armazenar resultados
methods = [median_filter, wiener_filter, bistable_model, new_potential_well_model, composite_multistable_model]
correlation_results = []
variance_results = []

# Aplicar os métodos de processamento de imagem e calcular correlação cruzada e variância
for method in methods:
    denoised_img = method(noisy_img, kernel_size, K)
    correlation = calculate_cross_correlation(lena_img, denoised_img)
    variance = calculate_variance(denoised_img)
    correlation_results.append(correlation)
    variance_results.append(variance)

# Exibir resultados em um gráfico
plt.figure(figsize=(10, 5))
plt.bar(range(len(methods)), correlation_results, align='center', color='blue', alpha=0.5)
plt.xticks(range(len(methods)), ['Median', 'Wiener', 'Bistable', 'New Potential Well', 'Composite Multistable'])
plt.xlabel('Methods')
plt.ylabel('Cross Correlation')
plt.title('Comparison of Cross Correlation')
plt.show()
