import numpy as np
import matplotlib.pyplot as plt
import cv2

# Função para denoizar a imagem
def denoise_image(original_img, method, noise_amplitude, input_signal):
    # Implemente a lógica de denoização para o método específico aqui
    denoised_img = method(original_img, noise_amplitude, input_signal)
    return denoised_img

# Função para calcular a correlação cruzada
def calculate_cross_correlation(original_img, denoised_img):
    cross_correlation = np.mean(np.correlate(original_img.flatten(), denoised_img.flatten()))
    return cross_correlation

# Função para calcular o PSNR
def calculate_psnr(original_img, denoised_img):
    mse = np.mean((original_img - denoised_img) ** 2)
    max_pixel = np.max(original_img)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

lena_img = cv2.imread('lena.png', 0)

def bistable_potential(x):
    return x**4 - x**2

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


# Lista de métodos de denoização
denoising_methods = [bistable_model]

# Lista de intensidades de ruído
noise_intensities = [0.01, 0.05, 0.1, 0.2, 0.5]

# Loop pelas intensidades de ruído
for intensity in noise_intensities:
    # Adicionar ruído à imagem Lena
    noise = np.random.normal(loc=0, scale=intensity, size=lena_img.shape)
    noisy_img = lena_img + noise

    # Inicializar arrays para armazenar resultados
    cross_correlation = np.zeros(len(denoising_methods))
    psnr_values = np.zeros(len(denoising_methods))
    denoised_imgs = []

    # Aplicar métodos de denoização e calcular correlação cruzada e PSNR
    for i, method in enumerate(denoising_methods):
        denoised_img = denoise_image(noisy_img, method, intensity, 0)  # Passar a intensidade do ruído e sinal de entrada
        denoised_imgs.append(denoised_img)
        cross_correlation[i] = calculate_cross_correlation(lena_img, denoised_img)
        psnr_values[i] = calculate_psnr(lena_img, denoised_img)

    # Exibir resultados
    print(f"Noise Intensity: {intensity}")
    print("Cross-Correlation:")
    for method, cc in zip(denoising_methods, cross_correlation):
        print(f"{method.__name__}: {cc}")
    print("PSNR:")
    for method, psnr in zip(denoising_methods, psnr_values):
        print(f"{method.__name__}: {psnr} dB")
    print()

    # Plotar as imagens denoizadas
    fig, axs = plt.subplots(2, len(denoising_methods) + 1, figsize=(10, 6))
    fig.suptitle(f"Noise Intensity: {intensity}")
    axs[0, 0].imshow(lena_img, cmap='gray')
    axs[0, 0].set_title("Original")
    for i, method in enumerate(denoising_methods):
        axs[0, i+1].imshow(denoised_imgs[i], cmap='gray')
        axs[0, i+1].set_title(method.__name__)
    for i, method in enumerate(denoising_methods):
        axs[1, i+1].imshow(noisy_img, cmap='gray')
        axs[1, i+1].set_title("Noisy")
    plt.show()
