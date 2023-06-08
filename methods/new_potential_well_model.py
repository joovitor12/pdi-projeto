import numpy as np
import cv2

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

# Carregar a imagem
image = cv2.imread('lena.png', 0)

# Definir os parâmetros do modelo New Potential Well
noise_amplitude = 0.1
input_signal = 0.5

# Gerar a imagem processada pelo modelo New Potential Well
processed_image = new_model(image, noise_amplitude, input_signal)

# Exibir as imagens
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
