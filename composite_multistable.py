import numpy as np
import cv2

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

# Carregar a imagem
image = cv2.imread('ohma.jpg', 0)

# Definir os parâmetros do modelo Composite Multistable
noise_amplitude = 0.1
input_signal = 0.5

# Gerar a imagem processada pelo modelo Composite Multistable
processed_image = composite_model(image, noise_amplitude, input_signal)

# Exibir as imagens
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
