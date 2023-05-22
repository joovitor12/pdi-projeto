import cv2
import numpy as np

def calculate_psnr(original_img, denoised_img):
    mse = np.mean((original_img - denoised_img) ** 2)
    if mse == 0:
        psnr = float('infinite')  # Definir PSNR como infinito se MSE for zero
    else:
        max_pixel = np.max(original_img)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Carregar imagens de exemplo (original e denoised)
original_img = cv2.imread('ohma.jpg', 0)  # Grayscale
denoised_img = cv2.imread('kita.jpeg', 0)  # Grayscale

# Converter as imagens para o tipo float32
original_img = original_img.astype(np.float32)
denoised_img = denoised_img.astype(np.float32)

# Calcular o PSNR
psnr = calculate_psnr(original_img, denoised_img)

# Exibir o resultado
print(f"PSNR: {psnr} dB")
