import numpy as np
import cv2

def wiener_filter(noisy_img, kernel, K):
    # Aplicar Transformada de Fourier na imagem ruidosa
    F = np.fft.fft2(noisy_img)

    # Aplicar Transformada de Fourier no kernel
    kernel = np.fft.fft2(kernel, s=noisy_img.shape)

    # Calcular o espectro de potência do ruído
    power_spectrum_noise = np.abs(kernel) ** 2

    # Calcular o espectro de potência do sinal original
    power_spectrum_signal = np.abs(F) ** 2

    # Calcular o espectro de potência do sinal denoizado
    power_spectrum_denoised = (1 / kernel) * power_spectrum_signal / (power_spectrum_signal + K * power_spectrum_noise)

    # Aplicar inversa da Transformada de Fourier no espectro de potência denoizado
    denoised_img = np.fft.ifft2(power_spectrum_denoised * F)

    # Retornar a parte real da imagem denoizada
    denoised_img = np.real(denoised_img)

    return denoised_img.astype(np.uint8)

# Carregar imagem ruidosa
noisy_img = cv2.imread('ohma.jpg', 0)

# Definir o kernel para o filtro Wiener
kernel_size = 3
kernel = cv2.getGaussianKernel(kernel_size, 0)

# Definir o parâmetro de regularização K
K = 0.01

# Aplicar o filtro Wiener na imagem ruidosa
denoised_img = wiener_filter(noisy_img, kernel, K)

# Exibir as imagens
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Denoised Image', denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()