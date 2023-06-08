import numpy as np
import cv2

def median_filter(image, kernel_size):
    # Obter as dimensões da imagem
    height, width = image.shape

    # Definir o tamanho do padding
    padding = kernel_size // 2

    # Criar uma cópia da imagem para aplicar o filtro
    filtered_image = np.copy(image)

    # Aplicar o filtro de mediana
    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            # Extrair a vizinhança do pixel
            neighborhood = image[i - padding : i + padding + 1, j - padding : j + padding + 1]

            # Calcular o valor da mediana
            median_value = np.median(neighborhood)

            # Definir o valor do pixel filtrado como a mediana
            filtered_image[i, j] = median_value

    return filtered_image.astype(np.uint8)

# Carregar a imagem
image = cv2.imread('lena.png', 0)

# Definir o tamanho do kernel
kernel_size = 3

# Aplicar o filtro de mediana
filtered_image = median_filter(image, kernel_size)

# Exibir as imagens
cv2.imshow('Input Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
