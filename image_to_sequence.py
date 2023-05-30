import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('ohma.jpg', 0)  # Carregar a imagem em escala de cinza

# Converter a imagem em uma sequência unidimensional
sequence = image.flatten()

# Plot da imagem original e da sequência unidimensional
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.plot(sequence)
plt.title('Sequência Unidimensional')
plt.xlabel('Índice de Amostra')
plt.ylabel('Valor do Pixel')

plt.tight_layout()
plt.show()
