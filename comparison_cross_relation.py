import numpy as np
import matplotlib.pyplot as plt
import cv2


def add_salt_and_pepper_noise(image, noise_ratio):
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


def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


def wiener_filter(image, kernel, K):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT) + K * (image - cv2.filter2D(image, -1, kernel))


def bistable_model(original_img, noise_amplitude):
    processed_img = np.zeros_like(original_img, dtype=float)

    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = bistable_snr_model(x, noise_amplitude)
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)

    return processed_img.astype(np.uint8)


def bistable_snr_model(x, noise_amplitude):
    dx = -bistable_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx


def bistable_potential(x):
    return x**4 - x**2


def new_potential_well_model(original_img, noise_amplitude):
    processed_img = np.zeros_like(original_img, dtype=float)

    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = new_potential_well_snr_model(x, noise_amplitude)
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)

    return processed_img.astype(np.uint8)


def new_potential_well_snr_model(x, noise_amplitude):
    dx = -new_potential_well_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx


def new_potential_well_potential(x):
    return x**3 - x


def composite_multistable_model(original_img, noise_amplitude):
    processed_img = np.zeros_like(original_img, dtype=float)

    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            x = original_img[i, j] / 255.0  # Normalizar o valor do pixel entre 0 e 1
            dx = composite_multistable_snr_model(x, noise_amplitude)
            processed_img[i, j] = int(dx * 255)  # Desnormalizar e converter para inteiro (0-255)

    return processed_img.astype(np.uint8)


def composite_multistable_snr_model(x, noise_amplitude):
    dx = -composite_multistable_potential(x)
    dx += noise_amplitude * np.random.normal(size=x.shape)
    return dx


def composite_multistable_potential(x):
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.2 * np.sin(5 * x)


def calculate_cross_correlation(original_img, denoised_img):
    cross_correlation = np.sum(original_img * denoised_img) / np.sqrt(
        np.sum(original_img ** 2) * np.sum(denoised_img ** 2)
    )
    return cross_correlation


def main():
    lena_img = cv2.imread('ohma.jpg', cv2.IMREAD_GRAYSCALE)

    noise_intensities = [0.01, 0.05, 0.1, 0.2, 0.5]
    kernel_size = 3
    K = 0.01

    denoising_methods = [
        median_filter,
        wiener_filter,
        bistable_model,
        new_potential_well_model,
        composite_multistable_model
    ]

    for intensity in noise_intensities:
        noisy_img = add_salt_and_pepper_noise(lena_img, intensity)

        cross_correlation = np.zeros(len(denoising_methods))

        for i, method in enumerate(denoising_methods):
            if method == wiener_filter:
                denoised_img = method(noisy_img, np.ones((kernel_size, kernel_size)), K)
            else:
                denoised_img = method(noisy_img, kernel_size, K)
            cross_correlation[i] = calculate_cross_correlation(lena_img, denoised_img)

        print(f"Noise Intensity: {intensity}")
        print("Cross-Correlation:")
        for method, cc in zip(denoising_methods, cross_correlation):
            print(f"{method.__name__}: {cc}")
        print()

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(lena_img, cmap='gray')
        plt.title("Original")
        for i, method in enumerate(denoising_methods):
            if method == wiener_filter:
                denoised_img = method(noisy_img, np.ones((kernel_size, kernel_size)), K)
            else:
                denoised_img = method(noisy_img, kernel_size, K)
            plt.subplot(2, 3, i + 2)
            plt.imshow(denoised_img, cmap='gray')
            plt.title(method.__name__)
        plt.show()


if __name__ == "__main__":
    main()
