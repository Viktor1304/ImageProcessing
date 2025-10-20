import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

NDArrayFloat64 = npt.NDArray[np.float64]


def generate_image(n: int, k: int):
    img = np.zeros((n, n))

    for i in range(k):
        for j in range(k):
            img[n // 2 - k // 2 + i, n // 2 - k // 2 + j] = 0.3

    return img


def fourier_transform(img: NDArrayFloat64):
    output = np.zeros_like(img)

    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    output[row, col] += img[i, j] * math.cos(
                        2 * math.pi * (row * i / img.shape[0] + col * j / img.shape[1])
                    )

    return output


img = generate_image(101, 11)
cpy_img = img.copy()
# output = fourier_transform(img)

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
# (
#     plt.subplot(1, 3, 2),
#     plt.imshow(output, cmap="gray"),
#     plt.title("Fourier Transform"),
# )
# plt.show()

f_transform = np.fft.fft2(img)  # Compute 2D FFT
f_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency component to center

# Compute magnitude spectrum
magnitude_spectrum = np.log(1 + np.abs(f_shifted))
# magnitude_spectrum = np.abs(f_shifted)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.axis("off")

plt.show()
