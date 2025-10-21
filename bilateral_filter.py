from typing import TypeAlias
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import numpy.typing as npt


NDArrayUint8: TypeAlias = npt.NDArray[np.uint8]


def read_image(image_path: str):
    """
    Reads an image from the specified file path.
    Args:
        image_path (str): The path to the image file.
    Returns:
        image (numpy.ndarray): The loaded image.
    """

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image.astype(np.uint8)


def apply_bilateral_filter(
    image: NDArrayUint8,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> NDArrayUint8:
    """
    Applies a bilateral filter to the input image.
    Args:
        image (numpy.ndarray): The input image.
        diameter (int): Diameter of each pixel neighborhood.
        sigma_color (float): Filter sigma in color space.
        sigma_space (float): Filter sigma in coordinate space.
    Returns:
        filtered_image (numpy.ndarray): The filtered image.
    """

    filtered_image = cv.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image.astype(np.uint8)


def add_noise(
    image: NDArrayUint8,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> NDArrayUint8:
    """
    Adds Gaussian noise to the input image.
    Args:
        image (numpy.ndarray): The input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.
    Returns:
        noisy_image (numpy.ndarray): The noisy image.
    """

    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv.add(image, gauss)
    return noisy_image.astype(np.uint8)


if __name__ == "__main__":
    img = read_image("your_image.jpg")
    noisy_img = add_noise(img)
    filtered_img = apply_bilateral_filter(
        noisy_img, diameter=9, sigma_color=75, sigma_space=75
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(cv.cvtColor(noisy_img.astype(np.uint8), cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Filtered Image")
    plt.imshow(cv.cvtColor(filtered_img.astype(np.uint8), cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
